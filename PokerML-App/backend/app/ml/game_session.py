"""
Game session management with player statistics tracking.

Manages session lifecycle and accumulates player statistics for ML features.
Stats are scoped to session_id and reset when a new session starts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class HandSummary:
    """Summary of a single hand for rolling statistics."""
    hand_number: int
    voluntary_preflop: bool = False
    preflop_raise: bool = False
    saw_flop: bool = False
    saw_turn: bool = False
    saw_river: bool = False
    went_to_showdown: bool = False
    aggressive_actions: int = 0
    passive_actions: int = 0
    stack_change: float = 0.0  # Net chips won/lost in this hand


@dataclass
class PlayerSessionStats:
    """
    Rolling statistics for a player within a session.

    These stats are used to calculate historical features like VPIP, PFR,
    aggression factor, etc. for the ML models.
    """
    player_name: str
    seat_index: int = -1

    # Cumulative counts
    hands_played: int = 0
    voluntary_preflop: int = 0
    preflop_raise: int = 0
    aggressive_actions: int = 0
    passive_actions: int = 0
    saw_flop: int = 0
    saw_turn: int = 0
    saw_river: int = 0
    showdown_count: int = 0

    # Rolling window of recent hands (for calculating last-N stats)
    recent_hands: deque = field(default_factory=lambda: deque(maxlen=10))

    def record_hand(self, summary: HandSummary) -> None:
        """Record stats from a completed hand."""
        self.hands_played += 1
        if summary.voluntary_preflop:
            self.voluntary_preflop += 1
        if summary.preflop_raise:
            self.preflop_raise += 1
        if summary.saw_flop:
            self.saw_flop += 1
        if summary.saw_turn:
            self.saw_turn += 1
        if summary.saw_river:
            self.saw_river += 1
        if summary.went_to_showdown:
            self.showdown_count += 1
        self.aggressive_actions += summary.aggressive_actions
        self.passive_actions += summary.passive_actions

        self.recent_hands.append(summary)
        logger.info(
            f"STATS RECORDED for {self.player_name}: hands={self.hands_played}, "
            f"vpip={self.voluntary_preflop}, pfr={self.preflop_raise}, "
            f"agg={self.aggressive_actions}, pas={self.passive_actions}, "
            f"summary: voluntary={summary.voluntary_preflop}, raise={summary.preflop_raise}"
        )

    def vpip(self, window: int = 10) -> float:
        """
        Voluntarily Put $ In Pot percentage.

        The percentage of hands where the player voluntarily put money in
        the pot preflop (called or raised, not just posted blinds).
        """
        recent = list(self.recent_hands)[-window:]
        if not recent:
            return 0.5  # Default for unknown players
        return sum(1 for h in recent if h.voluntary_preflop) / len(recent)

    def pfr(self, window: int = 10) -> float:
        """
        Preflop Raise percentage.

        The percentage of hands where the player raised preflop.
        """
        recent = list(self.recent_hands)[-window:]
        if not recent:
            return 0.3  # Default
        return sum(1 for h in recent if h.preflop_raise) / len(recent)

    def aggression_factor(self, window: int = 10) -> float:
        """
        Aggression factor = (bets + raises) / calls.

        Higher values indicate more aggressive play.
        """
        recent = list(self.recent_hands)[-window:]
        if not recent:
            return 1.0  # Default
        agg = sum(h.aggressive_actions for h in recent)
        pas = sum(h.passive_actions for h in recent)
        return agg / (pas + 1e-6)

    def stickiness(self, window: int = 10) -> float:
        """
        Stickiness / Saw Flop percentage.

        How often the player continues to the flop.
        """
        recent = list(self.recent_hands)[-window:]
        if not recent:
            return 0.5  # Default
        return sum(1 for h in recent if h.saw_flop) / len(recent)

    def stack_trend(self, window: int = 10) -> float:
        """
        Stack trend over recent hands.

        Average stack change per hand. Positive = winning, negative = losing.
        """
        recent = list(self.recent_hands)[-window:]
        if not recent:
            return 0.0  # Default - no trend
        return sum(h.stack_change for h in recent) / len(recent)

    def get_historical_features(self) -> Dict[str, float]:
        """Get all historical features for ML model input.

        IMPORTANT: V3 models expect VPIP, PFR, and street_adv as percentages (0-100),
        NOT as ratios (0-1). The model was trained on data where VPIP=50 means 50%.

        Note: Different models use different naming conventions:
        - Opponent model (03_OpponentModeling): uses 'street_adv_last*_hist'
        - Profit/Policy models (05/06): use 'stickiness_last*_hist' (from sp4_features)
        We provide BOTH to ensure compatibility with all Databricks endpoints.
        """
        # Pre-calculate values and convert to percentages (0-100 scale)
        # The individual methods return ratios (0-1), so multiply by 100
        stick3 = float(self.stickiness(3)) * 100.0
        stick5 = float(self.stickiness(5)) * 100.0
        stick10 = float(self.stickiness(10)) * 100.0

        vpip5_raw = self.vpip(5)
        pfr5_raw = self.pfr(5)
        logger.info(
            f"GET_HISTORICAL_FEATURES for {self.player_name}: "
            f"recent_hands={len(self.recent_hands)}, hands_played={self.hands_played}, "
            f"vpip5_raw={vpip5_raw}, pfr5_raw={pfr5_raw}, "
            f"vpip5_pct={vpip5_raw*100:.1f}%, pfr5_pct={pfr5_raw*100:.1f}%"
        )

        return {
            # VPIP: percentage 0-100 (e.g., 50 means player plays 50% of hands)
            "vpip_last3_hist": float(self.vpip(3)) * 100.0,
            "vpip_last5_hist": float(self.vpip(5)) * 100.0,
            "vpip_last10_hist": float(self.vpip(10)) * 100.0,
            # PFR: percentage 0-100 (e.g., 30 means player raises 30% preflop)
            "pfr_last3_hist": float(self.pfr(3)) * 100.0,
            "pfr_last5_hist": float(self.pfr(5)) * 100.0,
            "pfr_last10_hist": float(self.pfr(10)) * 100.0,
            # Aggression factor: ratio (0-10+), stays as-is
            "agg_factor_last3_hist": float(self.aggression_factor(3)),
            "agg_factor_last5_hist": float(self.aggression_factor(5)),
            "agg_factor_last10_hist": float(self.aggression_factor(10)),
            # Street advancement: percentage 0-100
            "street_adv_last3_hist": stick3,
            "street_adv_last5_hist": stick5,
            "street_adv_last10_hist": stick10,
            # Stickiness alias (same as street_adv)
            "stickiness_last3_hist": stick3,
            "stickiness_last5_hist": stick5,
            "stickiness_last10_hist": stick10,
            # Stack trend features (average stack change per hand, stays as-is)
            "stack_trend_last3_hist": float(self.stack_trend(3)),
            "stack_trend_last5_hist": float(self.stack_trend(5)),
            "stack_trend_last10_hist": float(self.stack_trend(10)),
        }


@dataclass
class HandTracker:
    """Tracks actions within a single hand for stats accumulation."""
    hand_number: int
    player_summaries: Dict[str, HandSummary] = field(default_factory=dict)
    action_count: int = 0
    raise_count: int = 0
    call_count: int = 0
    current_street: str = "preflop"

    def get_or_create_summary(self, player_name: str) -> HandSummary:
        """Get or create a hand summary for a player."""
        if player_name not in self.player_summaries:
            self.player_summaries[player_name] = HandSummary(hand_number=self.hand_number)
        return self.player_summaries[player_name]

    def record_action(
        self,
        player_name: str,
        action_type: str,
        street: str,
        amount: float = 0,
    ) -> None:
        """Record a player action."""
        self.action_count += 1
        self.current_street = street
        summary = self.get_or_create_summary(player_name)

        # Track street progression
        if street == "flop":
            summary.saw_flop = True
        elif street == "turn":
            summary.saw_flop = True
            summary.saw_turn = True
        elif street == "river":
            summary.saw_flop = True
            summary.saw_turn = True
            summary.saw_river = True

        # Track action types
        if action_type == "bet_or_raise_to":
            summary.aggressive_actions += 1
            self.raise_count += 1
            if street == "preflop":
                summary.voluntary_preflop = True
                summary.preflop_raise = True
        elif action_type == "call_or_check":
            # Check if it's a call (has amount) vs check (no amount)
            if amount > 0:
                summary.passive_actions += 1
                self.call_count += 1
                if street == "preflop":
                    summary.voluntary_preflop = True
            # Checks don't count as passive actions for aggression factor
        # Folds don't affect stats

    def record_showdown(self, player_names: List[str]) -> None:
        """Record which players went to showdown."""
        for name in player_names:
            summary = self.get_or_create_summary(name)
            summary.went_to_showdown = True


@dataclass
class MLPredictionRecord:
    """Record of an ML prediction for review."""
    hand_number: int
    street: str
    action_index: int
    prediction_type: str  # opponent, profit, policy
    target_seat: Optional[int]
    hero_seat: int
    features: Dict[str, float]
    prediction: Dict[str, Any]
    latency_ms: float
    from_fallback: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GameSession:
    """
    A game session representing a continuous play period.

    Stats are accumulated across hands within a session.
    When session_id changes (new game, refresh), stats reset.
    """
    session_id: str
    table_id: str
    hero_seat: int
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Player statistics (keyed by player name)
    player_stats: Dict[str, PlayerSessionStats] = field(default_factory=dict)

    # Current hand tracking
    current_hand: Optional[HandTracker] = None
    current_hand_number: int = 0

    # ML predictions for this session (for review)
    predictions: List[MLPredictionRecord] = field(default_factory=list)

    def start_hand(self, hand_number: int) -> HandTracker:
        """Start tracking a new hand."""
        # Finalize previous hand if exists
        if self.current_hand is not None:
            self._finalize_hand()

        self.current_hand_number = hand_number
        self.current_hand = HandTracker(hand_number=hand_number)
        logger.debug(f"Session {self.session_id}: Started hand {hand_number}")
        return self.current_hand

    def record_action(
        self,
        player_name: str,
        action_type: str,
        street: str,
        amount: float = 0,
        seat_index: int = -1,
    ) -> None:
        """Record a player action in the current hand."""
        if self.current_hand is None:
            logger.warning("No current hand to record action")
            return

        # Ensure player stats exist
        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerSessionStats(
                player_name=player_name,
                seat_index=seat_index,
            )

        self.current_hand.record_action(player_name, action_type, street, amount)
        logger.info(
            f"ACTION RECORDED: player={player_name}, action={action_type}, street={street}, amount={amount}"
        )

    def end_hand(self, showdown_players: Optional[List[str]] = None) -> None:
        """End the current hand and update stats."""
        if self.current_hand is None:
            return

        if showdown_players:
            self.current_hand.record_showdown(showdown_players)

        self._finalize_hand()

    def _finalize_hand(self) -> None:
        """Finalize hand and update player stats."""
        if self.current_hand is None:
            return

        for player_name, summary in self.current_hand.player_summaries.items():
            if player_name in self.player_stats:
                self.player_stats[player_name].record_hand(summary)
            else:
                stats = PlayerSessionStats(player_name=player_name)
                stats.record_hand(summary)
                self.player_stats[player_name] = stats

        logger.info(
            f"HAND FINALIZED: session={self.session_id}, hand={self.current_hand.hand_number}, "
            f"players_in_hand={list(self.current_hand.player_summaries.keys())}"
        )
        for player_name, summary in self.current_hand.player_summaries.items():
            logger.info(
                f"  HAND SUMMARY for {player_name}: voluntary={summary.voluntary_preflop}, "
                f"raised={summary.preflop_raise}, saw_flop={summary.saw_flop}, "
                f"aggressive={summary.aggressive_actions}, passive={summary.passive_actions}"
            )
        self.current_hand = None

    def get_player_stats(self, player_name: str) -> PlayerSessionStats:
        """Get stats for a player, creating if needed."""
        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerSessionStats(player_name=player_name)
        return self.player_stats[player_name]

    def add_prediction(self, record: MLPredictionRecord) -> None:
        """Add a prediction record for later review."""
        self.predictions.append(record)

    def get_predictions_for_hand(self, hand_number: int) -> List[MLPredictionRecord]:
        """Get all predictions for a specific hand."""
        return [p for p in self.predictions if p.hand_number == hand_number]

    def get_action_count(self) -> int:
        """Get current action count in hand."""
        return self.current_hand.action_count if self.current_hand else 0

    def get_raise_count(self) -> int:
        """Get raise count in current hand."""
        return self.current_hand.raise_count if self.current_hand else 0

    def get_call_count(self) -> int:
        """Get call count in current hand."""
        return self.current_hand.call_count if self.current_hand else 0


class GameSessionManager:
    """
    Manages game sessions.

    Each session_id maps to a GameSession with isolated player stats.
    """

    def __init__(self):
        self._sessions: Dict[str, GameSession] = {}

    def create_session(
        self,
        session_id: str,
        table_id: str,
        hero_seat: int,
    ) -> GameSession:
        """Create a new game session."""
        session = GameSession(
            session_id=session_id,
            table_id=table_id,
            hero_seat=hero_seat,
        )
        self._sessions[session_id] = session
        logger.info(f"Created session {session_id} for table {table_id}")
        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        table_id: str,
        hero_seat: int,
    ) -> GameSession:
        """Get existing session or create new one."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(session_id, table_id, hero_seat)

    def end_session(self, session_id: str) -> Optional[GameSession]:
        """End and remove a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            # Finalize any ongoing hand
            session.end_hand()
            logger.info(f"Ended session {session_id}")
        return session

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    @staticmethod
    def generate_session_id() -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())
