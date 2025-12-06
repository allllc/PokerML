"""
Feature calculation engine for ML models (V3 Schema).
Updated: 2025-12-03 - V3 models with sklearn Pipeline (StandardScaler built-in).

Calculates real-time features from game state matching the EXACT schemas
required by the Databricks v3 model serving endpoints.

Key V3 Changes:
- StandardScaler is built into the Pipeline - pass RAW features directly
- Chip amounts must be normalized to Big Blinds (BB) before passing to model
- Features that need BB normalization: pot_size, amount, starting_stack, etc.

EXACT Feature Schemas (from Databricks endpoint signatures):
- Opponent model (01): 25-28 features (varies by street - preflop has no board features)
- Profit model (02): EXACTLY 20 features
- Policy model (03): EXACTLY 43 features
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging
import statistics

from .game_session import GameSession, PlayerSessionStats

logger = logging.getLogger(__name__)

# Street rank mapping
STREET_RANKS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}

# Position bucket values for one-hot encoding (policy model)
POSITION_BUCKETS = ["BB", "BTN", "CO", "HJ", "MP", "SB", "UTG", "UTG+1"]

# Position count values for one-hot encoding (profit model)
POSITION_COUNTS = ["BB", "BTN", "CO", "HJ", "MP", "SB", "UTG", "UTG+1"]

# Predicted bucket values for one-hot encoding (policy model)
PREDICTED_BUCKETS = ["air", "middle", "nutted"]

# ============================================================================
# CANONICAL FEATURE ORDERS (EXACT ORDER FROM DATABRICKS TRAINING)
# These MUST match the order used when training the models
# ============================================================================

# Profit Model (02) - EXACTLY 63 features in this EXACT order
PROFIT_FEATURES_ORDER = [
    # === Core features (20) ===
    'pot_size', 'hand_rank', 'starting_stack', 'stack_vs_table_median',
    'position_from_button', 'num_players', 'hand_equity',
    'board_pair_or_better', 'board_flush_possible', 'board_straight_possible',
    'hole_pair_flag', 'has_flush_draw_flag', 'has_straight_draw_flag',
    'bet_pct_pot', 'action_no_in_hand', 'raises_so_far', 'calls_so_far',
    'vpip_last3_hist', 'pfr_last3_hist', 'street_adv_last3_hist',

    # === Additional historical stats (12) ===
    'agg_factor_last3_hist', 'stack_trend_last3_hist',
    'vpip_last5_hist', 'pfr_last5_hist', 'street_adv_last5_hist',
    'agg_factor_last5_hist', 'stack_trend_last5_hist',
    'vpip_last10_hist', 'pfr_last10_hist', 'street_adv_last10_hist',
    'agg_factor_last10_hist', 'stack_trend_last10_hist',

    # === Table dynamics (6) ===
    'players_at_table', 'players_active', 'opponents_active',
    'avg_opponent_stack', 'max_opponent_stack', 'min_opponent_stack',

    # === Position one-hot (8) ===
    'pos_count_BB', 'pos_count_BTN', 'pos_count_CO', 'pos_count_HJ',
    'pos_count_MP', 'pos_count_SB', 'pos_count_UTG', 'pos_count_UTG+1',

    # === Board texture advanced (5) ===
    'board_monotone', 'board_paired', 'board_straighty',
    'board_flush_pressure', 'board_straight_pressure',

    # === Opponent predictions from SP-03 (8) ===
    'predicted_strength', 'opponent_strength_mean', 'opponent_strength_max',
    'opponent_strength_min', 'opponent_air_count', 'opponent_middle_count',
    'opponent_nutted_count', 'has_opponent_predictions',

    # === Pot/action context (4) ===
    'pot_before_action', 'facing_call', 'pot_odds_call', 'bb',
]

# Opponent Model (01) - Preflop: 25 features, Postflop: 28 features
OPPONENT_FEATURES_ORDER_PREFLOP = [
    'position_from_button', 'num_players',
    'pot_size', 'amount', 'action_no_in_hand', 'raises_so_far', 'calls_so_far',
    'starting_stack', 'stack_vs_table_median',
    'vpip_last3_hist', 'vpip_last5_hist', 'vpip_last10_hist',
    'pfr_last3_hist', 'pfr_last5_hist', 'pfr_last10_hist',
    'agg_factor_last3_hist', 'agg_factor_last5_hist', 'agg_factor_last10_hist',
    'street_adv_last3_hist', 'street_adv_last5_hist', 'street_adv_last10_hist',
    'stack_trend_last3_hist', 'stack_trend_last5_hist', 'stack_trend_last10_hist',
    'bet_pct_pot',
]

OPPONENT_FEATURES_ORDER_POSTFLOP = OPPONENT_FEATURES_ORDER_PREFLOP + [
    'board_pair_or_better', 'board_flush_possible', 'board_straight_possible',
]

# Policy Model (03) - EXACTLY 44 features
POLICY_FEATURES_ORDER = [
    # === Stack & Position (4) ===
    'starting_stack', 'stack_vs_table_median', 'position_from_button', 'num_players',

    # === Hand Strength (1) ===
    'hand_equity',

    # === Board Texture (3) ===
    'board_pair_or_better', 'board_flush_possible', 'board_straight_possible',

    # === Draw Flags (3) ===
    'hole_pair_flag', 'has_flush_draw_flag', 'has_straight_draw_flag',

    # === Historical Stats (9) ===
    'vpip_last3_hist', 'pfr_last3_hist', 'agg_factor_last3_hist',
    'vpip_last5_hist', 'pfr_last5_hist', 'agg_factor_last5_hist',
    'vpip_last10_hist', 'pfr_last10_hist', 'agg_factor_last10_hist',

    # === Opponent Predictions (7) ===
    'predicted_strength', 'opponent_strength_mean', 'opponent_strength_max',
    'opponent_strength_min', 'opponent_air_count', 'opponent_middle_count',
    'opponent_nutted_count',

    # === Strength Context (3) ===
    'strength_advantage', 'strength_disadvantage', 'strength_strong',

    # === Opponent Count Flags (3) ===
    'heads_up', 'three_way', 'multiway',

    # === One-Hot Encoded Position (8) ===
    'position_bucket_CO', 'position_bucket_UTG+1', 'position_bucket_BTN',
    'position_bucket_UTG', 'position_bucket_BB', 'position_bucket_MP',
    'position_bucket_SB', 'position_bucket_HJ',

    # === One-Hot Encoded Predicted Bucket (3) ===
    'predicted_bucket_middle', 'predicted_bucket_air', 'predicted_bucket_nutted',
]

# Card rank values for hand strength and board texture
RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14
}

# ============================================================================
# V2 FEATURE SCHEMAS (from Databricks notebooks)
# ============================================================================

# Opponent Model (01) - from 03_OpponentModeling.ipynb
OPPONENT_POSITION_FEATURES = ['position_from_button', 'num_players']

OPPONENT_BASE_FEATURES = [
    'pot_size', 'amount', 'action_no_in_hand', 'raises_so_far', 'calls_so_far',
    'starting_stack', 'stack_vs_table_median',
    'vpip_last3_hist', 'vpip_last5_hist', 'vpip_last10_hist',
    'pfr_last3_hist', 'pfr_last5_hist', 'pfr_last10_hist',
    'agg_factor_last3_hist', 'agg_factor_last5_hist', 'agg_factor_last10_hist',
    'street_adv_last3_hist', 'street_adv_last5_hist', 'street_adv_last10_hist',
    'stack_trend_last3_hist', 'stack_trend_last5_hist', 'stack_trend_last10_hist'
]

OPPONENT_BET_FEATURES = ['bet_pct_pot']

OPPONENT_BOARD_FEATURES = ['board_pair_or_better', 'board_flush_possible', 'board_straight_possible']

# Policy Model (03) - from 09_PolicyTraining.ipynb (STRICT NO-LEAKAGE)
POLICY_ALLOWED_FEATURES = [
    # Hand strength (core decision input)
    'hand_equity',

    # Position (strategic factor)
    'position_from_button',

    # Table context
    'num_players',
    'starting_stack',
    'stack_vs_table_median',

    # Opponent modeling (from model 01)
    'predicted_strength',
    'opponent_strength_mean',
    'opponent_strength_max',
    'opponent_strength_min',
    'opponent_air_count',
    'opponent_middle_count',
    'opponent_nutted_count',

    # Player historical tendencies
    'vpip_last3_hist', 'vpip_last5_hist', 'vpip_last10_hist',
    'pfr_last3_hist', 'pfr_last5_hist', 'pfr_last10_hist',
    'agg_factor_last3_hist', 'agg_factor_last5_hist', 'agg_factor_last10_hist',

    # Board texture (postflop)
    'board_pair_or_better',
    'board_flush_possible',
    'board_straight_possible',

    # Draw flags
    'hole_pair_flag',
    'has_flush_draw_flag',
    'has_straight_draw_flag',

    # Strength comparison
    'strength_advantage',
    'strength_disadvantage',
    'strength_strong',

    # Multiway indicators
    'heads_up',
    'three_way',
    'multiway',
]


def parse_card(card_str: str) -> Tuple[str, str]:
    """Parse card string to (rank, suit)."""
    if len(card_str) != 2:
        return ("", "")
    return (card_str[0].upper(), card_str[1].lower())


def get_rank_value(rank: str) -> int:
    """Get numeric value for a rank."""
    return RANK_VALUES.get(rank.upper(), 0)


def get_position_from_button(seat_index: int, button_seat: int, num_players: int) -> int:
    """
    Calculate position relative to button.

    Returns:
        0 = Button (best position)
        1 = Small Blind
        2 = Big Blind
        3+ = Earlier positions (worse)
    """
    if num_players <= 0:
        return 0
    return (seat_index - button_seat) % num_players


class FeatureEngine:
    """
    Calculates ML features from game state (V2 Schema).

    Supports three feature sets matching Databricks v2 endpoints:
    1. Opponent Model Features (01): 26-29 features depending on street
    2. Profit Model Features (02): ~30+ features
    3. Policy Model Features (03): Conservative set (~25 features, no leakage)
    """

    def __init__(self):
        """Initialize the feature engine."""
        # Optional: treys evaluator for hand strength
        self._evaluator = None
        try:
            from treys import Evaluator
            self._evaluator = Evaluator()
            logger.info("Treys evaluator loaded for hand strength calculation")
        except ImportError:
            logger.warning("Treys not available - using simplified hand strength")

    def calculate_opponent_features(
        self,
        pot_size: float,
        amount: float,
        action_type: str,
        street: str,
        session: GameSession,
        player_name: str,
        player_stack: float,
        all_stacks: List[float],
        pot_contribution: float,
        board_cards: List[str],
        position_from_button: int = 0,
        num_players: int = 6,
        raises_override: Optional[int] = None,
        calls_override: Optional[int] = None,
        bb: float = 2.0,
    ) -> Dict[str, float]:
        """
        Calculate features for Opponent Strength Classifier (Model 01).

        Matches the V3 schema from 03_OpponentModeling.ipynb.
        V3 models have StandardScaler built-in, so we pass raw features.
        Chip amounts are normalized to Big Blinds (BB).

        Args:
            pot_size: Current pot size (in chips)
            amount: Current action amount (in chips)
            action_type: One of 'fold', 'call_or_check', 'bet_or_raise_to'
            street: Current street
            session: Game session for historical stats
            player_name: Name of the player to predict
            player_stack: Player's starting stack (in chips)
            all_stacks: All players' starting stacks (in chips)
            pot_contribution: Player's total pot contribution this hand
            board_cards: List of board cards (e.g., ["Ah", "Kd", "7c"])
            position_from_button: Position relative to button (0=BTN, 1=SB, etc.)
            num_players: Number of players at table
            bb: Big blind size for BB normalization

        Returns:
            Feature dictionary matching V3 model input schema (22-25 features)
        """
        # Get historical stats for this player
        stats = session.get_player_stats(player_name)
        hist_features = stats.get_historical_features()

        # Normalize chip values to BB (V3 requirement)
        bb_safe = max(bb, 1.0)  # Avoid division by zero
        pot_size_bb = float(pot_size / bb_safe)
        amount_bb = float(amount / bb_safe)
        player_stack_bb = float(player_stack / bb_safe)

        # Calculate base features - ALL features must be float for model compatibility
        median_stack = statistics.median(all_stacks) if all_stacks else player_stack
        stack_vs_median = float(player_stack / (median_stack + 1e-6))
        bet_pct_pot = float(amount / (pot_size + 1e-6))

        # Use override values if provided, otherwise fall back to session
        raises_so_far = raises_override if raises_override is not None else session.get_raise_count()
        calls_so_far = calls_override if calls_override is not None else session.get_call_count()

        # Build features matching V3 schema from 03_OpponentModeling.ipynb
        # V3 uses BB-normalized chip values
        features = {
            # Position features
            "position_from_button": float(position_from_button),
            "num_players": float(num_players),

            # Base features (BB-normalized for V3)
            "pot_size": pot_size_bb,
            "amount": amount_bb,
            "action_no_in_hand": float(session.get_action_count() + 1),
            "raises_so_far": float(raises_so_far),
            "calls_so_far": float(calls_so_far),
            "starting_stack": player_stack_bb,
            "stack_vs_table_median": stack_vs_median,

            # Historical features from session stats
            # NOTE: V3 models expect VPIP, PFR, street_adv as percentages (0-100)
            # Default values: VPIP ~50%, PFR ~30%, street_adv ~50% for unknown players
            "vpip_last3_hist": float(hist_features.get("vpip_last3_hist", 50.0)),
            "vpip_last5_hist": float(hist_features.get("vpip_last5_hist", 50.0)),
            "vpip_last10_hist": float(hist_features.get("vpip_last10_hist", 50.0)),
            "pfr_last3_hist": float(hist_features.get("pfr_last3_hist", 30.0)),
            "pfr_last5_hist": float(hist_features.get("pfr_last5_hist", 30.0)),
            "pfr_last10_hist": float(hist_features.get("pfr_last10_hist", 30.0)),
            "agg_factor_last3_hist": float(hist_features.get("agg_factor_last3_hist", 1.0)),
            "agg_factor_last5_hist": float(hist_features.get("agg_factor_last5_hist", 1.0)),
            "agg_factor_last10_hist": float(hist_features.get("agg_factor_last10_hist", 1.0)),
            "street_adv_last3_hist": float(hist_features.get("street_adv_last3_hist", 50.0)),
            "street_adv_last5_hist": float(hist_features.get("street_adv_last5_hist", 50.0)),
            "street_adv_last10_hist": float(hist_features.get("street_adv_last10_hist", 50.0)),
            "stack_trend_last3_hist": float(hist_features.get("stack_trend_last3_hist", 0.0)),
            "stack_trend_last5_hist": float(hist_features.get("stack_trend_last5_hist", 0.0)),
            "stack_trend_last10_hist": float(hist_features.get("stack_trend_last10_hist", 0.0)),

            # Bet sizing feature (ratio, no BB normalization needed)
            "bet_pct_pot": bet_pct_pot,
        }

        # Board texture features for post-flop streets
        if street in ("flop", "turn", "river"):
            if board_cards:
                board_features = self._calculate_board_texture(board_cards)
                features.update(board_features)
            else:
                # Add zero board features for postflop without cards
                features["board_pair_or_better"] = 0.0
                features["board_flush_possible"] = 0.0
                features["board_straight_possible"] = 0.0

        return features

    def calculate_profit_features(
        self,
        pot_size: float,
        street: str,
        session: GameSession,
        hero_seat: int,
        hero_name: str,
        hero_stack: float,
        hero_hole_cards: List[str],
        all_stacks: List[float],
        board_cards: List[str],
        active_opponents: List[Dict[str, Any]],
        opponent_predictions: List[Dict[str, Any]],
        facing_bet: float,
        bb: float,
        position_from_button: int = 0,
        num_players: int = 6,
        pot_contribution: float = 0.0,
        raises_so_far: int = 0,
        calls_so_far: int = 0,
        action_no_in_hand: int = 1,
    ) -> Dict[str, float]:
        """
        Calculate features for Profit Predictor (Model 02).

        Matches the EXACT V3 schema from Databricks endpoint (63 features).

        Args:
            pot_size: Current pot size
            street: Current street
            session: Game session
            hero_seat: Hero's seat index
            hero_name: Hero's name
            hero_stack: Hero's stack
            hero_hole_cards: Hero's hole cards
            all_stacks: All stacks at table
            board_cards: Community cards
            active_opponents: List of active opponent info
            opponent_predictions: Opponent strength predictions from Model 1
            facing_bet: Amount hero needs to call
            bb: Big blind size
            position_from_button: Position relative to button
            num_players: Number of players at table
            pot_contribution: Hero's total pot contribution this hand
            raises_so_far: Number of raises in current street
            calls_so_far: Number of calls in current street
            action_no_in_hand: Action sequence number

        Returns:
            Feature dictionary for V3 profit model (63 features)
        """
        # Get historical stats
        stats = session.get_player_stats(hero_name)
        hist_features = stats.get_historical_features()

        # Calculate stack vs median
        median_stack = statistics.median(all_stacks) if all_stacks else hero_stack
        stack_vs_median = float(hero_stack / (median_stack + 1e-6))

        # Board texture
        board_texture = self._calculate_board_texture(board_cards) if board_cards else {}

        # Hand strength and rank
        hand_strength = self._calculate_hand_strength(hero_hole_cards, board_cards, street)
        hand_rank = self._calculate_hand_rank(hero_hole_cards, board_cards, street)

        # Bet percentage of pot
        bet_pct_pot = float(facing_bet / (pot_size + 1e-6)) if pot_size > 0 else 0.0

        # Normalize chip values to BB (model was trained on BB-normalized data)
        bb_safe = max(bb, 1.0)  # Avoid division by zero
        pot_size_bb = float(pot_size / bb_safe)
        hero_stack_bb = float(hero_stack / bb_safe)

        # Calculate opponent stats
        opponent_stacks = [opp.get("stack", 0) for opp in active_opponents] if active_opponents else []
        num_opponents = len(active_opponents) if active_opponents else 0
        avg_opp_stack = float(sum(opponent_stacks) / len(opponent_stacks)) if opponent_stacks else hero_stack
        max_opp_stack = float(max(opponent_stacks)) if opponent_stacks else hero_stack
        min_opp_stack = float(min(opponent_stacks)) if opponent_stacks else hero_stack

        # Calculate position counts from active opponents
        pos_counts = {"BB": 0, "BTN": 0, "CO": 0, "HJ": 0, "MP": 0, "SB": 0, "UTG": 0, "UTG+1": 0}
        for opp in (active_opponents or []):
            pos = opp.get("position", "")
            if pos in pos_counts:
                pos_counts[pos] += 1

        # Process opponent predictions
        opp_strengths = []
        air_count = 0
        middle_count = 0
        nutted_count = 0
        has_opp_preds = 0
        if opponent_predictions:
            has_opp_preds = 1
            for pred in opponent_predictions:
                strength = pred.get("predicted_strength", 0.5)
                opp_strengths.append(strength)
                bucket = pred.get("predicted_class", "middle")
                if bucket == "air":
                    air_count += 1
                elif bucket == "middle":
                    middle_count += 1
                elif bucket == "nutted":
                    nutted_count += 1

        opp_strength_mean = float(sum(opp_strengths) / len(opp_strengths)) if opp_strengths else 0.5
        opp_strength_max = float(max(opp_strengths)) if opp_strengths else 0.5
        opp_strength_min = float(min(opp_strengths)) if opp_strengths else 0.5

        # Pot before action
        pot_before_action = pot_size - facing_bet if pot_size > facing_bet else pot_size
        pot_odds_call = float(facing_bet / (pot_size + facing_bet)) if (pot_size + facing_bet) > 0 else 0.0

        # Build features matching EXACT V3 schema (63 features)
        features = {
            # Core features (BB-normalized)
            "pot_size": pot_size_bb,
            "hand_rank": float(hand_rank),
            "starting_stack": hero_stack_bb,
            "stack_vs_table_median": stack_vs_median,
            "position_from_button": float(position_from_button),
            "num_players": float(num_players),
            "hand_equity": float(hand_strength),

            # Board features (original)
            "board_pair_or_better": float(board_texture.get("board_pair_or_better", 0)),
            "board_flush_possible": float(board_texture.get("board_flush_possible", 0)),
            "board_straight_possible": float(board_texture.get("board_straight_possible", 0)),

            # Draw flags
            "hole_pair_flag": 1.0 if self._has_pocket_pair(hero_hole_cards) else 0.0,
            "has_flush_draw_flag": 1.0 if self._has_flush_draw(hero_hole_cards, board_cards) else 0.0,
            "has_straight_draw_flag": 1.0 if self._has_straight_draw(hero_hole_cards, board_cards) else 0.0,

            # Betting context
            "bet_pct_pot": bet_pct_pot,
            "action_no_in_hand": float(action_no_in_hand),
            "raises_so_far": float(raises_so_far),
            "calls_so_far": float(calls_so_far),

            # Historical stats - last 3 (percentages 0-100)
            "vpip_last3_hist": float(hist_features.get("vpip_last3_hist", 50.0)),
            "pfr_last3_hist": float(hist_features.get("pfr_last3_hist", 30.0)),
            "street_adv_last3_hist": float(hist_features.get("street_adv_last3_hist", 50.0)),
            "agg_factor_last3_hist": float(hist_features.get("agg_factor_last3_hist", 1.5)),
            "stack_trend_last3_hist": float(hist_features.get("stack_trend_last3_hist", 0.0)),

            # Historical stats - last 5 (percentages 0-100)
            "vpip_last5_hist": float(hist_features.get("vpip_last5_hist", 50.0)),
            "pfr_last5_hist": float(hist_features.get("pfr_last5_hist", 30.0)),
            "street_adv_last5_hist": float(hist_features.get("street_adv_last5_hist", 50.0)),
            "agg_factor_last5_hist": float(hist_features.get("agg_factor_last5_hist", 1.5)),
            "stack_trend_last5_hist": float(hist_features.get("stack_trend_last5_hist", 0.0)),

            # Historical stats - last 10 (percentages 0-100)
            "vpip_last10_hist": float(hist_features.get("vpip_last10_hist", 50.0)),
            "pfr_last10_hist": float(hist_features.get("pfr_last10_hist", 30.0)),
            "street_adv_last10_hist": float(hist_features.get("street_adv_last10_hist", 50.0)),
            "agg_factor_last10_hist": float(hist_features.get("agg_factor_last10_hist", 1.5)),
            "stack_trend_last10_hist": float(hist_features.get("stack_trend_last10_hist", 0.0)),

            # Player counts
            "players_at_table": float(num_players),
            "players_active": float(num_opponents + 1),  # +1 for hero
            "opponents_active": float(num_opponents),

            # Opponent stacks (BB-normalized)
            "avg_opponent_stack": float(avg_opp_stack / bb_safe),
            "max_opponent_stack": float(max_opp_stack / bb_safe),
            "min_opponent_stack": float(min_opp_stack / bb_safe),

            # Position counts
            "pos_count_BB": float(pos_counts["BB"]),
            "pos_count_BTN": float(pos_counts["BTN"]),
            "pos_count_CO": float(pos_counts["CO"]),
            "pos_count_HJ": float(pos_counts["HJ"]),
            "pos_count_MP": float(pos_counts["MP"]),
            "pos_count_SB": float(pos_counts["SB"]),
            "pos_count_UTG": float(pos_counts["UTG"]),
            "pos_count_UTG+1": float(pos_counts["UTG+1"]),

            # Board texture (additional)
            "board_monotone": float(board_texture.get("board_monotone", 0)),
            "board_paired": float(board_texture.get("board_paired", 0)),
            "board_straighty": float(board_texture.get("board_straighty", 0)),
            "board_flush_pressure": float(board_texture.get("board_flush_pressure", 0)),
            "board_straight_pressure": float(board_texture.get("board_straight_pressure", 0)),

            # Opponent predictions
            "predicted_strength": float(hand_strength),  # Hero's predicted strength
            "opponent_strength_mean": opp_strength_mean,
            "opponent_strength_max": opp_strength_max,
            "opponent_strength_min": opp_strength_min,
            "opponent_air_count": float(air_count),
            "opponent_middle_count": float(middle_count),
            "opponent_nutted_count": float(nutted_count),
            "has_opponent_predictions": float(has_opp_preds),

            # Pot and betting context
            "pot_before_action": float(pot_before_action / bb_safe),
            "facing_call": float(facing_bet / bb_safe),
            "pot_odds_call": pot_odds_call,
            "bb": bb,
        }

        return features

    def _get_position_name(self, position_from_button: int, num_players: int) -> str:
        """Convert position_from_button to position name."""
        if num_players <= 2:
            return "BTN" if position_from_button == 0 else "BB"

        # Standard position mapping
        position_map = {
            0: "BTN",
            1: "SB",
            2: "BB",
        }

        if position_from_button in position_map:
            return position_map[position_from_button]

        # Early positions for larger tables
        remaining = num_players - 3  # Positions after BB
        pos_offset = position_from_button - 3

        if remaining <= 1:
            return "UTG"
        elif remaining == 2:
            return "UTG" if pos_offset == 0 else "CO"
        elif remaining == 3:
            return ["UTG", "MP", "CO"][min(pos_offset, 2)]
        elif remaining == 4:
            return ["UTG", "UTG+1", "MP", "CO"][min(pos_offset, 3)]
        else:
            return ["UTG", "UTG+1", "MP", "HJ", "CO"][min(pos_offset, 4)]

    def _calculate_hand_rank(self, hole_cards: List[str], board_cards: List[str], street: str) -> float:
        """Calculate hand rank (0-9 scale for poker hands)."""
        if not hole_cards or len(hole_cards) != 2:
            return 5.0  # Default middle rank

        if street == "preflop" or not board_cards:
            # Preflop: rank based on card values
            r1, _ = parse_card(hole_cards[0])
            r2, _ = parse_card(hole_cards[1])
            v1 = get_rank_value(r1)
            v2 = get_rank_value(r2)

            if r1 == r2:  # Pocket pair
                return min(9.0, 3.0 + (v1 / 4.0))  # 3-6.5 based on pair strength
            return min(9.0, 1.0 + ((v1 + v2) / 7.0))  # 1-5 based on high cards

        # Postflop: use simplified hand ranking
        all_cards = hole_cards + board_cards
        ranks = [parse_card(c)[0] for c in all_cards]
        suits = [parse_card(c)[1] for c in all_cards]

        # Count rank matches
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        max_match = max(rank_counts.values()) if rank_counts else 1

        # Check for flush
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        has_flush = any(c >= 5 for c in suit_counts.values())

        # Simplified hand ranks
        if has_flush:
            return 8.0  # Flush
        elif max_match >= 4:
            return 9.0  # Quads
        elif max_match == 3:
            # Check for full house
            pair_count = sum(1 for c in rank_counts.values() if c >= 2)
            return 7.0 if pair_count >= 2 else 6.0  # Full house or trips
        elif max_match == 2:
            pair_count = sum(1 for c in rank_counts.values() if c == 2)
            return 4.0 if pair_count >= 2 else 3.0  # Two pair or pair
        else:
            return 1.0  # High card

    def calculate_policy_features(
        self,
        profit_features: Dict[str, Any],
        hero_stack: float,
        pot_size: float,
        facing_bet: float,
        street: str,
        hero_hole_cards: List[str],
        board_cards: List[str],
        opponent_predictions: List[Dict[str, Any]],
        position_from_button: int = 0,
        num_players: int = 6,
        session: Optional[GameSession] = None,
        hero_name: str = "Hero",
        all_stacks: Optional[List[float]] = None,
        bb: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Calculate features for Policy Action Classifier (Model 03).

        Matches the EXACT V3 schema from Databricks endpoint (43 features):
        ['starting_stack', 'stack_vs_table_median', 'position_from_button', 'num_players',
         'hand_equity', 'board_pair_or_better', 'board_flush_possible', 'board_straight_possible',
         'hole_pair_flag', 'has_flush_draw_flag', 'has_straight_draw_flag',
         'vpip_last3_hist', 'pfr_last3_hist', 'agg_factor_last3_hist',
         'vpip_last5_hist', 'pfr_last5_hist', 'agg_factor_last5_hist',
         'vpip_last10_hist', 'pfr_last10_hist', 'agg_factor_last10_hist',
         'predicted_strength', 'opponent_strength_mean', 'opponent_strength_max',
         'opponent_strength_min', 'opponent_air_count', 'opponent_middle_count',
         'opponent_nutted_count', 'strength_advantage', 'strength_disadvantage',
         'strength_strong', 'heads_up', 'three_way', 'multiway',
         'position_bucket_CO', 'position_bucket_UTG+1', 'position_bucket_BTN',
         'position_bucket_UTG', 'position_bucket_BB', 'position_bucket_MP',
         'position_bucket_SB', 'position_bucket_HJ', 'predicted_bucket_middle',
         'predicted_bucket_air', 'predicted_bucket_nutted']

        Args:
            profit_features: Features from profit model calculation (used for some shared values)
            hero_stack: Hero's current stack
            pot_size: Current pot size
            facing_bet: Amount to call
            street: Current street
            hero_hole_cards: Hero's hole cards
            board_cards: Board cards
            opponent_predictions: Opponent predictions from Model 01
            position_from_button: Position relative to button
            num_players: Number of players at table
            session: Game session for historical stats
            hero_name: Hero's name for session lookup
            all_stacks: All player stacks for median calculation
            bb: Big blind size for BB normalization

        Returns:
            Feature dictionary for V3 policy model (43 features)
        """
        # Get position name for one-hot encoding
        position_name = self._get_position_name(position_from_button, num_players)

        # Get hand equity from profit features or calculate fresh
        hand_equity = float(profit_features.get("hand_equity", 0.5))

        # Get predicted bucket from hero's strength
        if hand_equity < 0.33:
            predicted_bucket = "air"
        elif hand_equity < 0.66:
            predicted_bucket = "middle"
        else:
            predicted_bucket = "nutted"

        # Calculate opponent strength aggregations from opponent_predictions
        strength_map = {"air": 0.2, "middle": 0.5, "nutted": 0.8}
        opp_strengths = [
            strength_map.get(pred.get("predicted_class", "middle"), 0.5)
            for pred in opponent_predictions
        ] if opponent_predictions else [0.5]

        opp_str_mean = float(statistics.mean(opp_strengths)) if opp_strengths else 0.5
        opp_str_max = float(max(opp_strengths)) if opp_strengths else 0.5
        opp_str_min = float(min(opp_strengths)) if opp_strengths else 0.5
        opp_air_count = float(sum(1 for p in opponent_predictions if p.get("predicted_class") == "air"))
        opp_middle_count = float(sum(1 for p in opponent_predictions if p.get("predicted_class") == "middle"))
        opp_nutted_count = float(sum(1 for p in opponent_predictions if p.get("predicted_class") == "nutted"))

        # Opponent count for multiway indicators
        opp_count = len(opponent_predictions) if opponent_predictions else 1

        # Strength comparison
        strength_adv = float(hand_equity - opp_str_mean)

        # Get historical stats - either from session or profit_features
        if session:
            stats = session.get_player_stats(hero_name)
            hist_features = stats.get_historical_features()
        else:
            hist_features = {}

        # Calculate stack vs median
        if all_stacks:
            median_stack = statistics.median(all_stacks)
            stack_vs_median = float(hero_stack / (median_stack + 1e-6))
        else:
            stack_vs_median = float(profit_features.get("stack_vs_table_median", 1.0))

        # Normalize stack to BB
        bb_safe = max(bb, 1.0)
        hero_stack_bb = float(hero_stack / bb_safe)

        # Board texture
        board_texture = self._calculate_board_texture(board_cards) if board_cards else {}

        # Build features matching EXACT V3 schema (43 features in exact order)
        features = {
            # Core features
            "starting_stack": hero_stack_bb,
            "stack_vs_table_median": stack_vs_median,
            "position_from_button": float(position_from_button),
            "num_players": float(num_players),
            "hand_equity": hand_equity,

            # Board features
            "board_pair_or_better": float(board_texture.get("board_pair_or_better", 0)),
            "board_flush_possible": float(board_texture.get("board_flush_possible", 0)),
            "board_straight_possible": float(board_texture.get("board_straight_possible", 0)),

            # Draw flags
            "hole_pair_flag": 1.0 if self._has_pocket_pair(hero_hole_cards) else 0.0,
            "has_flush_draw_flag": 1.0 if self._has_flush_draw(hero_hole_cards, board_cards) else 0.0,
            "has_straight_draw_flag": 1.0 if self._has_straight_draw(hero_hole_cards, board_cards) else 0.0,

            # Historical stats (last 3) - V3: percentages 0-100 for VPIP, PFR
            "vpip_last3_hist": float(hist_features.get("vpip_last3_hist", profit_features.get("vpip_last3_hist", 50.0))),
            "pfr_last3_hist": float(hist_features.get("pfr_last3_hist", profit_features.get("pfr_last3_hist", 30.0))),
            "agg_factor_last3_hist": float(hist_features.get("agg_factor_last3_hist", 1.0)),

            # Historical stats (last 5) - V3: percentages 0-100 for VPIP, PFR
            "vpip_last5_hist": float(hist_features.get("vpip_last5_hist", 50.0)),
            "pfr_last5_hist": float(hist_features.get("pfr_last5_hist", 30.0)),
            "agg_factor_last5_hist": float(hist_features.get("agg_factor_last5_hist", 1.0)),

            # Historical stats (last 10) - V3: percentages 0-100 for VPIP, PFR
            "vpip_last10_hist": float(hist_features.get("vpip_last10_hist", 50.0)),
            "pfr_last10_hist": float(hist_features.get("pfr_last10_hist", 30.0)),
            "agg_factor_last10_hist": float(hist_features.get("agg_factor_last10_hist", 1.0)),

            # Opponent predictions (calculated fresh from opponent_predictions)
            "predicted_strength": hand_equity,  # Hero's predicted strength
            "opponent_strength_mean": opp_str_mean,
            "opponent_strength_max": opp_str_max,
            "opponent_strength_min": opp_str_min,
            "opponent_air_count": opp_air_count,
            "opponent_middle_count": opp_middle_count,
            "opponent_nutted_count": opp_nutted_count,

            # Strength comparison
            "strength_advantage": strength_adv,
            "strength_disadvantage": 1.0 if strength_adv < -0.1 else 0.0,
            "strength_strong": 1.0 if strength_adv > 0.2 else 0.0,

            # Multiway indicators
            "heads_up": 1.0 if opp_count == 1 else 0.0,
            "three_way": 1.0 if opp_count == 2 else 0.0,
            "multiway": 1.0 if opp_count >= 3 else 0.0,

            # Position bucket one-hot encoding
            "position_bucket_CO": 1.0 if position_name == "CO" else 0.0,
            "position_bucket_UTG+1": 1.0 if position_name == "UTG+1" else 0.0,
            "position_bucket_BTN": 1.0 if position_name == "BTN" else 0.0,
            "position_bucket_UTG": 1.0 if position_name == "UTG" else 0.0,
            "position_bucket_BB": 1.0 if position_name == "BB" else 0.0,
            "position_bucket_MP": 1.0 if position_name == "MP" else 0.0,
            "position_bucket_SB": 1.0 if position_name == "SB" else 0.0,
            "position_bucket_HJ": 1.0 if position_name == "HJ" else 0.0,

            # Predicted bucket one-hot encoding
            "predicted_bucket_middle": 1.0 if predicted_bucket == "middle" else 0.0,
            "predicted_bucket_air": 1.0 if predicted_bucket == "air" else 0.0,
            "predicted_bucket_nutted": 1.0 if predicted_bucket == "nutted" else 0.0,
        }

        return features

    def _has_pocket_pair(self, hole_cards: List[str]) -> bool:
        """Check if hole cards form a pocket pair."""
        if len(hole_cards) != 2:
            return False
        r1, _ = parse_card(hole_cards[0])
        r2, _ = parse_card(hole_cards[1])
        return r1 == r2

    def _has_flush_draw(self, hole_cards: List[str], board_cards: List[str]) -> bool:
        """Check if there's a flush draw (4 cards of same suit)."""
        if not board_cards or len(hole_cards) != 2:
            return False
        all_cards = hole_cards + board_cards
        suits = [parse_card(c)[1] for c in all_cards if parse_card(c)[1]]
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        return any(c == 4 for c in suit_counts.values())

    def _has_straight_draw(self, hole_cards: List[str], board_cards: List[str]) -> bool:
        """Check if there's an open-ended straight draw."""
        if not board_cards or len(hole_cards) != 2:
            return False
        all_cards = hole_cards + board_cards
        ranks = [parse_card(c)[0] for c in all_cards if parse_card(c)[0]]
        rank_values = sorted(set(get_rank_value(r) for r in ranks))

        # Check for 4 cards within a 5-card span (OESD)
        if 14 in rank_values:
            rank_values = sorted(set(rank_values) | {1})  # Ace as 1 for wheel

        for i in range(len(rank_values) - 3):
            span = rank_values[i + 3] - rank_values[i]
            if span <= 4:  # 4 cards within 5 positions = straight draw
                return True
        return False

    def _calculate_board_texture(self, board_cards: List[str]) -> Dict[str, float]:
        """Calculate board texture features."""
        if not board_cards:
            return {
                "board_pair_or_better": 0.0,
                "board_flush_possible": 0.0,
                "board_straight_possible": 0.0,
                "board_monotone": 0.0,
                "board_paired": 0.0,
                "board_straighty": 0.0,
                "board_flush_pressure": 0.0,
                "board_straight_pressure": 0.0,
            }

        ranks = []
        suits = []
        for card in board_cards:
            rank, suit = parse_card(card)
            if rank:
                ranks.append(rank)
                suits.append(suit)

        # Pair or better: any rank appears 2+ times
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        pair_or_better = any(c >= 2 for c in rank_counts.values())

        # Flush possible: 3+ cards of same suit
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        flush_possible = max_suit_count >= 3

        # Straight possible: 3+ connected ranks
        rank_values = sorted(set(get_rank_value(r) for r in ranks))
        # Add ace as 1 for wheel straights
        if 14 in rank_values:
            rank_values = sorted(set(rank_values) | {1})

        straight_possible = False
        for i in range(len(rank_values) - 2):
            if rank_values[i + 2] - rank_values[i] <= 4:
                straight_possible = True
                break

        # Additional texture features for profit model
        # Monotone: all cards same suit
        monotone = max_suit_count == len(board_cards) and len(board_cards) >= 3

        # Board paired (same as pair_or_better for this purpose)
        board_paired = pair_or_better

        # Straighty: high connectivity (check if 3+ cards form a straight possibility)
        straighty = straight_possible

        # Flush pressure: 2+ cards of same suit (potential flush draw possible)
        flush_pressure = max_suit_count >= 2

        # Straight pressure: cards are connected (potential straight draws)
        straight_pressure = False
        if len(rank_values) >= 2:
            for i in range(len(rank_values) - 1):
                if rank_values[i + 1] - rank_values[i] <= 2:
                    straight_pressure = True
                    break

        return {
            "board_pair_or_better": 1.0 if pair_or_better else 0.0,
            "board_flush_possible": 1.0 if flush_possible else 0.0,
            "board_straight_possible": 1.0 if straight_possible else 0.0,
            "board_monotone": 1.0 if monotone else 0.0,
            "board_paired": 1.0 if board_paired else 0.0,
            "board_straighty": 1.0 if straighty else 0.0,
            "board_flush_pressure": 1.0 if flush_pressure else 0.0,
            "board_straight_pressure": 1.0 if straight_pressure else 0.0,
        }

    def _calculate_hand_strength(
        self,
        hole_cards: List[str],
        board_cards: List[str],
        street: str,
    ) -> float:
        """
        Calculate hand strength (0-1 scale).

        Uses treys library if available, otherwise simplified calculation.
        """
        logger.info(f"HAND_STRENGTH: hole_cards={hole_cards}, board_cards={board_cards}, street={street}")

        if not hole_cards or len(hole_cards) != 2:
            logger.warning(f"HAND_STRENGTH: Invalid hole cards, returning 0.5")
            return 0.5

        if street == "preflop":
            strength = self._preflop_hand_strength(hole_cards)
            logger.info(f"HAND_STRENGTH: Preflop calculation = {strength}")
            return strength

        if self._evaluator and board_cards:
            strength = self._treys_hand_strength(hole_cards, board_cards)
            logger.info(f"HAND_STRENGTH: Treys evaluation = {strength}")
            return strength

        strength = self._simplified_postflop_strength(hole_cards, board_cards)
        logger.info(f"HAND_STRENGTH: Simplified calculation = {strength}")
        return strength

    def _preflop_hand_strength(self, hole_cards: List[str]) -> float:
        """Calculate preflop hand strength based on card ranks."""
        if len(hole_cards) != 2:
            return 0.5

        r1, s1 = parse_card(hole_cards[0])
        r2, s2 = parse_card(hole_cards[1])

        v1 = get_rank_value(r1)
        v2 = get_rank_value(r2)
        high = max(v1, v2)
        low = min(v1, v2)

        suited = s1 == s2
        paired = r1 == r2

        if paired:
            # Pairs: AA=1.0, 22=0.5
            return 0.5 + (high / 28.0)
        elif suited:
            # Suited hands
            return ((high + low) / 28.0) * 0.6
        else:
            # Offsuit hands
            return ((high + low) / 28.0) * 0.5

    def _treys_hand_strength(self, hole_cards: List[str], board_cards: List[str]) -> float:
        """Calculate hand strength using treys evaluator."""
        try:
            from treys import Card

            logger.info(f"TREYS: Converting hole_cards={hole_cards}, board_cards={board_cards}")

            # Convert cards to treys format
            hole = [Card.new(c) for c in hole_cards]
            board = [Card.new(c) for c in board_cards]

            logger.info(f"TREYS: Converted hole={hole}, board={board}")

            # Evaluate (lower score = stronger hand)
            score = self._evaluator.evaluate(board, hole)
            # Score ranges from 1 (best) to 7462 (worst)
            strength = 1.0 - (score / 7462.0)
            logger.info(f"TREYS: score={score}, strength={strength}")
            return strength
        except Exception as e:
            logger.warning(f"Treys evaluation failed: {e}", exc_info=True)
            return self._simplified_postflop_strength(hole_cards, board_cards)

    def _simplified_postflop_strength(
        self,
        hole_cards: List[str],
        board_cards: List[str],
    ) -> float:
        """Simplified postflop strength estimation."""
        # Basic heuristics without full evaluation
        all_cards = hole_cards + board_cards
        ranks = [parse_card(c)[0] for c in all_cards]
        suits = [parse_card(c)[1] for c in all_cards]

        # Count rank matches (pairs, trips, etc.)
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        max_match = max(rank_counts.values()) if rank_counts else 1

        # Check for flush
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        has_flush = any(c >= 5 for c in suit_counts.values())

        # Base strength from pairs
        if max_match >= 4:
            strength = 0.9
        elif max_match >= 3:
            strength = 0.75
        elif max_match >= 2:
            strength = 0.55
        else:
            # High card
            hole_ranks = [get_rank_value(parse_card(c)[0]) for c in hole_cards]
            strength = 0.3 + (max(hole_ranks, default=7) / 40.0)

        if has_flush:
            strength = max(strength, 0.8)

        return min(strength, 1.0)

    def _is_monotone(self, board_cards: List[str]) -> bool:
        """Check if board is monotone (all same suit)."""
        if not board_cards:
            return False
        suits = [parse_card(c)[1] for c in board_cards if parse_card(c)[1]]
        return len(set(suits)) == 1 and len(suits) >= 3
