"""
ML Service - High-level orchestration of ML predictions.

Coordinates feature calculation, model calls, and result aggregation.
(with debug_info support)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import asyncio

from .endpoint_config import EndpointRegistry
from .model_client import DatabricksModelClient, PredictionResult
from .feature_engine import FeatureEngine
from .game_session import GameSession, GameSessionManager, MLPredictionRecord

logger = logging.getLogger(__name__)


@dataclass
class OpponentPrediction:
    """Prediction for a single opponent's hand strength."""
    seat_index: int
    player_name: str
    predicted_class: str  # 'air', 'middle', 'nutted'
    probabilities: Dict[str, float]
    confidence: float
    from_fallback: bool = False


@dataclass
class ActionRecommendation:
    """Recommended action from policy model."""
    action: str  # 'fold', 'call', 'bet' (3-class model)
    chip_amount: int
    probabilities: Dict[str, float]
    confidence: float
    reasoning: str
    from_fallback: bool = False


@dataclass
class ProfitPrediction:
    """Profit prediction from profit model."""
    predicted_profit_bb: float
    interpretation: str  # 'highly profitable', 'marginal', etc.
    from_fallback: bool = False


@dataclass
class MLPredictionsResult:
    """Complete ML predictions for a game state."""
    hand_id: int
    street: str
    hero_seat: int
    action_index: int

    # Predictions
    opponent_predictions: List[OpponentPrediction]
    profit_prediction: Optional[ProfitPrediction]
    action_recommendation: Optional[ActionRecommendation]

    # Metadata
    latencies_ms: Dict[str, float]
    any_from_fallback: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Debug info - model inputs and raw outputs
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "handId": self.hand_id,
            "street": self.street,
            "heroSeat": self.hero_seat,
            "actionIndex": self.action_index,
            "opponentPredictions": [
                {
                    "seatIndex": op.seat_index,
                    "playerName": op.player_name,
                    "predictedClass": op.predicted_class,
                    "probabilities": op.probabilities,
                    "confidence": op.confidence,
                    "fromFallback": op.from_fallback,
                }
                for op in self.opponent_predictions
            ],
            "profitPrediction": {
                "predictedProfitBB": self.profit_prediction.predicted_profit_bb,
                "interpretation": self.profit_prediction.interpretation,
                "fromFallback": self.profit_prediction.from_fallback,
            } if self.profit_prediction else None,
            "actionRecommendation": {
                "action": self.action_recommendation.action,
                "chipAmount": self.action_recommendation.chip_amount,
                "probabilities": self.action_recommendation.probabilities,
                "confidence": self.action_recommendation.confidence,
                "reasoning": self.action_recommendation.reasoning,
                "fromFallback": self.action_recommendation.from_fallback,
            } if self.action_recommendation else None,
            "latenciesMs": self.latencies_ms,
            "anyFromFallback": self.any_from_fallback,
            "timestamp": self.timestamp,
            "debugInfo": self.debug_info,
        }


class MLService:
    """
    High-level ML service for poker predictions.

    Orchestrates:
    1. Feature calculation from game state
    2. Parallel model endpoint calls
    3. Result aggregation and interpretation
    4. Prediction storage for review
    """

    def __init__(
        self,
        registry: EndpointRegistry,
        session_manager: GameSessionManager,
    ):
        """
        Initialize ML service.

        Args:
            registry: Endpoint configuration registry
            session_manager: Game session manager
        """
        self.registry = registry
        self.session_manager = session_manager
        self.model_client = DatabricksModelClient(registry)
        self.feature_engine = FeatureEngine()

    def _infer_action_type(
        self,
        player: Dict[str, Any],
        street: str,
        bb: float,
        all_players: List[Dict[str, Any]],
        hero_seat: int,
    ) -> str:
        """Infer action type from player's betting state."""
        opp_bet = player.get("betThisStreet", 0)
        opp_seat = player.get("seatIndex")
        opp_total_committed = player.get("totalCommitted", 0)

        # Calculate max bet from OTHER players
        other_bets = [p.get("betThisStreet", 0) for p in all_players
                      if p.get("seatIndex") != opp_seat and p.get("seatIndex") != hero_seat]
        max_other_bet = max(other_bets) if other_bets else 0

        if player.get("isAllIn", False) and opp_total_committed > 0:
            return "bet_or_raise_to"
        elif street == "preflop":
            if opp_bet > bb:
                return "bet_or_raise_to"
            else:
                return "call_or_check"
        else:
            if opp_bet > max_other_bet and opp_bet > 0:
                return "bet_or_raise_to"
            elif opp_bet > 0:
                return "call_or_check"
            else:
                return "call_or_check"

    async def generate_predictions(
        self,
        session_id: str,
        hand_id: int,
        street: str,
        hero_seat: int,
        hero_name: str,
        hero_stack: float,
        hero_hole_cards: List[str],
        pot_size: float,
        facing_bet: float,
        bb: float,
        all_players: List[Dict[str, Any]],
        board_cards: List[str],
        position_index: int,
    ) -> MLPredictionsResult:
        """
        Generate all ML predictions for current game state.

        This is called on every player action to update predictions.

        Args:
            session_id: Current session ID
            hand_id: Current hand number
            street: Current street ('preflop', 'flop', 'turn', 'river')
            hero_seat: Hero's seat index
            hero_name: Hero's name
            hero_stack: Hero's current stack
            hero_hole_cards: Hero's hole cards
            pot_size: Current pot size
            facing_bet: Amount hero needs to call
            bb: Big blind size
            all_players: List of all player states
            board_cards: Community cards
            position_index: Hero's position

        Returns:
            MLPredictionsResult with all predictions
        """
        session = self.session_manager.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found, creating new")
            session = self.session_manager.create_session(
                session_id, "", hero_seat
            )

        action_index = session.get_action_count()
        latencies: Dict[str, float] = {}
        any_fallback = False

        # Log game state for debugging
        logger.info(
            f"=== ML PREDICTION REQUEST: hand={hand_id}, street={street}, pot={pot_size}, "
            f"facing_bet={facing_bet}, board={board_cards}, bb={bb}, position_index={position_index} ==="
        )

        # Log all player states for debugging
        for p in all_players:
            logger.info(
                f"PLAYER STATE: seat={p.get('seatIndex')}, name={p.get('name')}, "
                f"betThisStreet={p.get('betThisStreet', 0)}, totalCommitted={p.get('totalCommitted', 0)}, "
                f"stack={p.get('stack', 0)}, isFolded={p.get('isFolded')}, isAllIn={p.get('isAllIn')}"
            )

        # 1. Calculate features for active opponents
        active_opponents = [
            p for p in all_players
            if p.get("seatIndex") != hero_seat
            and not p.get("isFolded", False)
            and not p.get("isSittingOut", False)
        ]

        all_stacks = [p.get("stack", 0) for p in all_players]

        # Calculate action counts from current game state
        # Look at bets THIS STREET to determine raises vs calls
        raises_this_hand = 0
        calls_this_hand = 0

        # Get all bets this street from active players
        street_bets = []
        for p in all_players:
            if p.get("isFolded", False) or p.get("isSittingOut", False):
                continue
            bet_this_street = p.get("betThisStreet", 0)
            street_bets.append(bet_this_street)

        # Find max bet this street (the current bet to call)
        max_bet = max(street_bets) if street_bets else 0

        # Now categorize each player's action
        for p in all_players:
            if p.get("isFolded", False) or p.get("isSittingOut", False):
                continue
            bet_this_street = p.get("betThisStreet", 0)

            if street == "preflop":
                # On preflop:
                # - Raising = betting more than the current bet (BB or raise)
                # - Calling = matching the bet but not raising
                # - Blinds posting their required amount don't count as voluntary actions
                position = p.get("position", "")
                if position == "SB" and bet_this_street == bb / 2:
                    continue  # Just posted SB
                if position == "BB" and bet_this_street == bb:
                    continue  # Just posted BB

                if bet_this_street > max_bet * 0.99:  # Allow small rounding tolerance
                    # They have the current max bet - could be raiser or caller
                    # Count as raise if they put in more than 1 BB voluntarily
                    if bet_this_street > bb:
                        raises_this_hand += 1
                elif bet_this_street > 0:
                    calls_this_hand += 1
            else:
                # Post-flop: any bet/raise is aggressive, calling the bet is passive
                if bet_this_street > 0:
                    if bet_this_street >= max_bet * 0.99:
                        # They have the max bet - check if they MADE it or called it
                        # For simplicity, first person to bet = raise, others = call
                        # This is approximate since we don't have full action history
                        raises_this_hand += 1
                    else:
                        calls_this_hand += 1

        # 2. Get opponent strength predictions (parallel)
        opponent_predictions = []
        opponent_features_list = []
        opp_results = []
        if active_opponents:
            # Find the big blind (minimum required bet)
            # BB is already passed in as parameter

            # Find bets from all non-hero players to determine what's a raise vs call
            all_bets = [p.get("betThisStreet", 0) for p in all_players if p.get("seatIndex") != hero_seat]
            max_bet_this_street = max(all_bets) if all_bets else 0

            for opp in active_opponents:
                opp_name = opp.get("name", f"Seat_{opp.get('seatIndex')}")
                opp_seat = opp.get("seatIndex")
                opp_bet = opp.get("betThisStreet", 0)
                opp_total_committed = opp.get("totalCommitted", 0)

                # Calculate bet as percentage of pot for debugging
                bet_pct_pot = opp_bet / (pot_size + 1e-6) if pot_size > 0 else 0

                # Calculate max bet from OTHER players (not this opponent)
                # This helps us determine if this player raised or called
                other_bets = [p.get("betThisStreet", 0) for p in all_players
                              if p.get("seatIndex") != opp_seat and p.get("seatIndex") != hero_seat]
                max_other_bet = max(other_bets) if other_bets else 0

                # Infer action type from betting behavior
                # Key insight: In preflop, bets > BB are raises. Post-flop, any bet is a bet.
                # We need to compare to the blind structure and other players' bets.
                # Also: Players who have committed significant amounts (e.g., calling a shove)
                # should be treated as making strong commitments.

                if opp.get("isAllIn", False) and opp_total_committed > 0:
                    # All-in with chips in = aggressive action
                    action_type = "bet_or_raise_to"
                elif street == "preflop":
                    # Preflop logic: BB=2 is the base. Any bet > BB is a raise.
                    # SB posting 1 is not a raise, BB posting 2 is not a raise.
                    # But voluntarily putting in more than the big blind is a raise.
                    if opp_bet > bb:
                        # They put in more than the BB - this is a raise
                        action_type = "bet_or_raise_to"
                    else:
                        # Just posted blinds or called
                        action_type = "call_or_check"
                else:
                    # Post-flop logic:
                    # Compare opponent's bet to max bet from OTHER players
                    # - If player bet MORE than max_other_bet, it's a bet/raise
                    # - If player bet EQUAL to max_other_bet (calling), it's a call
                    # - If player bet 0 (checked), it's check
                    if opp_bet > max_other_bet and opp_bet > 0:
                        # They bet more than the current bet - this is a raise/bet
                        action_type = "bet_or_raise_to"
                    elif opp_bet > 0 and opp_bet <= max_other_bet:
                        # They matched or bet less than max - this is a call
                        action_type = "call_or_check"
                    else:
                        # No bet (checked)
                        action_type = "call_or_check"

                opp_features = self.feature_engine.calculate_opponent_features(
                    pot_size=pot_size,
                    amount=opp_bet,
                    action_type=action_type,
                    street=street,
                    session=session,
                    player_name=opp_name,
                    player_stack=opp.get("stack", 0),
                    all_stacks=all_stacks,
                    pot_contribution=opp.get("totalCommitted", 0),
                    board_cards=board_cards,
                    raises_override=raises_this_hand,
                    calls_override=calls_this_hand,
                    bb=bb,  # V3: BB normalization
                )
                # Detailed logging for debugging predictions
                logger.info(
                    f"OPPONENT FEATURES {opp_name}: bet={opp_bet}, totalCommitted={opp_total_committed}, "
                    f"action_type={action_type}, "
                    f"isAllIn={opp.get('isAllIn')}, bet_pct_pot={opp_features.get('bet_pct_pot', 0):.2f}, "
                    f"raises_so_far={opp_features.get('raises_so_far')}, calls_so_far={opp_features.get('calls_so_far')}, "
                    f"board_flush_possible={opp_features.get('board_flush_possible')}, "
                    f"board_straight_possible={opp_features.get('board_straight_possible')}"
                )
                opponent_features_list.append((opp, opp_features))

            # Call opponent model for each
            opp_results = await self.model_client.predict_all_opponents(
                street,
                [f for _, f in opponent_features_list],
            )

            for (opp, opp_features), result in zip(opponent_features_list, opp_results):
                pred = result.prediction
                probs = pred.get("probabilities", {"air": 0.33, "middle": 0.34, "nutted": 0.33})

                # Ensure probs is a dict
                if isinstance(probs, list):
                    probs = {"air": probs[0], "middle": probs[1], "nutted": probs[2]} if len(probs) >= 3 else {}

                predicted_class = pred.get("predicted_class", "middle")
                confidence = max(probs.values()) if probs else 0.34

                # Log prediction result with context
                logger.info(
                    f"OPPONENT PREDICTION {opp.get('name')}: predicted={predicted_class}, "
                    f"probs={probs}, fallback={result.from_fallback}, error={result.error}"
                )

                opponent_predictions.append(OpponentPrediction(
                    seat_index=opp.get("seatIndex"),
                    player_name=opp.get("name", ""),
                    predicted_class=predicted_class,
                    probabilities=probs,
                    confidence=confidence,
                    from_fallback=result.from_fallback,
                ))

                if result.from_fallback:
                    any_fallback = True

            latencies["opponent"] = max(r.latency_ms for r in opp_results) if opp_results else 0

        # 3. Calculate profit features (needed for both profit and policy models)
        # Calculate num_players (non-folded players at table)
        num_active_players = len([p for p in all_players if not p.get("isFolded", False) and not p.get("isSittingOut", False)])

        profit_features = self.feature_engine.calculate_profit_features(
            pot_size=pot_size,
            street=street,
            session=session,
            hero_seat=hero_seat,
            hero_name=hero_name,
            hero_stack=hero_stack,
            hero_hole_cards=hero_hole_cards,
            all_stacks=all_stacks,
            board_cards=board_cards,
            active_opponents=[
                {"seat": op.seat_index, "name": op.player_name, "stack": all_stacks[op.seat_index] if op.seat_index < len(all_stacks) else 100.0}
                for op in opponent_predictions
            ],
            opponent_predictions=[
                {"predicted_class": op.predicted_class} for op in opponent_predictions
            ],
            facing_bet=facing_bet,
            bb=bb,
            position_from_button=position_index,
            num_players=num_active_players,
            raises_so_far=raises_this_hand,
            calls_so_far=calls_this_hand,
            action_no_in_hand=action_index + 1,
        )

        # 4. Calculate policy features
        policy_features = self.feature_engine.calculate_policy_features(
            profit_features=profit_features,
            hero_stack=hero_stack,
            pot_size=pot_size,
            facing_bet=facing_bet,
            street=street,
            hero_hole_cards=hero_hole_cards,
            board_cards=board_cards,
            opponent_predictions=[
                {"predicted_class": op.predicted_class} for op in opponent_predictions
            ],
            position_from_button=position_index,
            num_players=num_active_players,
        )

        # 5. Call profit and policy models in PARALLEL (they're independent after opponent results)
        profit_result, policy_result = await asyncio.gather(
            self.model_client.predict_profit(street, profit_features),
            self.model_client.predict_policy(street, policy_features),
        )

        # Process profit result
        latencies["profit"] = profit_result.latency_ms
        if profit_result.from_fallback:
            any_fallback = True

        profit_bb = profit_result.prediction.get("predicted_profit_bb", 0.0)
        profit_prediction = ProfitPrediction(
            predicted_profit_bb=profit_bb,
            interpretation=self._interpret_profit(profit_bb),
            from_fallback=profit_result.from_fallback,
        )

        # Process policy result
        latencies["policy"] = policy_result.latency_ms
        if policy_result.from_fallback:
            any_fallback = True

        pred = policy_result.prediction
        action = pred.get("predicted_action", "call")
        probs = pred.get("probabilities", {})

        # Ensure probs is a dict
        # Note: 3-class model returns ['bet', 'call', 'fold'] in alphabetical order from LabelEncoder
        if isinstance(probs, list):
            actions = ["bet", "call", "fold"]  # Alphabetical order from LabelEncoder
            probs = {a: p for a, p in zip(actions, probs)} if len(probs) >= 3 else {}

        # Custom action selection: if bet probability > 45%, recommend bet
        # This lowers the threshold to encourage more aggressive play
        if probs:
            bet_prob = probs.get("bet", 0)
            if bet_prob > 0.45:
                action = "bet"
            else:
                # Otherwise, pick the action with highest probability
                action = max(probs, key=probs.get)

        chip_amount = self._calculate_chip_amount(action, pot_size, facing_bet, hero_stack)
        confidence = max(probs.values()) if probs else 0.4

        action_recommendation = ActionRecommendation(
            action=action,
            chip_amount=chip_amount,
            probabilities=probs,
            confidence=confidence,
            reasoning=self._generate_reasoning(
                action, profit_bb, opponent_predictions, facing_bet, pot_size
            ),
            from_fallback=policy_result.from_fallback,
        )

        # 5. Build debug info with model inputs and outputs
        opponent_debug = []
        if active_opponents and opponent_features_list:
            for idx, ((opp, opp_features), action_type_for_opp) in enumerate(
                zip(opponent_features_list,
                    [self._infer_action_type(o, street, bb, all_players, hero_seat) for o in active_opponents])
            ):
                raw_output = opp_results[idx].prediction if idx < len(opp_results) else None
                # VPIP/PFR are already in percentage (0-100) from get_historical_features
                # Don't multiply again - just round for display
                vpip_val = opp_features.get("vpip_last5_hist", 50.0)  # Default 50%
                pfr_val = opp_features.get("pfr_last5_hist", 30.0)    # Default 30%
                opponent_debug.append({
                    "playerName": opp.get("name", ""),
                    "features": {
                        # Show ALL key features sent to the model
                        "action_type": action_type_for_opp,
                        "bet_pct_pot": round(opp_features.get("bet_pct_pot", 0), 3),
                        "amount": round(opp.get("betThisStreet", 0) / max(bb, 1), 2),  # Show in BB
                        "pot_size": round(opp_features.get("pot_size", 0), 2),
                        "pot_contribution": round(opp.get("totalCommitted", 0) / max(bb, 1), 2),  # Show in BB
                        "starting_stack": round(opp_features.get("starting_stack", 0), 1),
                        "stack_vs_table_median": round(opp_features.get("stack_vs_table_median", 1.0), 2),
                        "raises_so_far": int(opp_features.get("raises_so_far", 0)),
                        "calls_so_far": int(opp_features.get("calls_so_far", 0)),
                        "isAllIn": opp.get("isAllIn", False),
                        # Historical stats - already in percentage (0-100)
                        "vpip_last5": round(vpip_val, 1),
                        "pfr_last5": round(pfr_val, 1),
                        "agg_factor_last5": round(opp_features.get("agg_factor_last5_hist", 1.0), 2),
                    },
                    "rawOutput": raw_output,
                })

        # Build debug info with specific important fields for display
        # Select key fields explicitly rather than relying on dict order
        profit_debug_keys = [
            "pot_size", "hand_equity", "hand_rank", "starting_stack",
            "facing_call", "pot_odds_call", "bet_pct_pot",
            "opponents_active", "position_from_button", "num_players",
            "opponent_strength_mean", "opponent_nutted_count"
        ]
        # Policy model has different features than Profit model
        # Policy model does NOT have pot_size, facing_call, pot_odds_call, spr
        # It focuses on hand strength, position, and opponent predictions
        policy_debug_keys = [
            "hand_equity", "starting_stack", "stack_vs_table_median",
            "position_from_button", "num_players",
            "opponent_strength_mean", "opponent_strength_max", "opponent_nutted_count",
            "predicted_strength", "strength_advantage"
        ]

        # Log actual feature values before building debug_info
        logger.info(f"DEBUG_INFO: profit_features hand_equity = {profit_features.get('hand_equity', 'MISSING')}")
        logger.info(f"DEBUG_INFO: profit_features pot_size = {profit_features.get('pot_size', 'MISSING')}")
        logger.info(f"DEBUG_INFO: policy_features hand_equity = {policy_features.get('hand_equity', 'MISSING')}")
        logger.info(f"DEBUG_INFO: policy_features pot_size = {policy_features.get('pot_size', 'MISSING')}")

        debug_info = {
            "opponentInputs": opponent_debug,
            "profitInput": {k: round(profit_features.get(k, 0), 4) if isinstance(profit_features.get(k, 0), float) else profit_features.get(k, 0) for k in profit_debug_keys},
            "profitRawOutput": profit_result.prediction,
            "policyInput": {k: round(policy_features.get(k, 0), 4) if isinstance(policy_features.get(k, 0), float) else policy_features.get(k, 0) for k in policy_debug_keys},
            "policyRawOutput": policy_result.prediction,
        }
        logger.info(f"DEBUG_INFO: profitInput = {debug_info['profitInput']}")
        logger.info(f"DEBUG_INFO: policyInput = {debug_info['policyInput']}")

        # 6. Build result
        result = MLPredictionsResult(
            hand_id=hand_id,
            street=street,
            hero_seat=hero_seat,
            action_index=action_index,
            opponent_predictions=opponent_predictions,
            profit_prediction=profit_prediction,
            action_recommendation=action_recommendation,
            latencies_ms=latencies,
            any_from_fallback=any_fallback,
            debug_info=debug_info,
        )

        # 7. Store predictions for review
        self._store_predictions(session, result, profit_features, policy_features)

        return result

    def _interpret_profit(self, profit_bb: float) -> str:
        """Interpret profit prediction."""
        if profit_bb > 5.0:
            return "highly profitable"
        elif profit_bb > 1.0:
            return "profitable"
        elif profit_bb > -1.0:
            return "marginal"
        elif profit_bb > -5.0:
            return "unprofitable"
        else:
            return "highly unprofitable"

    def _calculate_chip_amount(
        self,
        action: str,
        pot_size: float,
        facing_bet: float,
        hero_stack: float,
    ) -> int:
        """Calculate chip amount for recommended action.

        For 3-class model: fold, call, bet
        When action is 'bet', we use a standard 75% pot sizing.
        """
        if action == "fold":
            return 0
        elif action == "call":
            return int(min(facing_bet, hero_stack))
        elif action == "bet":
            # Standard bet sizing: 75% of pot + facing bet (to raise)
            # If no bet to call, this is an open bet of 75% pot
            bet_size = pot_size * 0.75 + facing_bet
            return int(min(bet_size, hero_stack))
        else:
            # Fallback for any unexpected action
            return int(facing_bet)

    def _generate_reasoning(
        self,
        action: str,
        profit_bb: float,
        opponent_predictions: List[OpponentPrediction],
        facing_bet: float,
        pot_size: float,
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        opp_summary = ""
        if opponent_predictions:
            classes = [op.predicted_class for op in opponent_predictions]
            if all(c == "air" for c in classes):
                opp_summary = "Opponents likely have weak hands."
            elif any(c == "nutted" for c in classes):
                opp_summary = "At least one opponent appears strong."
            else:
                opp_summary = "Opponents show mixed strength."

        profit_summary = f"EV: {profit_bb:+.1f} BB."

        # 3-class action model: fold, call, bet
        action_reasons = {
            "fold": "Fold to minimize losses.",
            "call": "Call/check to see more cards.",
            "bet": "Bet or raise for value/pressure.",
        }

        reason = action_reasons.get(action, "Recommended action based on model analysis.")

        parts = [reason]
        if opp_summary:
            parts.append(opp_summary)
        parts.append(profit_summary)

        return " ".join(parts)

    def _store_predictions(
        self,
        session: GameSession,
        result: MLPredictionsResult,
        profit_features: Dict[str, float],
        policy_features: Dict[str, float],
    ) -> None:
        """Store predictions in session for later review."""
        # Store opponent predictions
        for op in result.opponent_predictions:
            session.add_prediction(MLPredictionRecord(
                hand_number=result.hand_id,
                street=result.street,
                action_index=result.action_index,
                prediction_type="opponent",
                target_seat=op.seat_index,
                hero_seat=result.hero_seat,
                features={},  # Don't store all features to save memory
                prediction={
                    "predicted_class": op.predicted_class,
                    "probabilities": op.probabilities,
                    "confidence": op.confidence,
                },
                latency_ms=result.latencies_ms.get("opponent", 0),
                from_fallback=op.from_fallback,
            ))

        # Store profit prediction
        if result.profit_prediction:
            session.add_prediction(MLPredictionRecord(
                hand_number=result.hand_id,
                street=result.street,
                action_index=result.action_index,
                prediction_type="profit",
                target_seat=None,
                hero_seat=result.hero_seat,
                features=profit_features,
                prediction={
                    "predicted_profit_bb": result.profit_prediction.predicted_profit_bb,
                    "interpretation": result.profit_prediction.interpretation,
                },
                latency_ms=result.latencies_ms.get("profit", 0),
                from_fallback=result.profit_prediction.from_fallback,
            ))

        # Store policy prediction
        if result.action_recommendation:
            session.add_prediction(MLPredictionRecord(
                hand_number=result.hand_id,
                street=result.street,
                action_index=result.action_index,
                prediction_type="policy",
                target_seat=None,
                hero_seat=result.hero_seat,
                features=policy_features,
                prediction={
                    "action": result.action_recommendation.action,
                    "chip_amount": result.action_recommendation.chip_amount,
                    "probabilities": result.action_recommendation.probabilities,
                    "confidence": result.action_recommendation.confidence,
                },
                latency_ms=result.latencies_ms.get("policy", 0),
                from_fallback=result.action_recommendation.from_fallback,
            ))

    async def close(self) -> None:
        """Clean up resources."""
        await self.model_client.close()
