"""
AuroraPokerEngine implementation loosely mirroring the clubs Dealer API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
import random
import logging

from .cards import Card, Deck
from .config import GameConfig
from .evaluator import HandEvaluator
from .player import PlayerState
from .pot import PotState, SidePot
from .types import EngineObservation, LegalAction, StepResult

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def snap_to_valid_bet(requested: int, valid_bets: List[int]) -> int:
    valid = sorted(set(valid_bets))
    closest = valid[0]
    min_diff = abs(requested - closest)
    for value in valid[1:]:
        diff = abs(requested - value)
        if diff < min_diff or (diff == min_diff and value < closest):
            closest = value
            min_diff = diff
    return closest


@dataclass
class HandContext:
    current_player: int = 0
    street_index: int = 0
    max_bet: int = 0
    last_raise_size: int = 0
    is_terminal: bool = False
    winners: Optional[List[int]] = None
    payouts: Optional[List[int]] = None
    last_aggressor: Optional[int] = None  # Seat of last player to bet/raise (None = no aggression yet)


class AuroraPokerEngine:
    def __init__(self, config: GameConfig, rng_seed: Optional[int | str] = None):
        self.config = config
        numeric_seed = (
            rng_seed
            if isinstance(rng_seed, int)
            else random.Random(rng_seed).randint(0, 2**32 - 1)
            if rng_seed is not None
            else random.randint(0, 2**32 - 1)
        )
        self.deck = Deck(config.suits, config.ranks, seed=numeric_seed)
        self.evaluator = HandEvaluator(config.suits, config.ranks)
        self.players = [PlayerState(seatIndex=i, stack=config.startingStack) for i in range(config.numPlayers)]
        self.pot_state = PotState()
        self._pot_layers: List[SidePot] = []
        self.board_cards: List[Card] = []
        self.button = config.buttonStartsAt % config.numPlayers
        self.hand_id = 0
        self.done_flags = [True] * config.numPlayers
        self.rewards = [0] * config.numPlayers
        self.hand_start_stacks = [config.startingStack] * config.numPlayers
        self.context = HandContext()

    def reset(self) -> EngineObservation:
        self.hand_id += 1
        logger.info(f"\n{'='*60}\n  HAND #{self.hand_id} - NEW HAND STARTING\n{'='*60}")
        self._rotate_button_to_active()
        logger.debug(f"Button positioned at seat {self.button}")
        self.pot_state.reset()
        self.board_cards = []
        self.deck.reset()
        self.done_flags = [False] * self.config.numPlayers
        self.rewards = [0] * self.config.numPlayers
        self.hand_start_stacks = [p.stack for p in self.players]
        self._pot_layers = []
        for player in self.players:
            player.betThisStreet = 0
            player.totalCommitted = 0
            player.isFolded = False
            player.isAllIn = player.stack == 0
            player.isSittingOut = player.stack == 0
            player.holeCards.clear()
        self.context = HandContext(street_index=0, last_raise_size=self._initial_raise_size())
        self._post_antes()
        self._post_blinds()
        logger.debug(f"Blinds posted. Pot: {self.pot_state.mainPot}")
        # Big blind is the last aggressor preflop
        bb_seat = (self.button + 2) % self.config.numPlayers
        self.context.last_aggressor = bb_seat
        self._deal_hole_cards()
        self.context.current_player = self._first_to_act(preflop=True)
        logger.info(f"PREFLOP begins. First to act: Seat {self.context.current_player}, last aggressor: Seat {bb_seat}")
        return self._build_observation()

    def step(self, bet: int) -> StepResult:
        if self.context.is_terminal:
            logger.warning(f"step() called on terminal hand. Returning observation.")
            return StepResult(self._build_observation(), self.rewards, self.done_flags)

        player = self.players[self.context.current_player]
        logger.debug(f"\n--- STEP: Seat {self.context.current_player} to act (bet={bet}) ---")
        logger.debug(f"Player state: folded={player.isFolded}, allIn={player.isAllIn}, sitting={player.isSittingOut}, stack={player.stack}")

        # Defensive check: if current player is not active, auto-advance
        # This should not happen if _advance() works correctly, but provides safety
        if player.isFolded or player.isAllIn or player.isSittingOut:
            logger.warning(f"Seat {self.context.current_player} is inactive! Auto-advancing...")
            self._advance()
            return StepResult(self._build_observation(), self.rewards, self.done_flags)

        call_amount = max(0, self.context.max_bet - player.betThisStreet)
        min_raise, max_raise = self._determine_raise_bounds(player, call_amount)
        valid_bets = [0]
        if call_amount > 0:
            valid_bets.append(min(call_amount, player.stack))
        if min_raise is not None and max_raise is not None and player.stack > call_amount:
            upper = min(max_raise, player.stack)
            if upper >= min_raise:
                valid_bets.extend(range(min_raise, upper + 1))
        snapped = snap_to_valid_bet(bet, valid_bets)

        if snapped == 0 and call_amount == 0:
            action = "CHECK"
        elif snapped == 0 and call_amount > 0:
            action = "FOLD"
        elif snapped <= call_amount or player.stack + player.betThisStreet <= call_amount:
            action = "CALL"
            snapped = min(call_amount, player.stack)
        else:
            action = "RAISE"

        if action == "FOLD":
            player.isFolded = True
            logger.info(f"Seat {self.context.current_player}: FOLD")
        elif action in {"CALL", "CHECK"}:
            contribution = min(snapped, player.stack)
            player.commit(contribution)
            self.context.max_bet = max(self.context.max_bet, player.betThisStreet)
            logger.info(f"Seat {self.context.current_player}: {action} (contribution={contribution}, betThisStreet={player.betThisStreet})")
        elif action == "RAISE":
            contribution = min(snapped, player.stack)
            player.commit(contribution)
            self.context.last_raise_size = contribution - call_amount
            self.context.max_bet = max(self.context.max_bet, player.betThisStreet)
            # Player who raises becomes the last aggressor
            self.context.last_aggressor = self.context.current_player
            logger.info(f"Seat {self.context.current_player}: RAISE to {player.betThisStreet} (contribution={contribution}) - now last aggressor")

        self._advance()
        obs = self._build_observation()
        return StepResult(obs, self.rewards, self.done_flags)

    # --- internal helpers ---

    def _initial_raise_size(self) -> int:
        non_zero = [amt for amt in self.config.blinds if amt > 0]
        return max(non_zero) if non_zero else 1

    def _rotate_button_to_active(self) -> None:
        for _ in range(self.config.numPlayers):
            self.button = (self.button + 1) % self.config.numPlayers
            player = self.players[self.button]
            if not player.isSittingOut and player.stack > 0:
                return
        self.button = (self.button + 1) % self.config.numPlayers

    def _post_antes(self) -> None:
        for offset, amount in enumerate(self.config.antes):
            seat = (self.button + offset) % self.config.numPlayers
            player = self.players[seat]
            if amount > 0 and not player.isSittingOut and player.stack > 0:
                contributed = player.commit(amount)
                self.pot_state.mainPot += contributed

    def _post_blinds(self) -> None:
        for offset, amount in enumerate(self.config.blinds):
            seat = (self.button + offset) % self.config.numPlayers
            player = self.players[seat]
            if amount > 0 and player.stack > 0:
                contributed = player.commit(min(amount, player.stack))
                self.context.max_bet = max(self.context.max_bet, player.betThisStreet)
                self.pot_state.mainPot += contributed

    def _deal_hole_cards(self) -> None:
        for _ in range(self.config.holeCardsPerPlayer):
            for player in self.players:
                if not player.isSittingOut and player.stack > 0:
                    player.holeCards.extend(self.deck.draw(1))

    def _first_to_act(self, preflop: bool) -> int:
        if preflop:
            offsets = [idx for idx, amount in enumerate(self.config.blinds) if amount > 0]
            last_blind_offset = max(offsets) if offsets else 1
            start = (self.button + last_blind_offset + 1) % self.config.numPlayers
        else:
            start = (self.button + 1) % self.config.numPlayers
        return self._next_active(start)

    def _next_active(self, seat: int) -> int:
        for i in range(self.config.numPlayers):
            idx = (seat + i) % self.config.numPlayers
            player = self.players[idx]
            if not player.isFolded and not player.isAllIn and not player.isSittingOut:
                return idx
        return seat % self.config.numPlayers

    def _determine_raise_bounds(self, player: PlayerState, call_amount: int) -> tuple[Optional[int], Optional[int]]:
        max_stack_contrib = player.stack
        if max_stack_contrib <= call_amount:
            return None, None
        min_raise = call_amount + max(self.context.last_raise_size, self._initial_raise_size())
        max_raise = call_amount + max_stack_contrib
        return min_raise, max_raise

    def _advance(self) -> None:
        street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]

        if self._betting_round_complete():
            logger.debug(f"Betting round complete on {street_names[min(self.context.street_index, 3)]}")
            self._lock_street()
            logger.debug(f"Street locked. Pot: {self.pot_state.mainPot}")

            # Check if hand should end (last street reached or only 1 player left)
            active_count = self._active_player_count()
            logger.debug(f"Active players: {active_count}, Street index: {self.context.street_index}")

            if self.context.street_index + 1 >= self.config.numStreets or active_count <= 1:
                logger.info(f"Hand ending. Reason: {'last street' if self.context.street_index + 1 >= self.config.numStreets else 'only 1 player left'}")
                self._finish_hand()
                return

            # Move to next street
            self.context.street_index += 1
            self._deal_street_cards(self.context.street_index)
            logger.info(f"\n>>> {street_names[min(self.context.street_index, 3)]} - Board: {[str(c) for c in self.board_cards]}")

            # Check if all remaining players are all-in
            if self._all_active_all_in():
                logger.info("All active players are all-in. Running out remaining streets...")
                while self.context.street_index + 1 < self.config.numStreets:
                    self.context.street_index += 1
                    self._deal_street_cards(self.context.street_index)
                    logger.info(f"  {street_names[min(self.context.street_index, 3)]}: {[str(c) for c in self.board_cards]}")
                self._finish_hand()
                return

            # Reset betting parameters for new street
            self.context.max_bet = 0
            self.context.last_raise_size = self._initial_raise_size()
            self.context.last_aggressor = None  # No aggressor yet on new street
            self.context.current_player = self._first_to_act(preflop=False)
            logger.debug(f"New street betting begins. First to act: Seat {self.context.current_player}, last aggressor: None")
        else:
            old_player = self.context.current_player
            self.context.current_player = self._next_active(self.context.current_player + 1)
            logger.debug(f"Moving to next player: {old_player} -> {self.context.current_player}")

    def _lock_street(self) -> None:
        street_total = 0
        for player in self.players:
            street_total += player.betThisStreet
            player.betThisStreet = 0
        if street_total > 0:
            self.pot_state.mainPot += street_total
        self.context.max_bet = 0

    def _betting_round_complete(self) -> bool:
        active = [p for p in self.players if not p.isFolded and not p.isSittingOut]
        if len(active) <= 1:
            return True

        # Check if all active players have either:
        # 1. Matched the current max_bet, OR
        # 2. Are all-in
        all_matched = True
        for player in active:
            if player.isAllIn:
                continue
            if self.context.max_bet - player.betThisStreet > 0:
                all_matched = False
                break

        if not all_matched:
            return False

        # All players have matched max_bet. But if no one has bet/raised yet (max_bet==0),
        # we need to ensure action has gone around the table at least once.
        # The betting round is complete when:
        # - There's been aggression (last_aggressor != None), and action is back to that player, OR
        # - There's no aggression (last_aggressor == None), and we've completed a full orbit

        if self.context.last_aggressor is not None:
            # There was a bet/raise. Round is complete when action is back to the aggressor.
            # We check if current_player is about to be the aggressor
            next_to_act = self._next_active(self.context.current_player + 1)
            return next_to_act == self.context.last_aggressor or self.players[self.context.last_aggressor].isFolded or self.players[self.context.last_aggressor].isAllIn
        else:
            # No aggression yet (everyone checked). Round is complete when we're back to first position.
            # On new streets (not preflop), first to act is after button
            first_position = self._first_to_act(preflop=False)
            next_to_act = self._next_active(self.context.current_player + 1)
            return next_to_act == first_position

    def _deal_street_cards(self, street_index: int) -> None:
        cards_to_deal = self.config.communityCardsPerStreet[street_index]
        if cards_to_deal > 0:
            self.board_cards.extend(self.deck.draw(cards_to_deal))

    def _all_active_all_in(self) -> bool:
        active = [p for p in self.players if not p.isFolded and not p.isSittingOut]
        return bool(active) and all(player.isAllIn for player in active)

    def _finish_hand(self) -> None:
        remaining = [p for p in self.players if not p.isFolded and not p.isSittingOut]
        logger.info(f"\n{'='*60}\n  FINISHING HAND #{self.hand_id}\n{'='*60}")
        logger.info(f"Remaining players: {[p.seatIndex for p in remaining]}")
        logger.info(f"Board: {[str(c) for c in self.board_cards]}")
        logger.info(f"Pot: {self.pot_state.total()}")

        if len(remaining) <= 1:
            winner = remaining[0] if remaining else None
            if winner:
                logger.info(f"Winner by elimination: Seat {winner.seatIndex} wins {self.pot_state.total()}")
                winner.stack += self.pot_state.total()
                self.context.winners = [winner.seatIndex]
            else:
                logger.warning("No remaining players!")
            self._finalize_rewards()
            return
        logger.info("Going to showdown...")
        self._showdown()

    def _showdown(self) -> None:
        self._build_side_pots()
        pots = self._pot_layers or [SidePot(self.pot_state.mainPot, [p.seatIndex for p in self.players])]
        for pot_index, side_pot in enumerate(pots):
            eligible = [seat for seat in side_pot.eligibleSeats if not self.players[seat].isFolded]
            if not eligible or side_pot.amount == 0:
                continue
            winners = self._evaluate_winners(eligible)
            share = side_pot.amount // len(winners)
            remainder = side_pot.amount % len(winners)
            for idx, seat in enumerate(winners):
                payout = share + (1 if idx < remainder else 0)
                self.players[seat].stack += payout
        self.context.winners = self._evaluate_winners([p.seatIndex for p in self.players if not p.isFolded])
        self._finalize_rewards()

    def _build_side_pots(self) -> None:
        contributions = {player.seatIndex: player.totalCommitted for player in self.players}
        pots: List[SidePot] = []
        while True:
            eligible = [seat for seat, amount in contributions.items() if amount > 0]
            if not eligible:
                break
            min_commit = min(contributions[seat] for seat in eligible)
            amount = min_commit * len(eligible)
            for seat in eligible:
                contributions[seat] -= min_commit
            pots.append(SidePot(amount=amount, eligibleSeats=eligible.copy()))
        self._pot_layers = pots
        if pots:
            self.pot_state.mainPot = pots[0].amount
            self.pot_state.sidePots = pots[1:]
        else:
            self.pot_state.mainPot = 0
            self.pot_state.sidePots = []

    def _evaluate_winners(self, seats: List[int]) -> List[int]:
        best_score = None
        winners: List[int] = []
        for seat in seats:
            player = self.players[seat]
            if player.isFolded or player.isSittingOut:
                continue
            score = self._best_hand_score(player)
            if score is None:
                continue
            if best_score is None or score.as_score() > best_score.as_score():
                best_score = score
                winners = [seat]
            elif score.as_score() == best_score.as_score():
                winners.append(seat)
        return winners

    def _best_hand_score(self, player: PlayerState):
        total_cards = player.holeCards + self.board_cards
        if len(total_cards) < self.config.cardsPerHand:
            return None
        best = None
        hole = player.holeCards
        min_hole = self.config.minHoleCardsUsedInHand
        max_hole = min(len(hole), self.config.cardsPerHand)
        from itertools import combinations

        for hole_count in range(min_hole, max_hole + 1):
            for hole_choice in combinations(hole, hole_count):
                board_needed = self.config.cardsPerHand - hole_count
                if board_needed > len(self.board_cards):
                    continue
                for board_choice in combinations(self.board_cards, board_needed):
                    score = self.evaluator.evaluate(list(hole_choice) + list(board_choice))
                    if best is None or score.as_score() > best.as_score():
                        best = score
        return best

    def _finalize_rewards(self) -> None:
        logger.info("\nFinalizing hand rewards:")
        for idx, player in enumerate(self.players):
            self.rewards[idx] = player.stack - self.hand_start_stacks[idx]
            self.done_flags[idx] = True
            player.isSittingOut = player.stack <= 0
            if self.rewards[idx] != 0:
                logger.info(f"  Seat {idx}: {'+' if self.rewards[idx] > 0 else ''}{self.rewards[idx]} (stack: {self.hand_start_stacks[idx]} -> {player.stack})")
        self.context.is_terminal = True
        self.context.current_player = 0
        logger.info(f"Hand #{self.hand_id} is now TERMINAL\n{'='*60}\n")

    def _active_player_count(self) -> int:
        return len([p for p in self.players if not p.isFolded and not p.isSittingOut])

    def _build_observation(self) -> EngineObservation:
        player_copy = deepcopy(self.players)
        pot_copy = PotState(self.pot_state.mainPot, deepcopy(self.pot_state.sidePots))
        call_amount = max(0, self.context.max_bet - self.players[self.context.current_player].betThisStreet)
        min_raise, max_raise = self._determine_raise_bounds(self.players[self.context.current_player], call_amount)
        legal_actions = self._legal_actions(self.players[self.context.current_player], call_amount, min_raise, max_raise)
        return EngineObservation(
            currentPlayer=self.context.current_player,
            streetIndex=self.context.street_index,
            pot=pot_copy,
            buttonSeat=self.button,
            callAmount=call_amount,
            minRaise=min_raise,
            maxRaise=max_raise,
            legalActions=legal_actions,
            players=player_copy,
            boardCards=list(self.board_cards),
            deckRemainingCount=self.deck.remaining(),
            isTerminal=self.context.is_terminal,
            winners=self.context.winners,
            payouts=self.rewards if self.context.is_terminal else None,
        )

    def _legal_actions(
        self, player: PlayerState, call_amount: int, min_raise: Optional[int], max_raise: Optional[int]
    ) -> List[LegalAction]:
        actions: List[LegalAction] = []
        if call_amount == 0:
            actions.append(LegalAction(type="CHECK"))
        else:
            actions.append(LegalAction(type="FOLD"))
            actions.append(LegalAction(type="CALL", amount=min(call_amount, player.stack)))
        if min_raise is not None and max_raise is not None and player.stack > call_amount:
            actions.append(
                LegalAction(
                    type="RAISE",
                    min=min_raise,
                    max=min(max_raise, player.stack),
                )
            )
        return actions

    def observe(self) -> EngineObservation:
        """
        Public helper for retrieving the most recent observation snapshot.
        """
        return self._build_observation()
