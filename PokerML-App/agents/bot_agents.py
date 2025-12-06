"""
Poker bot personas backed by heuristics and optional OpenAI reasoning.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

from langchain_openai import ChatOpenAI

from .poker_advisor import PLAYER_NAMES

# agents/ is at project root, engine/ is sibling folder
ENGINE_PATH = Path(__file__).resolve().parent.parent / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.append(str(ENGINE_PATH))

from aurora_poker_engine.types import EngineObservation  # noqa: E402
from aurora_poker_engine.evaluator import HandEvaluator  # noqa: E402
from aurora_poker_engine.cards import Card  # noqa: E402
from aurora_poker_engine.config import GameConfig  # noqa: E402

PersonaKey = Literal["expert_pot", "expert_pressure", "cautious", "caller", "caller_loose"]
OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))

ENABLE_OPENAI = os.getenv("ENABLE_OPENAI_BOTS", "false").lower() == "true" and OPENAI_AVAILABLE

# Use neutral names that don't indicate playing style
# Names are shuffled at module load to vary between sessions
_shuffled_names = PLAYER_NAMES.copy()
random.shuffle(_shuffled_names)

PERSONA_ROTATION: list[tuple[str, PersonaKey]] = [
    (_shuffled_names[0], "expert_pot"),
    (_shuffled_names[1], "expert_pressure"),
    (_shuffled_names[2], "cautious"),
    (_shuffled_names[3], "caller"),
    (_shuffled_names[4], "caller_loose"),
]

PERSONA_LABELS: Dict[PersonaKey, str] = {
    "expert_pot": "Expert – pot control",
    "expert_pressure": "Expert – pressure",
    "cautious": "Newbie – cautious",
    "caller": "Newbie – caller",
    "caller_loose": "Newbie – loose caller",
}


def cards_to_text(cards: List[Card]) -> str:
    return " ".join(f"{card.rank}{card.suit}" for card in cards)


def parse_card_text(text: str) -> List[Card]:
    cards: List[Card] = []
    for token in text.split():
        if len(token) < 2:
            continue
        rank, suit = token[:-1], token[-1]
        cards.append(Card(rank=rank, suit=suit, id=len(cards)))
    return cards


def compute_strength(
    hole_text: str,
    board_text: str,
    evaluator: HandEvaluator,
    rank_order: Dict[str, int],
    cards_per_hand: int,
) -> Dict[str, int | str]:
    hole_cards = parse_card_text(hole_text)
    board_cards = parse_card_text(board_text)
    all_cards = hole_cards + board_cards
    category = 0
    label = "unknown"
    if len(all_cards) >= cards_per_hand:
        evaluation = evaluator.evaluate_best(all_cards)
        category = evaluation.category
        labels = [
            "High Card",
            "Pair",
            "Two Pair",
            "Trips",
            "Straight",
            "Flush",
            "Full House",
            "Quads",
            "Straight Flush",
        ]
        label = labels[min(category, len(labels) - 1)]

    flush_danger = 0
    straight_danger = 0
    if board_cards:
        suit_counts: Dict[str, int] = {}
        for card in board_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        flush_danger = max(suit_counts.values())

        rank_values = sorted(rank_order.get(card.rank, 0) for card in board_cards)
        run = 1
        for i in range(1, len(rank_values)):
            if rank_values[i] == rank_values[i - 1] + 1:
                run += 1
            elif rank_values[i] != rank_values[i - 1]:
                run = 1
            straight_danger = max(straight_danger, run)

    return {
        "category": category,
        "label": label,
        "flush_danger": flush_danger,
        "straight_danger": straight_danger,
    }


def describe_state(
    obs: EngineObservation,
    seat_index: int,
    evaluator: HandEvaluator,
    rank_order: Dict[str, int],
    config: GameConfig,
) -> Dict[str, object]:
    player = obs.players[seat_index]
    side_pot_total = sum(sp.amount for sp in obs.pot.sidePots)
    streets = ["PREFLOP", "FLOP", "TURN", "RIVER"]
    street_name = streets[min(obs.streetIndex, len(streets) - 1)]
    stacks = {p.seatIndex: p.stack for p in obs.players}
    commits = {p.seatIndex: p.totalCommitted for p in obs.players}
    active = [p.seatIndex for p in obs.players if not p.isFolded and not p.isSittingOut]
    position = (seat_index - obs.buttonSeat) % len(obs.players)
    board_text = cards_to_text(obs.boardCards)
    hole_text = cards_to_text(player.holeCards)
    strength = compute_strength(hole_text, board_text, evaluator, rank_order, config.cardsPerHand)
    return {
        "call": obs.callAmount or 0,
        "min_raise": obs.minRaise,
        "max_raise": obs.maxRaise,
        "stack": player.stack,
        "pot_total": obs.pot.mainPot + side_pot_total,
        "street": street_name,
        "board_cards": board_text,
        "hole_cards": hole_text,
        "stacks": stacks,
        "commits": commits,
        "active_players": active,
        "position": position,
        "strength": strength,
    }


class BotPolicy:
    def choose_bet(self, obs: EngineObservation, seat_index: int) -> tuple[int, Dict[str, object]]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class HeuristicBotPolicy(BotPolicy):
    persona: PersonaKey
    display_name: str
    style_label: str
    config: GameConfig

    def __post_init__(self) -> None:
        self.evaluator = HandEvaluator(self.config.suits, self.config.ranks)
        self.rank_order = {rank: idx for idx, rank in enumerate(self.config.ranks)}

    def choose_bet(self, obs: EngineObservation, seat_index: int) -> tuple[int, Dict[str, object]]:
        features = describe_state(obs, seat_index, self.evaluator, self.rank_order, self.config)
        strength = features["strength"]
        category = int(strength["category"])
        flush_danger = int(strength["flush_danger"])
        straight_danger = int(strength["straight_danger"])
        call_amount = int(features["call"])
        min_raise = features["min_raise"]
        max_raise = features["max_raise"]
        stack = max(int(features["stack"]), 1)
        pot_total = int(features["pot_total"])
        street = str(features["street"])
        position = int(features["position"])
        active_count = len(features["active_players"])
        pressure = call_amount / stack if stack else 1
        is_board_scary = flush_danger >= 4 or straight_danger >= 4
        strong_category = category >= 4

        bet = 0
        reasoning = ""
        style = self.persona

        if style == "expert_pot":
            if call_amount == 0:
                if min_raise and pot_total < stack * 0.3 and strong_category:
                    bet = int(min_raise)
                    reasoning = "Pot small and hand strong → apply pressure."
                else:
                    reasoning = "Check to control pot."
            else:
                if pressure < 0.25 or (street in {"TURN", "RIVER"} and active_count <= 2 and not is_board_scary):
                    bet = call_amount
                    reasoning = "Continue for affordable price."
                elif min_raise and pressure < 0.18 and not is_board_scary:
                    bet = int(min_raise)
                    reasoning = "Balanced raise when pressure manageable."
                else:
                    reasoning = "Fold/check due to high pressure."
        elif style == "expert_pressure":
            if call_amount == 0 and min_raise:
                size = pot_total // (2 if strong_category else 4) or min_raise
                bet = int(max(min_raise, size))
                reasoning = "Lead to seize initiative."
            else:
                if pressure < 0.15:
                    bet = call_amount
                    reasoning = "Cheap call keeps range wide."
                elif (
                    min_raise
                    and pressure < 0.35
                    and street in {"FLOP", "TURN"}
                    and (strong_category or not is_board_scary)
                ):
                    bet = int(min_raise)
                    reasoning = "Apply pressure with semi-bluff."
                elif min_raise and pressure < 0.5 and position >= active_count - 1:
                    bet = int(min_raise)
                    reasoning = "Late position stab."
                else:
                    reasoning = "Fold/check due to pressure."
        elif style == "cautious":
            if call_amount == 0:
                bet = 0
                reasoning = "Prefer checking marginal hands."
            else:
                bet = call_amount if call_amount <= max(2, stack * 0.05) else 0
                reasoning = "Only continue for cheap price."
        elif style == "caller":
            if call_amount == 0:
                bet = 0
                reasoning = "Check option."
            else:
                bet = call_amount if random.random() < 0.85 else 0
                reasoning = "Loose caller continues most of the time."
        else:  # caller_loose
            if call_amount == 0:
                bet = min_raise or 0 if random.random() < 0.1 else 0
                reasoning = "Occasional stab when unchecked."
            else:
                if random.random() < 0.95:
                    bet = call_amount
                    reasoning = "Loose caller continues."
                elif min_raise:
                    bet = int(min_raise)
                    reasoning = "Occasional raise to mix strategy."
                else:
                    bet = call_amount
                    reasoning = "Fallback to call."

        legal = obs.legalActions
        call_action = next((act for act in legal if act.type == "CALL"), None)
        raise_action = next((act for act in legal if act.type == "RAISE"), None)
        can_check = any(act.type == "CHECK" for act in legal)

        if bet <= 0:
            final_bet = 0  # check or fold handled by engine
            final_action = "check" if call_amount == 0 and can_check else "fold"
        elif call_action and bet <= call_action.amount:
            final_bet = call_action.amount
            final_action = "call"
        elif raise_action:
            final_bet = max(raise_action.min, min(bet, raise_action.max))
            final_action = "raise"
        else:
            final_bet = call_action.amount if call_action else 0
            final_action = "call" if final_bet else "fold"

        debug = {
            "persona": self.style_label,
            "features": features,
            "decision": reasoning,
            "final_action": final_action,
            "final_bet": final_bet,
            "source": "heuristic",
        }
        return final_bet, debug


@dataclass
class OpenAIAgentPolicy(BotPolicy):
    persona: PersonaKey
    display_name: str
    style_label: str
    config: GameConfig

    def __post_init__(self) -> None:
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OPENAI_API_KEY not configured.")
        self.evaluator = HandEvaluator(self.config.suits, self.config.ranks)
        self.rank_order = {rank: idx for idx, rank in enumerate(self.config.ranks)}
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
        )
        self.fallback = HeuristicBotPolicy(
            persona=self.persona,
            display_name=self.display_name,
            style_label=self.style_label,
            config=self.config,
        )

    def choose_bet(self, obs: EngineObservation, seat_index: int) -> tuple[int, Dict[str, object]]:
        features = describe_state(obs, seat_index, self.evaluator, self.rank_order, self.config)
        call_amount = int(features["call"])
        min_raise = features["min_raise"]
        max_raise = features["max_raise"]
        can_check = any(act.type == "CHECK" for act in obs.legalActions)
        call_action = next((act for act in obs.legalActions if act.type == "CALL"), None)
        raise_action = next((act for act in obs.legalActions if act.type == "RAISE"), None)

        instructions = (
            "You are an expert No-Limit Hold'em player. Decide whether to fold, check, call, or raise. "
            "Respond with JSON: {\"action\": \"fold|check|call|raise\", \"amount\": int, \"reasoning\": \"...\"}. "
            "Amount represents total chips to commit this action (use call amount for calls, a number between "
            "min_raise and max_raise for raises). Prefer folding when board texture, price, or position is dangerous."
        )
        state_payload = {
            "persona": self.style_label,
            "state": features,
        }
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(state_payload, ensure_ascii=False)},
        ]

        try:
            response = self.llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
            data = json.loads(text)
            action = str(data.get("action", "fold")).lower()
            amount = int(data.get("amount", 0))
            reasoning = data.get("reasoning", "LLM decision.")
        except Exception as exc:  # pragma: no cover - best-effort
            fallback_bet, fallback_debug = self.fallback.choose_bet(obs, seat_index)
            fallback_debug["llm_error"] = str(exc)
            fallback_debug["source"] = "heuristic (fallback)"
            return fallback_bet, fallback_debug

        if action == "fold":
            desired_bet = 0
            final_action = "fold"
        elif action == "check":
            desired_bet = 0
            final_action = "check"
        elif action == "call":
            desired_bet = call_amount
            final_action = "call"
        elif action == "raise":
            desired_bet = amount if amount > 0 else (min_raise or call_amount)
            final_action = "raise"
        else:
            desired_bet = call_amount
            final_action = "call"

        if desired_bet <= 0:
            final_bet = 0 if call_amount == 0 and can_check else 0
            final_action = "check" if call_amount == 0 else "fold"
        elif call_action and desired_bet <= call_action.amount:
            final_bet = call_action.amount
            final_action = "call"
        elif raise_action:
            upper = min(raise_action.max, max_raise or raise_action.max)
            final_bet = max(raise_action.min, min(desired_bet, upper))
            final_action = "raise"
        else:
            final_bet = call_action.amount if call_action else 0
            final_action = "call" if final_bet else "fold"

        debug = {
            "persona": self.style_label,
            "features": features,
            "decision": reasoning,
            "final_action": final_action,
            "final_bet": final_bet,
            "llm_output": data,
            "source": "openai",
        }
        return final_bet, debug


def create_bot_policy(persona: PersonaKey, display_name: str, config: GameConfig) -> BotPolicy:
    style_label = PERSONA_LABELS.get(persona, persona.replace("_", " ").title())
    if ENABLE_OPENAI:
        return OpenAIAgentPolicy(persona=persona, display_name=display_name, style_label=style_label, config=config)
    return HeuristicBotPolicy(persona=persona, display_name=display_name, style_label=style_label, config=config)
