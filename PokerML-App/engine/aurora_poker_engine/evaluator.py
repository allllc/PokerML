"""
Generic 5-card evaluator inspired by Deuces but implemented in pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence, Tuple

from .cards import Card, Rank, Suit


@dataclass
class HandEvaluation:
    category: int
    kickers: Tuple[int, ...]

    def as_score(self) -> Tuple[int, Tuple[int, ...]]:
        return (self.category, self.kickers)


class HandEvaluator:
    CATEGORY_NAMES = [
        "high_card",
        "pair",
        "two_pair",
        "three_of_a_kind",
        "straight",
        "flush",
        "full_house",
        "four_of_a_kind",
        "straight_flush",
    ]

    def __init__(self, suits: Sequence[Suit], ranks: Sequence[Rank]):
        self.rank_to_value = {rank: idx for idx, rank in enumerate(ranks)}
        self.value_to_rank = {idx: rank for rank, idx in self.rank_to_value.items()}
        self.max_rank_value = max(self.rank_to_value.values())

    def evaluate(self, cards: Iterable[Card]) -> HandEvaluation:
        combo = list(cards)
        if len(combo) != 5:
            raise ValueError("Evaluator expects exactly 5 cards")
        values = sorted([self.rank_to_value[card.rank] for card in combo], reverse=True)
        suits = [card.suit for card in combo]
        value_counts = {v: values.count(v) for v in set(values)}
        counts_sorted = sorted(value_counts.items(), key=lambda kv: (-kv[1], -kv[0]))

        is_flush = len(set(suits)) == 1
        is_straight, straight_high = self._detect_straight(values)

        if is_straight and is_flush:
            return HandEvaluation(8, (straight_high,))
        if counts_sorted[0][1] == 4:
            four_val = counts_sorted[0][0]
            kicker = max(v for v in values if v != four_val)
            return HandEvaluation(7, (four_val, kicker))
        if counts_sorted[0][1] == 3 and counts_sorted[1][1] == 2:
            return HandEvaluation(6, (counts_sorted[0][0], counts_sorted[1][0]))
        if is_flush:
            return HandEvaluation(5, tuple(values))
        if is_straight:
            return HandEvaluation(4, (straight_high,))
        if counts_sorted[0][1] == 3:
            trips = counts_sorted[0][0]
            kickers = [v for v in values if v != trips]
            return HandEvaluation(3, (trips, *kickers))
        if counts_sorted[0][1] == 2 and counts_sorted[1][1] == 2:
            high_pair = max(counts_sorted[0][0], counts_sorted[1][0])
            low_pair = min(counts_sorted[0][0], counts_sorted[1][0])
            kicker = max(v for v in values if v not in (high_pair, low_pair))
            return HandEvaluation(2, (high_pair, low_pair, kicker))
        if counts_sorted[0][1] == 2:
            pair = counts_sorted[0][0]
            kickers = [v for v in values if v != pair]
            return HandEvaluation(1, (pair, *kickers))
        return HandEvaluation(0, tuple(values))

    def evaluate_best(self, cards: Iterable[Card]) -> HandEvaluation:
        best: HandEvaluation | None = None
        for combo in combinations(cards, 5):
            score = self.evaluate(combo)
            if best is None or score.as_score() > best.as_score():
                best = score
        if best is None:
            raise ValueError("Need at least 5 cards to evaluate")
        return best

    def compare(self, cards_a: Iterable[Card], cards_b: Iterable[Card]) -> int:
        score_a = self.evaluate(cards_a).as_score()
        score_b = self.evaluate(cards_b).as_score()
        if score_a > score_b:
            return 1
        if score_a < score_b:
            return -1
        return 0

    def _detect_straight(self, values: list[int]) -> tuple[bool, int]:
        unique = sorted(set(values), reverse=True)
        if len(unique) < 5:
            return False, -1
        # handle wheel A-2-3-4-5
        wheel = [4, 3, 2, 1, 0]
        if set(unique[-5:]) == set(wheel) and unique[0] == self.max_rank_value:
            return True, 3  # treat 5-high straight, kicker = 3 (value for "5")
        for start in range(len(unique) - 4):
            window = unique[start:start + 5]
            if window[0] - window[-1] == 4 and len(set(window)) == 5:
                return True, window[0]
        return False, -1
