"""
Card and deck primitives with deterministic shuffle support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import random

Suit = str
Rank = str


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank
    id: int

    def short_name(self) -> str:
        return f"{self.rank}{self.suit}"


class Deck:
    """
    Deterministic deck representation that accepts subsets of ranks/suits and
    supports deterministic shuffling via seedable RNG instances.
    """

    def __init__(self, suits: Sequence[Suit], ranks: Sequence[Rank], seed: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self.suits = list(suits)
        self.ranks = list(ranks)
        self.cards: List[Card] = []
        self._build_deck()
        self.shuffle()

    def _build_deck(self) -> None:
        card_id = 0
        self.cards.clear()
        for rank in self.ranks:
            for suit in self.suits:
                self.cards.append(Card(suit=suit, rank=rank, id=card_id))
                card_id += 1

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)

    def reset(self) -> None:
        # Don't reseed! We want different cards each hand.
        # Only rebuild and shuffle using the existing RNG state.
        self._build_deck()
        self.shuffle()

    def draw(self, n: int = 1) -> list[Card]:
        drawn = []
        for _ in range(min(n, len(self.cards))):
            drawn.append(self.cards.pop(0))
        return drawn

    def remaining(self) -> int:
        return len(self.cards)
