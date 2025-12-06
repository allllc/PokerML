"""
Player state models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .cards import Card


@dataclass
class PlayerState:
    seatIndex: int
    stack: int
    betThisStreet: int = 0
    totalCommitted: int = 0
    isFolded: bool = False
    isAllIn: bool = False
    holeCards: List[Card] = field(default_factory=list)
    isSittingOut: bool = False

    def reset_for_new_street(self) -> None:
        self.betThisStreet = 0

    def commit(self, amount: int) -> int:
        actual = min(amount, self.stack)
        self.stack -= actual
        self.betThisStreet += actual
        self.totalCommitted += actual
        if self.stack == 0 and not self.isFolded:
            self.isAllIn = True
        return actual
