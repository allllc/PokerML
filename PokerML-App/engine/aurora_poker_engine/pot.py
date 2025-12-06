"""
Pot state helper classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SidePot:
    amount: int
    eligibleSeats: List[int]


@dataclass
class PotState:
    mainPot: int = 0
    sidePots: List[SidePot] = field(default_factory=list)

    def total(self) -> int:
        return self.mainPot + sum(p.amount for p in self.sidePots)

    def reset(self) -> None:
        self.mainPot = 0
        self.sidePots.clear()
