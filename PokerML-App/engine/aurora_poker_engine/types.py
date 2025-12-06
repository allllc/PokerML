"""
Structured types for observations/actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .player import PlayerState
from .pot import PotState


LegalActionType = Literal["FOLD", "CHECK", "CALL", "RAISE"]


@dataclass
class LegalAction:
    type: LegalActionType
    min: Optional[int] = None
    max: Optional[int] = None
    amount: Optional[int] = None


@dataclass
class EngineObservation:
    currentPlayer: int
    streetIndex: int
    pot: PotState
    buttonSeat: int
    callAmount: int
    minRaise: Optional[int]
    maxRaise: Optional[int]
    legalActions: List[LegalAction]
    players: List[PlayerState]
    boardCards: list
    deckRemainingCount: int
    isTerminal: bool
    winners: Optional[List[int]] = None
    payouts: Optional[List[int]] = None


@dataclass
class StepResult:
    observation: EngineObservation
    rewards: List[int]
    done: List[bool]
