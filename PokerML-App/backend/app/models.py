"""
Pydantic models shared by the FastAPI layer.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class APICard(BaseModel):
    rank: str
    suit: str
    id: int


class SidePotModel(BaseModel):
    amount: int
    eligibleSeats: list[int]


class PotModel(BaseModel):
    mainPot: int
    sidePots: list[SidePotModel]


class PublicPlayerView(BaseModel):
    seatIndex: int
    name: str
    stack: int
    betThisStreet: int
    totalCommitted: int
    isFolded: bool
    isAllIn: bool
    hasButton: bool
    visibility: Literal["HIDDEN", "FACE_UP"]
    holeCards: Optional[list[APICard]] = None
    styleLabel: Optional[str] = None
    agentDebug: Optional[str] = None


class AgentHistoryEntry(BaseModel):
    handId: int
    seatIndex: int
    streetIndex: int
    streetName: Optional[str]
    debug: dict


class PublicGameState(BaseModel):
    tableId: str
    handId: int
    phase: Literal["PRE_HAND", "BETTING", "SHOWDOWN", "FINISHED"]
    streetName: Optional[Literal["PREFLOP", "FLOP", "TURN", "RIVER"]]
    heroSeat: int
    activeSeat: Optional[int]
    players: list[PublicPlayerView]
    boardCards: list[APICard]
    pot: PotModel
    callAmount: Optional[int]
    minRaise: Optional[int]
    maxRaise: Optional[int]
    legalActions: list[dict]
    showAllCards: bool
    agentHistory: list[AgentHistoryEntry]
    winners: Optional[list[int]] = None  # Seat indices of winners
    payouts: Optional[list[int]] = None  # Payout amounts per seat


class CreateTableRequest(BaseModel):
    configPreset: str = "NO_LIMIT_HOLDEM_SIX_MAX"
    heroSeat: int = 0
    heroName: str = "Hero"
    botNames: list[str] | None = None


class CreateTableResponse(BaseModel):
    tableId: str
    state: PublicGameState


class StartHandRequest(BaseModel):
    autoAdvance: bool = Field(True, description="Auto-play bots until hero acts")


class ActionRequest(BaseModel):
    viewerSeat: int
    betAmount: int
    autoAdvance: bool = True


class SettingsRequest(BaseModel):
    showAllCards: bool
