"""
Game configuration schema and preset helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

Suit = Literal["♠", "♥", "♦", "♣"]
Rank = Literal["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]


@dataclass
class GameConfig:
    numPlayers: int
    numStreets: int
    buttonStartsAt: int
    blinds: List[int]
    antes: List[int]
    startingStack: int
    raiseCapPerStreet: List[int | Literal["NO_LIMIT", "POT_LIMIT"]]
    maxRaisesPerStreet: List[int | Literal["INFINITE"]]
    suits: Sequence[Suit]
    ranks: Sequence[Rank]
    holeCardsPerPlayer: int
    communityCardsPerStreet: List[int]
    cardsPerHand: int
    minHoleCardsUsedInHand: int


def get_preset_config(name: str) -> GameConfig:
    try:
        return PRESET_CONFIGS[name]
    except KeyError:
        raise ValueError(f"Unknown preset {name}")


PRESET_CONFIGS: dict[str, GameConfig] = {
    "NO_LIMIT_HOLDEM_SIX_MAX": GameConfig(
        numPlayers=6,
        numStreets=4,
        buttonStartsAt=0,
        blinds=[0, 1, 2, 0, 0, 0],
        antes=[0, 0, 0, 0, 0, 0],
        startingStack=200,
        raiseCapPerStreet=["NO_LIMIT", "NO_LIMIT", "NO_LIMIT", "NO_LIMIT"],
        maxRaisesPerStreet=["INFINITE", "INFINITE", "INFINITE", "INFINITE"],
        suits=["♠", "♥", "♦", "♣"],
        ranks=["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"],
        holeCardsPerPlayer=2,
        communityCardsPerStreet=[0, 3, 1, 1],
        cardsPerHand=5,
        minHoleCardsUsedInHand=0,
    ),
    "KUHN_TWO_PLAYER": GameConfig(
        numPlayers=2,
        numStreets=1,
        buttonStartsAt=0,
        blinds=[0, 0],
        antes=[1, 1],
        startingStack=10,
        raiseCapPerStreet=[1],
        maxRaisesPerStreet=[1],
        suits=["♠"],
        ranks=["K", "Q", "J"],
        holeCardsPerPlayer=1,
        communityCardsPerStreet=[0],
        cardsPerHand=1,
        minHoleCardsUsedInHand=1,
    ),
}
