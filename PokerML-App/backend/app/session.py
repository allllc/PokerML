"""
Table session manager bridging FastAPI and the Aurora poker engine.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from itertools import cycle
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = PROJECT_ROOT / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.append(str(ENGINE_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aurora_poker_engine import AuroraPokerEngine, GameConfig
from aurora_poker_engine.types import EngineObservation, LegalAction

from .models import APICard, PublicGameState, PotModel, PublicPlayerView, SidePotModel
from agents import BotPolicy, create_bot_policy, PERSONA_ROTATION


@dataclass
class TableSession:
    table_id: str
    engine: AuroraPokerEngine
    hero_seat: int
    player_names: List[str]
    show_all_cards: bool = False
    bot_policies: Dict[int, BotPolicy] = field(default_factory=dict)
    player_styles: Dict[int, str] = field(default_factory=dict)
    agent_debug: Dict[int, Dict[str, object]] = field(default_factory=dict)
    agent_history: List[Dict[str, object]] = field(default_factory=list)


class TableManager:
    def __init__(self):
        self.tables: Dict[str, TableSession] = {}

    def create_table(
        self,
        table_id: str,
        config: GameConfig,
        hero_seat: int,
        names: List[str],
        bot_policies: Dict[int, BotPolicy],
        player_styles: Dict[int, str],
    ) -> TableSession:
        engine = AuroraPokerEngine(config=config)
        session = TableSession(
            table_id=table_id,
            engine=engine,
            hero_seat=hero_seat,
            player_names=names,
            bot_policies=bot_policies,
            player_styles=player_styles,
            agent_debug={},
        )
        self.tables[table_id] = session
        return session

    def get(self, table_id: str) -> TableSession:
        return self.tables[table_id]


def observation_to_public_state(
    session: TableSession, obs: EngineObservation, show_all: bool
) -> PublicGameState:
    def to_card(card) -> APICard:
        return APICard(rank=card.rank, suit=card.suit, id=card.id)

    board = [to_card(c) for c in obs.boardCards]
    players = []
    for player in obs.players:
        has_button = player.seatIndex == obs.buttonSeat
        is_hero = player.seatIndex == session.hero_seat
        visible = show_all or obs.isTerminal or is_hero
        hole_cards = [to_card(card) for card in player.holeCards] if visible else None
        debug_payload = session.agent_debug.get(player.seatIndex)
        debug_string = json.dumps(debug_payload, default=str, indent=2) if debug_payload else None

        players.append(
            PublicPlayerView(
                seatIndex=player.seatIndex,
                name=session.player_names[player.seatIndex],
                stack=player.stack,
                betThisStreet=player.betThisStreet,
                totalCommitted=player.totalCommitted,
                isFolded=player.isFolded,
                isAllIn=player.isAllIn,
                hasButton=has_button,
                visibility="FACE_UP" if hole_cards else "HIDDEN",
                holeCards=hole_cards,
                styleLabel=session.player_styles.get(player.seatIndex),
                agentDebug=debug_string,
            )
        )

    pot = PotModel(
        mainPot=obs.pot.mainPot,
        sidePots=[SidePotModel(amount=sp.amount, eligibleSeats=sp.eligibleSeats) for sp in obs.pot.sidePots],
    )

    phase = "BETTING"
    street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
    street = street_names[obs.streetIndex] if obs.streetIndex < len(street_names) else None
    if obs.isTerminal:
        phase = "SHOWDOWN"
        street = None

    history_entries = []
    for entry in session.agent_history:
        if entry.get("handId") != session.engine.hand_id:
            continue
        history_entries.append(
            {
                "handId": entry["handId"],
                "seatIndex": entry["seatIndex"],
                "streetIndex": entry["streetIndex"],
                "streetName": entry.get("streetName"),
                "debug": entry.get("debug"),
            }
        )

    return PublicGameState(
        tableId=session.table_id,
        handId=session.engine.hand_id,
        phase=phase,
        streetName=street,
        heroSeat=session.hero_seat,
        activeSeat=None if obs.isTerminal else obs.currentPlayer,
        players=players,
        boardCards=board,
        pot=pot,
        callAmount=obs.callAmount,
        minRaise=obs.minRaise,
        maxRaise=obs.maxRaise,
        legalActions=[action.__dict__ for action in obs.legalActions],
        showAllCards=show_all,
        agentHistory=history_entries,
        winners=obs.winners if obs.isTerminal else None,
        payouts=obs.payouts if obs.isTerminal else None,
    )


def build_names_and_policies(
    config: GameConfig,
    hero_seat: int,
    hero_name: str,
    requested_bot_names: List[str] | None,
):
    names: List[str] = [""] * config.numPlayers
    policies: Dict[int, BotPolicy] = {}
    styles: Dict[int, str] = {}
    names[hero_seat] = hero_name
    styles[hero_seat] = "Hero (You)"
    bot_name_iter = iter(requested_bot_names or [])
    persona_iter = cycle(PERSONA_ROTATION)
    for seat in range(config.numPlayers):
        if seat == hero_seat:
            continue
        persona_name, persona_key = next(persona_iter)
        display_name = next(bot_name_iter, persona_name)
        policy = create_bot_policy(persona_key, display_name, config)
        names[seat] = display_name
        policies[seat] = policy
        styles[seat] = policy.style_label
    return names, policies, styles
