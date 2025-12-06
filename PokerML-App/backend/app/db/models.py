"""
Database record models for PostgreSQL.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid


@dataclass
class SessionRecord:
    """Session database record."""
    session_id: str
    player_id: str
    table_id: Optional[str] = None
    config_preset: Optional[str] = None
    hero_seat: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    is_active: bool = True

    @staticmethod
    def new(table_id: str, hero_seat: int, config_preset: str = "NO_LIMIT_HOLDEM_SIX_MAX") -> "SessionRecord":
        """Create a new session record."""
        return SessionRecord(
            session_id=str(uuid.uuid4()),
            player_id=str(uuid.uuid4()),
            table_id=table_id,
            config_preset=config_preset,
            hero_seat=hero_seat,
        )


@dataclass
class HandRecord:
    """Hand database record."""
    session_id: str
    hand_number: int
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    bb: Optional[float] = None
    sb: Optional[float] = None
    board_cards: Optional[str] = None  # e.g., "AhKd7c2s9h"
    final_pot: Optional[int] = None
    winners: Optional[List[int]] = None
    payouts: Optional[List[int]] = None


@dataclass
class ActionRecord:
    """Action database record."""
    session_id: str
    hand_number: int
    action_index: int
    street: str
    actor_seat: int
    actor_name: Optional[str] = None
    action_type: str = ""  # fold, call_or_check, bet_or_raise_to
    amount: Optional[float] = None
    pot_before: Optional[float] = None
    pot_after: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionRecord:
    """Prediction database record."""
    session_id: str
    hand_number: int
    street: str
    prediction_type: str  # opponent, profit, policy
    hero_seat: int
    features: Dict[str, Any] = field(default_factory=dict)
    prediction: Dict[str, Any] = field(default_factory=dict)
    action_index: Optional[int] = None
    target_seat: Optional[int] = None  # For opponent predictions
    model_endpoint: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PlayerStatsRecord:
    """Player session stats database record."""
    session_id: str
    player_name: str
    seat_index: Optional[int] = None
    hands_played: int = 0
    voluntary_preflop: int = 0
    preflop_raise: int = 0
    aggressive_actions: int = 0
    passive_actions: int = 0
    saw_flop: int = 0
    saw_turn: int = 0
    saw_river: int = 0
    showdown_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
