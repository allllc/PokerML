"""
PostgreSQL database layer using asyncpg.

Provides async database operations for Cloud SQL PostgreSQL.
Falls back gracefully when database is not configured.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import json
import logging

logger = logging.getLogger(__name__)

# Optional asyncpg import - database features disabled if not available
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not installed - database features disabled")

from .models import (
    SessionRecord,
    HandRecord,
    ActionRecord,
    PredictionRecord,
    PlayerStatsRecord,
)


# SQL schema for initialization
SCHEMA_SQL = """
-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id UUID PRIMARY KEY,
    player_id UUID NOT NULL,
    table_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    config_preset VARCHAR(50),
    hero_seat INT,
    is_active BOOLEAN DEFAULT TRUE
);

-- Hands table
CREATE TABLE IF NOT EXISTS hands (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    hand_number INT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    bb DECIMAL(10,2),
    sb DECIMAL(10,2),
    board_cards VARCHAR(20),
    final_pot INT,
    winners INT[],
    payouts INT[],
    UNIQUE(session_id, hand_number)
);

-- Actions table
CREATE TABLE IF NOT EXISTS actions (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    hand_number INT NOT NULL,
    action_index INT NOT NULL,
    street VARCHAR(10) NOT NULL,
    actor_seat INT NOT NULL,
    actor_name VARCHAR(100),
    action_type VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2),
    pot_before DECIMAL(10,2),
    pot_after DECIMAL(10,2),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, hand_number, action_index)
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    hand_number INT NOT NULL,
    street VARCHAR(10) NOT NULL,
    action_index INT,
    prediction_type VARCHAR(20) NOT NULL,
    target_seat INT,
    hero_seat INT NOT NULL,
    features JSONB NOT NULL,
    prediction JSONB NOT NULL,
    model_endpoint VARCHAR(200),
    latency_ms DECIMAL(8,2),
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Player stats table
CREATE TABLE IF NOT EXISTS player_session_stats (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(session_id) ON DELETE CASCADE,
    player_name VARCHAR(100) NOT NULL,
    seat_index INT,
    hands_played INT DEFAULT 0,
    voluntary_preflop INT DEFAULT 0,
    preflop_raise INT DEFAULT 0,
    aggressive_actions INT DEFAULT 0,
    passive_actions INT DEFAULT 0,
    saw_flop INT DEFAULT 0,
    saw_turn INT DEFAULT 0,
    saw_river INT DEFAULT 0,
    showdown_count INT DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, player_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id);
CREATE INDEX IF NOT EXISTS idx_actions_session_hand ON actions(session_id, hand_number);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id, hand_number);
CREATE INDEX IF NOT EXISTS idx_player_stats_session ON player_session_stats(session_id);
"""


class Database:
    """
    Async PostgreSQL database interface.

    Provides CRUD operations for poker ML data with automatic
    fallback when database is not configured.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            database_url: PostgreSQL connection string.
                         If None, reads from DATABASE_URL env var.
        """
        self._url = database_url or os.getenv("DATABASE_URL", "")
        self._pool: Optional[Any] = None
        self._enabled = bool(self._url) and ASYNCPG_AVAILABLE

        if not self._enabled:
            if not ASYNCPG_AVAILABLE:
                logger.info("Database disabled: asyncpg not installed")
            else:
                logger.info("Database disabled: DATABASE_URL not configured")

    @property
    def is_enabled(self) -> bool:
        """Check if database is enabled."""
        return self._enabled

    async def connect(self) -> bool:
        """
        Connect to the database.

        Returns:
            True if connected successfully, False otherwise.
        """
        if not self._enabled:
            return False

        try:
            # Parse the URL to extract components for asyncpg
            # asyncpg has issues with certain URL formats, so we pass params directly
            from urllib.parse import urlparse, unquote

            parsed = urlparse(self._url)
            host = parsed.hostname
            port = parsed.port or 5432
            user = unquote(parsed.username) if parsed.username else None
            password = unquote(parsed.password) if parsed.password else None
            database = parsed.path.lstrip('/') if parsed.path else None

            logger.debug(f"Connecting to PostgreSQL: host={host}, port={port}, user={user}, db={database}")

            self._pool = await asyncpg.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._enabled = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from database."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from database")

    async def initialize_schema(self) -> bool:
        """
        Create database tables if they don't exist.

        Returns:
            True if successful, False otherwise.
        """
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            logger.info("Database schema initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False

    # ==================== Session Operations ====================

    async def create_session(self, record: SessionRecord) -> bool:
        """Create a new session record."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO sessions (session_id, player_id, table_id, config_preset, hero_seat, is_active)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    record.session_id,
                    record.player_id,
                    record.table_id,
                    record.config_preset,
                    record.hero_seat,
                    record.is_active,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    async def end_session(self, session_id: str) -> bool:
        """Mark session as ended."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE sessions SET ended_at = NOW(), is_active = FALSE
                    WHERE session_id = $1
                    """,
                    session_id,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False

    # ==================== Hand Operations ====================

    async def create_hand(self, record: HandRecord) -> bool:
        """Create a new hand record."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO hands (session_id, hand_number, bb, sb)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (session_id, hand_number) DO NOTHING
                    """,
                    record.session_id,
                    record.hand_number,
                    record.bb,
                    record.sb,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to create hand: {e}")
            return False

    async def end_hand(
        self,
        session_id: str,
        hand_number: int,
        board_cards: str,
        final_pot: int,
        winners: List[int],
        payouts: List[int],
    ) -> bool:
        """Update hand with final results."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE hands
                    SET ended_at = NOW(), board_cards = $3, final_pot = $4,
                        winners = $5, payouts = $6
                    WHERE session_id = $1 AND hand_number = $2
                    """,
                    session_id,
                    hand_number,
                    board_cards,
                    final_pot,
                    winners,
                    payouts,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to end hand: {e}")
            return False

    # ==================== Action Operations ====================

    async def record_action(self, record: ActionRecord) -> bool:
        """Record a player action."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO actions (
                        session_id, hand_number, action_index, street,
                        actor_seat, actor_name, action_type, amount,
                        pot_before, pot_after
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (session_id, hand_number, action_index) DO NOTHING
                    """,
                    record.session_id,
                    record.hand_number,
                    record.action_index,
                    record.street,
                    record.actor_seat,
                    record.actor_name,
                    record.action_type,
                    record.amount,
                    record.pot_before,
                    record.pot_after,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to record action: {e}")
            return False

    # ==================== Prediction Operations ====================

    async def save_prediction(self, record: PredictionRecord) -> bool:
        """Save an ML prediction."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO predictions (
                        session_id, hand_number, street, action_index,
                        prediction_type, target_seat, hero_seat,
                        features, prediction, model_endpoint, latency_ms
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    record.session_id,
                    record.hand_number,
                    record.street,
                    record.action_index,
                    record.prediction_type,
                    record.target_seat,
                    record.hero_seat,
                    json.dumps(record.features),
                    json.dumps(record.prediction),
                    record.model_endpoint,
                    record.latency_ms,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False

    async def get_session_predictions(
        self,
        session_id: str,
        hand_number: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get predictions for a session."""
        if not self._enabled or not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                if hand_number is not None:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM predictions
                        WHERE session_id = $1 AND hand_number = $2
                        ORDER BY timestamp
                        """,
                        session_id,
                        hand_number,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM predictions
                        WHERE session_id = $1
                        ORDER BY hand_number, timestamp
                        """,
                        session_id,
                    )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            return []

    # ==================== Player Stats Operations ====================

    async def upsert_player_stats(self, record: PlayerStatsRecord) -> bool:
        """Insert or update player session stats."""
        if not self._enabled or not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO player_session_stats (
                        session_id, player_name, seat_index, hands_played,
                        voluntary_preflop, preflop_raise, aggressive_actions,
                        passive_actions, saw_flop, saw_turn, saw_river,
                        showdown_count, last_updated
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                    ON CONFLICT (session_id, player_name)
                    DO UPDATE SET
                        hands_played = EXCLUDED.hands_played,
                        voluntary_preflop = EXCLUDED.voluntary_preflop,
                        preflop_raise = EXCLUDED.preflop_raise,
                        aggressive_actions = EXCLUDED.aggressive_actions,
                        passive_actions = EXCLUDED.passive_actions,
                        saw_flop = EXCLUDED.saw_flop,
                        saw_turn = EXCLUDED.saw_turn,
                        saw_river = EXCLUDED.saw_river,
                        showdown_count = EXCLUDED.showdown_count,
                        last_updated = NOW()
                    """,
                    record.session_id,
                    record.player_name,
                    record.seat_index,
                    record.hands_played,
                    record.voluntary_preflop,
                    record.preflop_raise,
                    record.aggressive_actions,
                    record.passive_actions,
                    record.saw_flop,
                    record.saw_turn,
                    record.saw_river,
                    record.showdown_count,
                )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert player stats: {e}")
            return False


# Global database instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get the global database instance, using Secret Manager for credentials."""
    global _database
    if _database is None:
        # Try to get database URL from Secret Manager
        try:
            from ..secrets import get_secrets
            secrets = get_secrets()
            database_url = secrets.get_database_url()
            _database = Database(database_url=database_url)
        except ImportError:
            # Fall back to env var if secrets module not available
            _database = Database()
    return _database
