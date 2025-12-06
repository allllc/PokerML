"""
FastAPI service exposing the Aurora Poker Engine with ML predictions.
(v3 - with proper opponent action type detection)
"""

from __future__ import annotations

import sys
import uuid
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

ENGINE_PATH = Path(__file__).resolve().parents[2] / "engine"
if str(ENGINE_PATH) not in sys.path:
    sys.path.append(str(ENGINE_PATH))

from aurora_poker_engine import get_preset_config  # noqa: E402

from .models import (
    ActionRequest,
    CreateTableRequest,
    CreateTableResponse,
    PublicGameState,
    SettingsRequest,
    StartHandRequest,
)
from .session import TableManager, build_names_and_policies, observation_to_public_state

# ML imports
from .ml import (
    EndpointRegistry,
    GameSessionManager,
    MLService,
)
from .ml.feature_engine import get_position_from_button
from .db import get_database


# ==================== ML Service Initialization ====================

# Global instances
ml_registry = EndpointRegistry()
game_sessions = GameSessionManager()
ml_service: Optional[MLService] = None
database = get_database()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    global ml_service

    # Startup
    logger.info("Starting Aurora Poker API with ML support...")

    # Initialize ML service
    ml_service = MLService(ml_registry, game_sessions)
    logger.info("ML Service initialized")

    # Connect to database if configured
    if database.is_enabled:
        connected = await database.connect()
        if connected:
            await database.initialize_schema()
            logger.info("Database connected and schema initialized")

    yield

    # Shutdown
    logger.info("Shutting down...")
    if ml_service:
        await ml_service.close()
    if database.is_enabled:
        await database.disconnect()


app = FastAPI(
    title="Aurora Poker API",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5177",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5177",
        "https://aurora-poker-frontend-t2jadyhkrq-uc.a.run.app",
        "https://aurora-poker-frontend-745748443990.us-central1.run.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

tables = TableManager()


def hero_should_act(obs, hero_seat: int) -> bool:
    hero = next((p for p in obs.players if p.seatIndex == hero_seat), None)
    if hero is None:
        return False
    return (
        not hero.isFolded
        and not hero.isSittingOut
        and not hero.isAllIn
        and obs.currentPlayer == hero_seat
    )


def hero_active(obs, hero_seat: int) -> bool:
    hero = next((p for p in obs.players if p.seatIndex == hero_seat), None)
    if hero is None:
        return False
    return not hero.isFolded and not hero.isSittingOut and not hero.isAllIn


def auto_play_bots(
    session,
    obs,
    hero_seat,
    pause_on_street_change: bool = False,
    max_actions: int | None = None,
    game_session=None,
) -> PublicGameState:
    street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
    actions_taken = 0
    logger.debug(f"\n>>> AUTO_PLAY_BOTS: Starting (pause_on_street={pause_on_street_change}, max_actions={max_actions})")
    logger.debug(f"    Hero seat: {hero_seat}, Current player: {obs.currentPlayer}, Terminal: {obs.isTerminal}")

    while not obs.isTerminal:
        # Check if hero should act (hero is active and it's their turn)
        if hero_should_act(obs, hero_seat):
            logger.debug(f"    Hero should act at seat {hero_seat}. Stopping auto-play.")
            break

        # Get bot policy for current player
        policy = session.bot_policies.get(obs.currentPlayer)
        if policy is None:
            # Current player is hero but hero is not active (folded/all-in/sitting out)
            # This indicates the engine's state may be inconsistent
            # Call step(0) to auto-advance past this invalid state
            if obs.currentPlayer == hero_seat:
                logger.warning(f"    No policy for seat {obs.currentPlayer} (hero seat, but hero inactive). Calling step(0)...")
                obs = session.engine.step(0).observation
                actions_taken += 1
                if max_actions is not None and actions_taken >= max_actions:
                    logger.debug(f"    Max actions ({max_actions}) reached. Stopping.")
                    break
                continue
            # If it's not the hero and there's no policy, something is wrong - break
            logger.error(f"    No policy found for seat {obs.currentPlayer}! Breaking auto-play.")
            break

        # Bot makes decision
        logger.debug(f"    Bot at seat {obs.currentPlayer} is deciding...")
        bet, debug = policy.choose_bet(obs, obs.currentPlayer)
        logger.debug(f"    Bot chose bet={bet}")
        session.agent_debug[obs.currentPlayer] = debug
        session.agent_history.append(
            {
                "handId": session.engine.hand_id,
                "seatIndex": obs.currentPlayer,
                "streetIndex": obs.streetIndex,
                "streetName": street_names[min(obs.streetIndex, len(street_names) - 1)],
                "debug": debug,
            }
        )
        previous_street = obs.streetIndex

        # Record bot action in game session for stats
        if game_session:
            bot_name = session.player_names[obs.currentPlayer]
            street = street_names[min(obs.streetIndex, len(street_names) - 1)].lower()
            action_type = "fold" if bet == -1 else ("call_or_check" if bet <= (obs.callAmount or 0) else "bet_or_raise_to")
            game_session.record_action(
                player_name=bot_name,
                action_type=action_type,
                amount=bet,
                street=street,
            )

        obs = session.engine.step(bet).observation
        actions_taken += 1

        # Pause if street changed and hero is still active
        if (
            pause_on_street_change
            and hero_active(obs, hero_seat)
            and obs.streetIndex != previous_street
            and not obs.isTerminal
        ):
            logger.debug(f"    Street changed ({previous_street} -> {obs.streetIndex}) and hero is active. Pausing.")
            break
        if max_actions is not None and actions_taken >= max_actions:
            logger.debug(f"    Max actions ({max_actions}) reached. Stopping.")
            break

    logger.debug(f">>> AUTO_PLAY_BOTS: Completed. Actions taken: {actions_taken}, Terminal: {obs.isTerminal}\n")
    return observation_to_public_state(session, obs, session.show_all_cards)


@app.post("/api/tables", response_model=CreateTableResponse, status_code=201)
def create_table(payload: CreateTableRequest) -> CreateTableResponse:
    try:
        config = get_preset_config(payload.configPreset)
        if payload.heroSeat < 0 or payload.heroSeat >= config.numPlayers:
            raise HTTPException(status_code=400, detail="heroSeat outside table bounds")
        table_id = f"tbl_{uuid.uuid4().hex[:8]}"
        names, bot_policies, player_styles = build_names_and_policies(
            config,
            payload.heroSeat,
            payload.heroName,
            payload.botNames,
        )
        session = tables.create_table(
            table_id,
            config,
            payload.heroSeat,
            names,
            bot_policies,
            player_styles,
        )
        session.agent_history.clear()
        obs = session.engine.reset()
        state = auto_play_bots(session, obs, payload.heroSeat, pause_on_street_change=True)
        return CreateTableResponse(tableId=table_id, state=state)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tables/{table_id}/hand", response_model=PublicGameState)
def start_hand(table_id: str, payload: StartHandRequest) -> PublicGameState:
    session = tables.get(table_id)
    session.agent_history.clear()
    obs = session.engine.reset()

    # Initialize/update game session for ML stats tracking (use tableId as session_id)
    game_session = game_sessions.get_or_create_session(
        session_id=table_id,
        table_id=table_id,
        hero_seat=session.hero_seat,
    )
    # Start the hand - player stats are created automatically when actions are recorded
    game_session.start_hand(session.engine.hand_id)

    if payload.autoAdvance:
        return auto_play_bots(session, obs, session.hero_seat, pause_on_street_change=True, game_session=game_session)
    return observation_to_public_state(session, obs, session.show_all_cards)


@app.get("/api/tables/{table_id}/state", response_model=PublicGameState)
def get_state(table_id: str) -> PublicGameState:
    session = tables.get(table_id)
    obs = session.engine.observe()
    return observation_to_public_state(session, obs, session.show_all_cards)


@app.post("/api/tables/{table_id}/action", response_model=PublicGameState)
def act(table_id: str, payload: ActionRequest) -> PublicGameState:
    session = tables.get(table_id)

    # Get game session for stats tracking
    game_session = game_sessions.get_session(table_id)

    # Record hero action before executing
    pre_obs = session.engine.observe()
    hero_name = session.player_names[session.hero_seat]
    street_names = ["preflop", "flop", "turn", "river"]
    street = street_names[min(pre_obs.streetIndex, 3)]

    obs = session.engine.step(payload.betAmount).observation

    # Record the hero action in game session
    if game_session:
        action_type = "fold" if payload.betAmount == -1 else ("call_or_check" if payload.betAmount <= (pre_obs.callAmount or 0) else "bet_or_raise_to")
        game_session.record_action(
            player_name=hero_name,
            action_type=action_type,
            amount=payload.betAmount,
            street=street,
        )

    if payload.autoAdvance:
        return auto_play_bots(session, obs, session.hero_seat, pause_on_street_change=True, game_session=game_session)
    return observation_to_public_state(session, obs, session.show_all_cards)


@app.post("/api/tables/{table_id}/advance", response_model=PublicGameState)
def advance(table_id: str) -> PublicGameState:
    session = tables.get(table_id)
    obs = session.engine.observe()
    return auto_play_bots(session, obs, session.hero_seat, max_actions=1)


@app.post("/api/tables/{table_id}/settings", response_model=PublicGameState)
def update_settings(table_id: str, payload: SettingsRequest) -> PublicGameState:
    session = tables.get(table_id)
    session.show_all_cards = payload.showAllCards
    obs = session.engine.observe()
    return observation_to_public_state(session, obs, session.show_all_cards)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tables/{table_id}/debug")
def debug_state(table_id: str) -> dict:
    """Debug endpoint to see raw engine state"""
    session = tables.get(table_id)
    obs = session.engine.observe()

    current_player = session.engine.players[obs.currentPlayer]

    return {
        "hand_id": session.engine.hand_id,
        "street_index": obs.streetIndex,
        "current_player": obs.currentPlayer,
        "current_player_state": {
            "betThisStreet": current_player.betThisStreet,
            "totalCommitted": current_player.totalCommitted,
            "stack": current_player.stack,
            "isFolded": current_player.isFolded,
            "isAllIn": current_player.isAllIn,
        },
        "context": {
            "max_bet": session.engine.context.max_bet,
            "last_aggressor": session.engine.context.last_aggressor,
            "is_terminal": session.engine.context.is_terminal,
        },
        "call_amount": obs.callAmount,
        "legal_actions": [{"type": a.type, "amount": getattr(a, 'amount', None), "min": getattr(a, 'min', None), "max": getattr(a, 'max', None)} for a in obs.legalActions],
        "all_players_bet_this_street": [p.betThisStreet for p in session.engine.players],
    }


# ==================== ML Prediction Endpoints ====================

class EndpointUpdateRequest(BaseModel):
    """Request to update an ML endpoint."""
    url: Optional[str] = None
    enabled: Optional[bool] = None
    model_version: Optional[str] = None
    timeout_ms: Optional[int] = None


class AuthUpdateRequest(BaseModel):
    """Request to update Databricks auth."""
    workspace_url: Optional[str] = None
    token: Optional[str] = None


@app.get("/api/admin/ml/endpoints")
def list_ml_endpoints():
    """List all ML endpoint configurations."""
    return {
        "endpoints": ml_registry.list_endpoints(),
        "status": ml_registry.get_status_summary(),
    }


@app.put("/api/admin/ml/endpoints/{key}")
def update_ml_endpoint(key: str, payload: EndpointUpdateRequest):
    """Update an ML endpoint configuration."""
    try:
        ml_registry.update_endpoint(
            key,
            url=payload.url,
            enabled=payload.enabled,
            model_version=payload.model_version,
            timeout_ms=payload.timeout_ms,
        )
        return {"status": "updated", "key": key}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.put("/api/admin/ml/auth")
def update_ml_auth(payload: AuthUpdateRequest):
    """Update Databricks authentication credentials."""
    ml_registry.update_auth(
        workspace_url=payload.workspace_url,
        token=payload.token,
    )
    return {
        "status": "updated",
        "auth_configured": ml_registry.auth.is_configured(),
    }


@app.get("/api/admin/ml/status")
def ml_status():
    """Get ML service status."""
    return {
        "ml_service_ready": ml_service is not None,
        "database_connected": database._pool is not None,
        "endpoints": ml_registry.get_status_summary(),
        "active_sessions": len(game_sessions.list_sessions()),
    }


@app.post("/api/admin/ml/test")
async def test_ml_endpoints():
    """
    Test connectivity to Databricks ML endpoints.

    Sends a sample request to one endpoint of each model type (opponent, profit, policy)
    and returns the results.
    """
    if ml_service is None:
        raise HTTPException(status_code=503, detail="ML service not initialized")

    if not ml_registry.auth.is_configured():
        return {
            "status": "error",
            "error": "Databricks authentication not configured. Use PUT /api/admin/ml/auth to set credentials.",
            "auth_configured": False,
        }

    # V2 Opponent Model Schema - 25 base features (28 for postflop with board texture)
    sample_opponent_features = {
        "position_from_button": 2.0,
        "num_players": 6.0,
        "pot_size": 3.0,
        "amount": 2.0,
        "action_no_in_hand": 3.0,
        "raises_so_far": 1.0,
        "calls_so_far": 1.0,
        "starting_stack": 100.0,
        "stack_vs_table_median": 1.0,
        "vpip_last3_hist": 0.5,
        "vpip_last5_hist": 0.5,
        "vpip_last10_hist": 0.5,
        "pfr_last3_hist": 0.3,
        "pfr_last5_hist": 0.3,
        "pfr_last10_hist": 0.3,
        "agg_factor_last3_hist": 1.0,
        "agg_factor_last5_hist": 1.0,
        "agg_factor_last10_hist": 1.0,
        "street_adv_last3_hist": 0.5,
        "street_adv_last5_hist": 0.5,
        "street_adv_last10_hist": 0.5,
        "stack_trend_last3_hist": 0.0,
        "stack_trend_last5_hist": 0.0,
        "stack_trend_last10_hist": 0.0,
        "bet_pct_pot": 0.67,
        # Board texture (postflop only)
        "board_pair_or_better": 0.0,
        "board_flush_possible": 0.0,
        "board_straight_possible": 1.0,
    }

    # V2 Profit Model Schema - 65 features
    sample_profit_features = {
        # Core features
        "pot_size": 10.0,
        "hand_rank": 5.0,
        "starting_stack": 100.0,
        "stack_vs_table_median": 1.0,
        "position_from_button": 2.0,
        "num_players": 6.0,
        "hand_equity": 0.6,
        # Board features
        "board_pair_or_better": 0.0,
        "board_flush_possible": 0.0,
        "board_straight_possible": 1.0,
        # Draw flags
        "hole_pair_flag": 0.0,
        "has_flush_draw_flag": 0.0,
        "has_straight_draw_flag": 1.0,
        # Betting context
        "bet_pct_pot": 0.5,
        "action_no_in_hand": 3.0,
        "raises_so_far": 1.0,
        "calls_so_far": 1.0,
        # Historical stats
        "vpip_last3_hist": 0.5,
        "pfr_last3_hist": 0.3,
        "street_adv_last3_hist": 0.5,
        "agg_factor_last3_hist": 1.0,
        "stack_trend_last3_hist": 0.0,
        "vpip_last5_hist": 0.5,
        "pfr_last5_hist": 0.3,
        "street_adv_last5_hist": 0.5,
        "agg_factor_last5_hist": 1.0,
        "stack_trend_last5_hist": 0.0,
        "vpip_last10_hist": 0.5,
        "pfr_last10_hist": 0.3,
        "street_adv_last10_hist": 0.5,
        "agg_factor_last10_hist": 1.0,
        "stack_trend_last10_hist": 0.0,
        # Player counts
        "players_at_table": 6.0,
        "players_active": 3.0,
        "opponents_active": 2.0,
        # Opponent stack info
        "avg_opponent_stack": 100.0,
        "max_opponent_stack": 120.0,
        "min_opponent_stack": 80.0,
        # Position one-hot encoding
        "pos_count_BB": 0.0,
        "pos_count_BTN": 0.0,
        "pos_count_CO": 1.0,
        "pos_count_HJ": 0.0,
        "pos_count_MP": 0.0,
        "pos_count_SB": 0.0,
        "pos_count_UTG": 0.0,
        "pos_count_UTG+1": 0.0,
        # Board texture flags
        "board_monotone": 0.0,
        "board_paired": 0.0,
        "board_straighty": 1.0,
        "board_flush_pressure": 0.0,
        "board_straight_pressure": 0.0,
        # Opponent predictions
        "predicted_strength": 0.5,
        "opponent_strength_mean": 0.5,
        "opponent_strength_max": 0.7,
        "opponent_strength_min": 0.3,
        "opponent_air_count": 1.0,
        "opponent_middle_count": 1.0,
        "opponent_nutted_count": 0.0,
        "has_opponent_predictions": 1.0,
        # Pot/betting context
        "pot_before_action": 10.0,
        "facing_call": 4.0,
        "pot_odds_call": 0.29,
        "bb": 1.0,
        "is_winner": 0.0,
    }

    # V2 Policy Model Schema - 43 features
    sample_policy_features = {
        # Core features
        "starting_stack": 100.0,
        "stack_vs_table_median": 1.0,
        "position_from_button": 2.0,
        "num_players": 6.0,
        "hand_equity": 0.6,
        # Board features
        "board_pair_or_better": 0.0,
        "board_flush_possible": 0.0,
        "board_straight_possible": 1.0,
        # Draw flags
        "hole_pair_flag": 0.0,
        "has_flush_draw_flag": 0.0,
        "has_straight_draw_flag": 1.0,
        # Historical stats
        "vpip_last3_hist": 0.5,
        "pfr_last3_hist": 0.3,
        "agg_factor_last3_hist": 1.0,
        "vpip_last5_hist": 0.5,
        "pfr_last5_hist": 0.3,
        "agg_factor_last5_hist": 1.0,
        "vpip_last10_hist": 0.5,
        "pfr_last10_hist": 0.3,
        "agg_factor_last10_hist": 1.0,
        # Opponent predictions
        "predicted_strength": 0.5,
        "opponent_strength_mean": 0.5,
        "opponent_strength_max": 0.7,
        "opponent_strength_min": 0.3,
        "opponent_air_count": 1.0,
        "opponent_middle_count": 1.0,
        "opponent_nutted_count": 0.0,
        # Strength comparison
        "strength_advantage": 0.1,
        "strength_disadvantage": 0.0,
        "strength_strong": 0.0,
        # Multiway indicators
        "heads_up": 0.0,
        "three_way": 1.0,
        "multiway": 0.0,
        # Position bucket one-hot encoding
        "position_bucket_CO": 1.0,
        "position_bucket_UTG+1": 0.0,
        "position_bucket_BTN": 0.0,
        "position_bucket_UTG": 0.0,
        "position_bucket_BB": 0.0,
        "position_bucket_MP": 0.0,
        "position_bucket_SB": 0.0,
        "position_bucket_HJ": 0.0,
        # Predicted bucket one-hot encoding
        "predicted_bucket_middle": 1.0,
        "predicted_bucket_air": 0.0,
        "predicted_bucket_nutted": 0.0,
    }

    results = {}

    # Test opponent model (flop)
    try:
        opponent_result = await ml_service.model_client.predict_opponent_strength("flop", sample_opponent_features)
        results["opponent_flop"] = {
            "status": "success" if not opponent_result.from_fallback else "fallback",
            "prediction": opponent_result.prediction,
            "latency_ms": opponent_result.latency_ms,
            "error": opponent_result.error,
        }
    except Exception as e:
        results["opponent_flop"] = {"status": "error", "error": str(e)}

    # Test profit model (flop)
    try:
        profit_result = await ml_service.model_client.predict_profit("flop", sample_profit_features)
        results["profit_flop"] = {
            "status": "success" if not profit_result.from_fallback else "fallback",
            "prediction": profit_result.prediction,
            "latency_ms": profit_result.latency_ms,
            "error": profit_result.error,
        }
    except Exception as e:
        results["profit_flop"] = {"status": "error", "error": str(e)}

    # Test policy model (flop)
    try:
        policy_result = await ml_service.model_client.predict_policy("flop", sample_policy_features)
        results["policy_flop"] = {
            "status": "success" if not policy_result.from_fallback else "fallback",
            "prediction": policy_result.prediction,
            "latency_ms": policy_result.latency_ms,
            "error": policy_result.error,
        }
    except Exception as e:
        results["policy_flop"] = {"status": "error", "error": str(e)}

    # Overall status
    all_success = all(r.get("status") == "success" for r in results.values())

    return {
        "status": "success" if all_success else "partial",
        "auth_configured": True,
        "results": results,
    }


@app.get("/api/admin/ml/test-raw")
async def test_ml_raw():
    """
    Test ML endpoints with the correct 56-column schema for profit model.
    """
    import httpx

    # Correct 56-column schema for profit model (flop)
    profit_features = {
        # Action context
        "street_rank": 1,  # flop = 1
        "is_player_action": True,
        "is_raise": False,
        "is_call": True,
        "is_fold": False,
        "is_voluntary": True,

        # Pot and stack
        "pot_contribution": 30.0,
        "pot_size": 100.0,
        "seat_no": 3,
        "starting_stack": 1000.0,
        "players_at_table": 6,
        "position_index": 3,

        # Position one-hot
        "position_bucket_blinds": 0,
        "position_bucket_button": 0,
        "position_bucket_early": 0,
        "position_bucket_late": 1,
        "position_bucket_middle": 0,
        "position_bucket_small_blind": 0,

        # Historical stats
        "vpip_last3_hist": 0.33,
        "pfr_last3_hist": 0.20,
        "stickiness_last3_hist": 0.67,
        "agg_factor_last3_hist": 1.5,
        "vpip_last5_hist": 0.40,
        "pfr_last5_hist": 0.25,
        "stickiness_last5_hist": 0.60,
        "agg_factor_last5_hist": 1.3,
        "vpip_last10_hist": 0.35,
        "pfr_last10_hist": 0.22,
        "stickiness_last10_hist": 0.65,
        "agg_factor_last10_hist": 1.4,

        # Player counts
        "players_active": 3,
        "opponents_active": 2,
        "avg_opponent_stack": 950.0,
        "max_opponent_stack": 1200.0,
        "min_opponent_stack": 700.0,

        # Opponent positions
        "opponents_early": 0,
        "opponents_middle": 1,
        "opponents_late": 1,
        "opponents_blinds": 0,

        # Betting
        "bb": 2.0,
        "facing_call": 20.0,
        "pot_before_action": 80.0,
        "pot_odds_call": 0.20,

        # Board texture
        "board_monotone": 0,
        "board_paired": 0,
        "board_straighty": 0,
        "board_flush_pressure": 1,
        "board_straight_pressure": 0,

        # Opponent strength
        "opponent_strength_mean": 0.5,
        "opponent_strength_max": 0.7,
        "opponent_strength_min": 0.3,
        "opponent_air_count": 1,
        "opponent_middle_count": 1,
        "opponent_nutted_count": 0,
        "has_opponent_predictions": 1,

        # Hand strength
        "hand_strength": 0.65,
    }

    endpoint_url = "https://3866123326870389.9.gcp.databricks.com/serving-endpoints/02-profit-modeling-flop/invocations"
    columns = list(profit_features.keys())
    data = [[profit_features[col] for col in columns]]
    payload = {"dataframe_split": {"columns": columns, "data": data}}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            endpoint_url,
            json=payload,
            headers=ml_registry.auth.get_headers(),
        )

        profit_result = {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text[:500],
            "n_columns": len(columns),
        }

        # Now test policy model with 85 columns
        policy_features = {
            **profit_features,
            # Additional policy features
            "amount": 20.0,
            "spr": 10.0,
            "spr_low": 0,
            "spr_medium": 0,
            "spr_high": 1,
            "spr_very_deep": 0,
            "can_check": 0,
            "pot_committed": 0.02,
            "pot_committed_heavy": 0,
            "raise_33_size": 53.0,
            "raise_75_size": 95.0,
            "raise_100_size": 120.0,
            "can_afford_raise_33": 1,
            "can_afford_raise_75": 1,
            "can_afford_raise_100": 1,
            "raise_33_pct_stack": 0.053,
            "raise_75_pct_stack": 0.095,
            "remaining_stack_after_call": 980.0,
            "remaining_stack_bb": 490.0,
            "remaining_stack_pct": 0.98,
            "strength_advantage": 0.15,
            "strength_disadvantage": 0,
            "strength_strong": 0,
            "position_late": 1,
            "position_allows_bluff": 1,
            "heads_up": 0,
            "three_way": 1,
            "multiway": 0,
            "facing_nutted_opponent": 0,
            "facing_multiple_strong": 0,
            "pot_odds_favorable": 1,
            "pot_odds_unfavorable": 0,
            "postflop_low_spr": 0,
            "preflop_deep_stack": 0,
        }

        policy_url = "https://3866123326870389.9.gcp.databricks.com/serving-endpoints/03-policy-training-flop/invocations"
        policy_columns = list(policy_features.keys())
        policy_data = [[policy_features[col] for col in policy_columns]]
        policy_payload = {"dataframe_split": {"columns": policy_columns, "data": policy_data}}

        policy_response = await client.post(
            policy_url,
            json=policy_payload,
            headers=ml_registry.auth.get_headers(),
        )

        policy_result = {
            "status_code": policy_response.status_code,
            "response": policy_response.json() if policy_response.status_code == 200 else policy_response.text[:500],
            "n_columns": len(policy_columns),
        }

        return {
            "profit_model": profit_result,
            "policy_model": policy_result,
        }


@app.get("/api/admin/ml/schemas")
async def get_ml_schemas():
    """
    Fetch model input schemas from Databricks Unity Catalog.

    Queries the Databricks API to get the exact input schema
    that each model expects.
    """
    import httpx
    import json as json_lib

    if not ml_registry.auth.is_configured():
        return {
            "status": "error",
            "error": "Databricks authentication not configured",
        }

    workspace_url = "https://3866123326870389.9.gcp.databricks.com"
    endpoints = [
        "02-profit-modeling-flop",
        "03-policy-training-flop",
    ]

    schemas = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for endpoint_name in endpoints:
            try:
                # Get serving endpoint details
                url = f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}"
                response = await client.get(
                    url,
                    headers=ml_registry.auth.get_headers(),
                )

                if response.status_code == 200:
                    data = response.json()
                    config = data.get("config", {})
                    served_entities = config.get("served_entities", [])

                    if served_entities:
                        entity = served_entities[0]
                        entity_name = entity.get("entity_name")  # e.g., "pokerml.default.02-profit-modeling-flop"
                        entity_version = entity.get("entity_version")

                        schemas[endpoint_name] = {
                            "entity_name": entity_name,
                            "entity_version": entity_version,
                        }

                        # For Unity Catalog models, use the UC API
                        # Parse: catalog.schema.model_name
                        if entity_name and "." in entity_name:
                            parts = entity_name.split(".")
                            if len(parts) == 3:
                                catalog, schema_name, model_name = parts

                                # Get model version from UC via MLflow API
                                full_name = entity_name
                                uc_url = f"{workspace_url}/api/2.0/mlflow/unity-catalog/model-versions/get"
                                uc_response = await client.get(
                                    uc_url,
                                    headers=ml_registry.auth.get_headers(),
                                    params={"name": full_name, "version": entity_version},
                                )

                                if uc_response.status_code == 200:
                                    uc_data = uc_response.json()
                                    model_version = uc_data.get("model_version", {})
                                    run_id = model_version.get("run_id")
                                    schemas[endpoint_name]["run_id"] = run_id
                                    schemas[endpoint_name]["source"] = model_version.get("source")

                                    # Get the run to find signature
                                    if run_id:
                                        run_url = f"{workspace_url}/api/2.0/mlflow/runs/get"
                                        run_response = await client.get(
                                            run_url,
                                            headers=ml_registry.auth.get_headers(),
                                            params={"run_id": run_id},
                                        )

                                        if run_response.status_code == 200:
                                            run_data = run_response.json()
                                            run_info = run_data.get("run", {}).get("data", {})
                                            tags = {t["key"]: t["value"] for t in run_info.get("tags", [])}
                                            params = {p["key"]: p["value"] for p in run_info.get("params", [])}

                                            # The model signature is in mlflow.log-model.history tag
                                            model_history = tags.get("mlflow.log-model.history")
                                            schemas[endpoint_name]["has_history_tag"] = model_history is not None
                                            if model_history:
                                                try:
                                                    history = json_lib.loads(model_history)
                                                    if history and len(history) > 0:
                                                        signature = history[0].get("signature")
                                                        if signature:
                                                            inputs_str = signature.get("inputs")
                                                            if inputs_str:
                                                                # Parse the inputs JSON string
                                                                try:
                                                                    inputs = json_lib.loads(inputs_str)
                                                                    # Extract column names
                                                                    columns = [inp.get("name") for inp in inputs if inp.get("name")]
                                                                    schemas[endpoint_name]["input_columns"] = columns
                                                                    schemas[endpoint_name]["n_columns"] = len(columns)
                                                                except:
                                                                    schemas[endpoint_name]["inputs_raw"] = inputs_str[:2000]
                                                except Exception as parse_err:
                                                    schemas[endpoint_name]["signature_parse_error"] = str(parse_err)
                                                    schemas[endpoint_name]["signature_raw"] = model_history[:3000]

                                            # Also save all tags for debugging
                                            schemas[endpoint_name]["all_tags"] = list(tags.keys())
                                            schemas[endpoint_name]["params"] = params

                                            # Try to list artifacts to find model path
                                            list_url = f"{workspace_url}/api/2.0/mlflow/artifacts/list"
                                            list_response = await client.get(
                                                list_url,
                                                headers=ml_registry.auth.get_headers(),
                                                params={"run_id": run_id},
                                            )
                                            if list_response.status_code == 200:
                                                artifacts = list_response.json()
                                                schemas[endpoint_name]["artifacts"] = artifacts.get("files", [])[:5]

                                            # Try to get MLmodel file from DBFS
                                            artifact_root = f"dbfs:/databricks/mlflow-tracking/{run_data.get('run', {}).get('info', {}).get('experiment_id', '')}/{run_id}/artifacts"
                                            mlmodel_path = f"{artifact_root}/model/MLmodel"

                                            # Use DBFS API to read the file
                                            dbfs_url = f"{workspace_url}/api/2.0/dbfs/read"
                                            dbfs_response = await client.get(
                                                dbfs_url,
                                                headers=ml_registry.auth.get_headers(),
                                                params={"path": mlmodel_path.replace("dbfs:", "")},
                                            )
                                            if dbfs_response.status_code == 200:
                                                dbfs_data = dbfs_response.json()
                                                # Content is base64 encoded
                                                import base64
                                                content_b64 = dbfs_data.get("data", "")
                                                try:
                                                    mlmodel_content = base64.b64decode(content_b64).decode('utf-8')
                                                    schemas[endpoint_name]["mlmodel_yaml"] = mlmodel_content[:4000]
                                                except:
                                                    schemas[endpoint_name]["mlmodel_b64"] = content_b64[:500]
                                            else:
                                                schemas[endpoint_name]["dbfs_error"] = dbfs_response.text[:300]
                                        else:
                                            schemas[endpoint_name]["run_error"] = run_response.text[:300]
                                else:
                                    schemas[endpoint_name]["uc_error"] = uc_response.text[:500]
                    else:
                        schemas[endpoint_name] = {"error": "No served entities"}
                else:
                    schemas[endpoint_name] = {
                        "error": f"HTTP {response.status_code}",
                        "message": response.text[:500],
                    }
            except Exception as e:
                schemas[endpoint_name] = {"error": str(e)}

    return {"schemas": schemas}


@app.get("/api/tables/{table_id}/ml/features")
async def get_ml_features(
    table_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """
    Debug endpoint: Get computed ML features without calling models.

    Returns all three feature sets (opponent, profit, policy) computed
    from the current game state.
    """
    try:
        if ml_service is None:
            raise HTTPException(status_code=503, detail="ML service not initialized")

        try:
            table_session = tables.get(table_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

        session_id = x_session_id or "debug_session"
        game_session = game_sessions.get_or_create_session(
            session_id=session_id,
            table_id=table_id,
            hero_seat=table_session.hero_seat,
        )

        obs = table_session.engine.observe()
        street_names = ["preflop", "flop", "turn", "river"]
        street = street_names[min(obs.streetIndex, 3)]

        hero = next((p for p in obs.players if p.seatIndex == table_session.hero_seat), None)
        if hero is None:
            raise HTTPException(status_code=400, detail="Hero not found")

        all_players = [
            {
                "seatIndex": p.seatIndex,
                "name": table_session.player_names[p.seatIndex],
                "stack": p.stack,
                "betThisStreet": p.betThisStreet,
                "totalCommitted": p.totalCommitted,
                "isFolded": p.isFolded,
                "isAllIn": p.isAllIn,
                "isSittingOut": p.isSittingOut,
            }
            for p in obs.players
        ]

        hero_hole_cards = []
        if hero.holeCards:
            hero_hole_cards = [f"{c.rank}{c.suit[0].lower()}" for c in hero.holeCards]

        board_cards = []
        if obs.boardCards:
            board_cards = [f"{c.rank}{c.suit[0].lower()}" for c in obs.boardCards]

        active_players = [p for p in obs.players if not p.isFolded and not p.isSittingOut]
        active_opponents = [p for p in all_players if p["seatIndex"] != table_session.hero_seat and not p["isFolded"]]
        all_stacks = [p["stack"] for p in all_players]

        # Calculate position relative to button (0=BTN, 1=SB, 2=BB, etc.)
        position_index = get_position_from_button(
            seat_index=table_session.hero_seat,
            button_seat=obs.buttonSeat,
            num_players=len(obs.players)
        )

        bb = table_session.engine.config.blinds[1] if len(table_session.engine.config.blinds) > 1 else 2

        # Calculate features using the feature engine
        feature_engine = ml_service.feature_engine

        # Opponent features (for first active opponent as example)
        opponent_features = {}
        if active_opponents:
            opp = active_opponents[0]
            opponent_features = feature_engine.calculate_opponent_features(
                pot_size=obs.pot.mainPot,
                amount=opp["betThisStreet"],
                action_type="call_or_check",
                street=street,
                session=game_session,
                player_name=opp["name"],
                player_stack=opp["stack"],
                all_stacks=all_stacks,
                pot_contribution=opp["totalCommitted"],
                board_cards=board_cards,
                bb=bb,  # V3: BB normalization
            )

        # Profit features
        profit_features = feature_engine.calculate_profit_features(
            pot_size=obs.pot.mainPot,
            street=street,
            session=game_session,
            hero_seat=table_session.hero_seat,
            hero_name=table_session.player_names[table_session.hero_seat],
            hero_stack=hero.stack,
            hero_hole_cards=hero_hole_cards,
            all_stacks=all_stacks,
            board_cards=board_cards,
            active_opponents=[{"seat": p["seatIndex"], "name": p["name"], "stack": p["stack"]} for p in active_opponents],
            opponent_predictions=[{"predicted_class": "middle"} for _ in active_opponents],  # Placeholder
            facing_bet=obs.callAmount or 0,
            bb=bb,
            position_from_button=position_index,
            num_players=len(active_players),
        )

        # Policy features
        policy_features = feature_engine.calculate_policy_features(
            profit_features=profit_features,
            hero_stack=hero.stack,
            pot_size=obs.pot.mainPot,
            facing_bet=obs.callAmount or 0,
            street=street,
            hero_hole_cards=hero_hole_cards,
            board_cards=board_cards,
            opponent_predictions=[{"predicted_class": "middle"} for _ in active_opponents],
            position_from_button=position_index,
            num_players=len(active_players),
        )

        return {
            "gameState": {
                "street": street,
                "pot": obs.pot.mainPot,
                "heroSeat": table_session.hero_seat,
                "heroStack": hero.stack,
                "heroHoleCards": hero_hole_cards,
                "boardCards": board_cards,
                "facingBet": obs.callAmount or 0,
                "bb": bb,
                "activeOpponents": len(active_opponents),
                "positionIndex": position_index,
            },
            "opponentFeatures": opponent_features,
            "opponentFeatureCount": len(opponent_features),
            "profitFeatures": profit_features,
            "profitFeatureCount": len(profit_features),
            "policyFeatures": policy_features,
            "policyFeatureCount": len(policy_features),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error calculating ML features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tables/{table_id}/ml/predict")
async def get_ml_predictions(
    table_id: str,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """
    Generate ML predictions for current game state.

    Called automatically on player actions or explicitly via this endpoint.
    Requires X-Session-ID header for session tracking.
    """
    if ml_service is None:
        raise HTTPException(status_code=503, detail="ML service not initialized")

    # Get table session
    try:
        table_session = tables.get(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

    # Get or create game session
    session_id = x_session_id or str(uuid.uuid4())
    game_session = game_sessions.get_or_create_session(
        session_id=session_id,
        table_id=table_id,
        hero_seat=table_session.hero_seat,
    )

    # Get current game state
    obs = table_session.engine.observe()
    street_names = ["preflop", "flop", "turn", "river"]
    street = street_names[min(obs.streetIndex, 3)]

    # Get hero info
    hero = next((p for p in obs.players if p.seatIndex == table_session.hero_seat), None)
    if hero is None:
        raise HTTPException(status_code=400, detail="Hero not found")

    # Build player info
    all_players = [
        {
            "seatIndex": p.seatIndex,
            "name": table_session.player_names[p.seatIndex],
            "stack": p.stack,
            "betThisStreet": p.betThisStreet,
            "totalCommitted": p.totalCommitted,
            "isFolded": p.isFolded,
            "isAllIn": p.isAllIn,
            "isSittingOut": p.isSittingOut,
        }
        for p in obs.players
    ]

    # Get hero hole cards
    hero_hole_cards = []
    if hero.holeCards:
        hero_hole_cards = [f"{c.rank}{c.suit[0].lower()}" for c in hero.holeCards]

    # Get board cards
    board_cards = []
    if obs.boardCards:
        board_cards = [f"{c.rank}{c.suit[0].lower()}" for c in obs.boardCards]

    # Calculate hero position relative to button (0=BTN, 1=SB, 2=BB, etc.)
    position_index = get_position_from_button(
        seat_index=table_session.hero_seat,
        button_seat=obs.buttonSeat,
        num_players=len(obs.players)
    )
    logger.info(
        f"POSITION CALC: hero_seat={table_session.hero_seat}, button_seat={obs.buttonSeat}, "
        f"num_players={len(obs.players)}, position_index={position_index}"
    )

    # Get BB from config
    bb = table_session.engine.config.blinds[1] if len(table_session.engine.config.blinds) > 1 else 2

    # Generate predictions
    result = await ml_service.generate_predictions(
        session_id=session_id,
        hand_id=table_session.engine.hand_id,
        street=street,
        hero_seat=table_session.hero_seat,
        hero_name=table_session.player_names[table_session.hero_seat],
        hero_stack=hero.stack,
        hero_hole_cards=hero_hole_cards,
        pot_size=obs.pot.mainPot,
        facing_bet=obs.callAmount or 0,
        bb=bb,
        all_players=all_players,
        board_cards=board_cards,
        position_index=position_index,
    )

    return result.to_dict()


@app.get("/api/tables/{table_id}/ml/history")
async def get_prediction_history(
    table_id: str,
    hand_number: Optional[int] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    """Get prediction history for current session."""
    if not x_session_id:
        return {"predictions": [], "message": "No session ID provided"}

    game_session = game_sessions.get_session(x_session_id)
    if not game_session:
        return {"predictions": [], "message": "Session not found"}

    if hand_number is not None:
        predictions = game_session.get_predictions_for_hand(hand_number)
    else:
        predictions = game_session.predictions

    return {
        "session_id": x_session_id,
        "predictions": [
            {
                "handNumber": p.hand_number,
                "street": p.street,
                "actionIndex": p.action_index,
                "predictionType": p.prediction_type,
                "targetSeat": p.target_seat,
                "prediction": p.prediction,
                "latencyMs": p.latency_ms,
                "fromFallback": p.from_fallback,
                "timestamp": p.timestamp.isoformat(),
            }
            for p in predictions
        ],
    }


@app.delete("/api/sessions/{session_id}")
async def end_session(session_id: str):
    """End a game session and clear its data."""
    session = game_sessions.end_session(session_id)
    if session:
        return {"status": "ended", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/api/tables/{table_id}/ml/stats")
async def get_ml_stats(table_id: str):
    """
    Get player statistics and ML feature debug info.

    Returns accumulated stats for each player in the session,
    plus the current computed features for debugging.
    """
    try:
        try:
            table_session = tables.get(table_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Table {table_id} not found")

        # Use tableId as session_id for consistent session tracking
        game_session = game_sessions.get_session(table_id)

        # Get player stats from session (including current hand in-progress)
        player_stats = {}
        if game_session:
            # Get current hand summaries for intra-hand stats
            current_hand_summaries = {}
            if game_session.current_hand:
                current_hand_summaries = game_session.current_hand.player_summaries

            for name, stats in game_session.player_stats.items():
                # Include current hand in count if player has acted this hand
                hands_in_progress = 1 if name in current_hand_summaries else 0
                current_summary = current_hand_summaries.get(name)

                # Calculate intra-hand cumulative stats
                intra_voluntary = stats.voluntary_preflop + (1 if current_summary and current_summary.voluntary_preflop else 0)
                intra_pfr = stats.preflop_raise + (1 if current_summary and current_summary.preflop_raise else 0)
                intra_agg = stats.aggressive_actions + (current_summary.aggressive_actions if current_summary else 0)
                intra_passive = stats.passive_actions + (current_summary.passive_actions if current_summary else 0)
                intra_flop = stats.saw_flop + (1 if current_summary and current_summary.saw_flop else 0)
                intra_turn = stats.saw_turn + (1 if current_summary and current_summary.saw_turn else 0)
                intra_river = stats.saw_river + (1 if current_summary and current_summary.saw_river else 0)

                player_stats[name] = {
                    "seatIndex": stats.seat_index,
                    "handsPlayed": stats.hands_played + hands_in_progress,
                    "vpip": round(stats.vpip(10) * 100, 1),
                    "pfr": round(stats.pfr(10) * 100, 1),
                    "aggressionFactor": round(stats.aggression_factor(10), 2),
                    "stickiness": round(stats.stickiness(10) * 100, 1),
                    "cumulativeStats": {
                        "voluntaryPreflop": intra_voluntary,
                        "preflopRaise": intra_pfr,
                        "aggressiveActions": intra_agg,
                        "passiveActions": intra_passive,
                        "sawFlop": intra_flop,
                        "sawTurn": intra_turn,
                        "sawRiver": intra_river,
                        "showdownCount": stats.showdown_count,
                    },
                    "recentHandsCount": len(stats.recent_hands),
                    # Include current hand action info
                    "currentHand": {
                        "voluntaryPreflop": current_summary.voluntary_preflop if current_summary else False,
                        "preflopRaise": current_summary.preflop_raise if current_summary else False,
                        "sawFlop": current_summary.saw_flop if current_summary else False,
                        "sawTurn": current_summary.saw_turn if current_summary else False,
                        "sawRiver": current_summary.saw_river if current_summary else False,
                        "aggressiveActions": current_summary.aggressive_actions if current_summary else 0,
                        "passiveActions": current_summary.passive_actions if current_summary else 0,
                    } if current_summary else None,
                }

        # Get current game state
        obs = table_session.engine.observe()
        street_names = ["preflop", "flop", "turn", "river"]
        street = street_names[min(obs.streetIndex, 3)]

        # Current hand info
        current_hand = None
        if game_session and game_session.current_hand:
            current_hand = {
                "handNumber": game_session.current_hand.hand_number,
                "actionCount": game_session.current_hand.action_count,
                "raiseCount": game_session.current_hand.raise_count,
                "callCount": game_session.current_hand.call_count,
                "currentStreet": game_session.current_hand.current_street,
            }

        # Get ALL ML features for debugging
        all_features = {}
        features_summary = None
        if ml_service:
            try:
                hero = next((p for p in obs.players if p.seatIndex == table_session.hero_seat), None)
                if hero and game_session:
                    all_stacks = [p.stack for p in obs.players]
                    active_opponents = [
                        p for p in obs.players
                        if p.seatIndex != table_session.hero_seat and not p.isFolded
                    ]

                    hero_hole_cards = []
                    if hero.holeCards:
                        hero_hole_cards = [f"{c.rank}{c.suit[0].lower()}" for c in hero.holeCards]

                    board_cards = []
                    if obs.boardCards:
                        board_cards = [f"{c.rank}{c.suit[0].lower()}" for c in obs.boardCards]

                    bb = table_session.engine.config.blinds[1] if len(table_session.engine.config.blinds) > 1 else 2

                    # Calculate policy features (extends profit features)
                    active_players = [p for p in obs.players if not p.isFolded and not p.isSittingOut]
                    # Calculate hero position relative to button (0=BTN, 1=SB, 2=BB, etc.)
                    position_index = get_position_from_button(
                        seat_index=table_session.hero_seat,
                        button_seat=obs.buttonSeat,
                        num_players=len(obs.players)
                    )

                    # Calculate profit features
                    profit_features = ml_service.feature_engine.calculate_profit_features(
                        pot_size=obs.pot.mainPot,
                        street=street,
                        session=game_session,
                        hero_seat=table_session.hero_seat,
                        hero_name=table_session.player_names[table_session.hero_seat],
                        hero_stack=hero.stack,
                        hero_hole_cards=hero_hole_cards,
                        all_stacks=all_stacks,
                        board_cards=board_cards,
                        active_opponents=[{"seat": p.seatIndex, "name": table_session.player_names[p.seatIndex], "stack": p.stack} for p in active_opponents],
                        opponent_predictions=[{"predicted_class": "middle"} for _ in active_opponents],
                        facing_bet=obs.callAmount or 0,
                        bb=bb,
                        position_from_button=position_index,
                        num_players=len(active_players),
                    )

                    policy_features = ml_service.feature_engine.calculate_policy_features(
                        profit_features=profit_features,
                        hero_stack=hero.stack,
                        pot_size=obs.pot.mainPot,
                        facing_bet=obs.callAmount or 0,
                        street=street,
                        hero_hole_cards=hero_hole_cards,
                        board_cards=board_cards,
                        opponent_predictions=[{"predicted_class": "middle"} for _ in active_opponents],
                        position_from_button=position_index,
                        num_players=len(active_players),
                    )

                    # Return ALL features
                    all_features = {k: round(v, 4) if isinstance(v, float) else v for k, v in policy_features.items()}

                    # Also provide a summary for quick display
                    features_summary = {
                        "handStrength": round(profit_features.get("hand_strength", 0) * 100, 1),
                        "potOdds": round(profit_features.get("pot_odds_call", 0) * 100, 1),
                        "opponentsActive": profit_features.get("opponents_active", 0),
                        "potSize": profit_features.get("pot_size", 0),
                        "facingBet": obs.callAmount or 0,
                    }
            except Exception as e:
                logger.debug(f"Failed to calculate features for stats: {e}")

        return {
            "tableId": table_id,
            "sessionId": table_id,  # Use tableId as sessionId for consistency
            "street": street,
            "handNumber": game_session.current_hand_number if game_session else 0,
            "currentHand": current_hand,
            "playerStats": player_stats,
            "features": features_summary,
            "allFeatures": all_features,
            "featureCount": len(all_features),
            "sessionActive": game_session is not None,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting ML stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AI Advisor Endpoint ====================

class AIAdvisorRequest(BaseModel):
    """Request model for AI poker advisor."""
    tableId: str
    riskLevel: int = 50  # 0-100, 0=conservative, 100=aggressive
    playStyle: int = 50  # 0-100, 0=by the book, 100=variable pro
    gameState: Optional[Dict[str, Any]] = None
    mlPredictions: Optional[Dict[str, Any]] = None
    rawFeatures: Optional[Dict[str, Any]] = None  # Raw ML features with explanations
    handHistory: Optional[List[Dict[str, Any]]] = None  # Betting actions this hand


@app.post("/api/ai-advisor")
async def get_ai_advice(request: AIAdvisorRequest):
    """
    Get AI poker advice based on ML predictions and player preferences.

    Uses OpenAI GPT to analyze the current game state, ML predictions, raw features,
    hand history, and board texture to provide strategic advice adjusted for
    the player's risk tolerance and play style preferences.

    The AI advisor considers predictions as guidance, not absolute rules, and
    evaluates the cards, board texture, and betting patterns independently.
    """
    from agents import POKER_ADVISOR_SYSTEM_PROMPT, build_poker_advisor_prompt
    from openai import OpenAI
    import os

    try:
        # Validate inputs
        risk_level = max(0, min(100, request.riskLevel))
        play_style = max(0, min(100, request.playStyle))

        # Build the prompt with all available context
        game_state = request.gameState or {}
        ml_predictions = request.mlPredictions or {}
        raw_features = request.rawFeatures or {}
        hand_history = request.handHistory or []

        user_prompt = build_poker_advisor_prompt(
            game_state=game_state,
            ml_predictions=ml_predictions,
            risk_level=risk_level,
            play_style=play_style,
            raw_features=raw_features,
            hand_history=hand_history,
        )

        # Check for OpenAI API key (stored as OPENAI_API_KEY or ANTHROPIC_API_KEY for backwards compat)
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            # Return enhanced fallback response using all available data
            logger.warning("OPENAI_API_KEY not set, returning enhanced fallback response")
            return {
                "advice": _generate_fallback_advice(
                    game_state,
                    ml_predictions,
                    risk_level,
                    play_style,
                    raw_features,
                    hand_history,
                ),
                "model": "fallback",
                "riskLevel": risk_level,
                "playStyle": play_style,
                "promptUsed": user_prompt,
            }

        # Call OpenAI GPT API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=256,
            messages=[
                {"role": "system", "content": POKER_ADVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        advice = response.choices[0].message.content if response.choices else "Unable to generate advice."

        return {
            "advice": advice,
            "model": "gpt-4o",
            "riskLevel": risk_level,
            "playStyle": play_style,
            "promptUsed": user_prompt,
        }

    except Exception as e:
        logger.exception(f"Error getting AI advice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_fallback_advice(
    game_state: Dict[str, Any],
    ml_predictions: Dict[str, Any],
    risk_level: int,
    play_style: int,
    raw_features: Dict[str, Any] = None,
    hand_history: List[Dict[str, Any]] = None,
) -> str:
    """
    Generate enhanced fallback advice when Claude API is not available.

    Uses all available data: cards, board texture, ML predictions, features, and history.
    """
    raw_features = raw_features or {}
    hand_history = hand_history or []

    # Extract game state
    hero_cards = game_state.get("heroCards", "unknown")
    board = game_state.get("boardCards", "")
    street = game_state.get("street", "preflop")
    pot_size = game_state.get("potSize", 0)
    facing_bet = game_state.get("facingBet", 0)
    hero_stack = game_state.get("heroStack", 0)
    bb = game_state.get("bb", 2)

    # Calculate values in BB
    pot_bb = pot_size / bb if bb else 0
    facing_bb = facing_bet / bb if bb else 0
    stack_bb = hero_stack / bb if bb else 0

    # Get ML predictions
    ml_action = "call"
    if ml_predictions.get("actionRecommendation"):
        ml_action = ml_predictions["actionRecommendation"].get("action", "call")

    profit_bb = 0.0
    if ml_predictions.get("profitPrediction"):
        profit_bb = ml_predictions["profitPrediction"].get("predictedProfitBB", 0)

    # Get opponent reads
    opp_reads = []
    nutted_count = 0
    air_count = 0
    if ml_predictions.get("opponentPredictions"):
        for opp in ml_predictions["opponentPredictions"]:
            strength = opp.get("predictedClass", "middle")
            if strength == "nutted":
                nutted_count += 1
            elif strength == "air":
                air_count += 1
            opp_reads.append(f"{opp.get('playerName', 'Opp')}: {strength}")

    # Extract key features
    hand_strength = raw_features.get("hand_strength", 0.5)
    spr = raw_features.get("spr", 10)
    pot_odds = raw_features.get("pot_odds_call", 0.25)
    strength_advantage = raw_features.get("strength_advantage", 0)

    # Board texture analysis
    board_flush = raw_features.get("board_flush_pressure", 0)
    board_straight = raw_features.get("board_straight_pressure", 0)
    board_paired = raw_features.get("board_paired", 0)

    # Decision logic based on all factors
    recommended_action = ml_action
    reasoning_parts = []

    # Preflop logic
    if street == "preflop":
        if hand_strength > 0.7:
            reasoning_parts.append(f"Your {hero_cards} is a premium hand")
            if risk_level > 50:
                recommended_action = "raise_75"
                reasoning_parts.append("raise for value")
            else:
                recommended_action = "raise_33" if facing_bet == 0 else "call"
        elif hand_strength > 0.5:
            reasoning_parts.append(f"Your {hero_cards} is playable")
            if facing_bb > 4:
                recommended_action = "fold"
                reasoning_parts.append(f"but facing {facing_bb:.1f} BB is too much")
            else:
                recommended_action = "call"
        else:
            reasoning_parts.append(f"Your {hero_cards} is marginal")
            if facing_bet > 0:
                recommended_action = "fold"
                reasoning_parts.append("fold to aggression")
            else:
                recommended_action = "call"  # check option

    # Postflop logic
    else:
        # Strong hand
        if hand_strength > 0.7:
            reasoning_parts.append(f"You have a strong hand ({hand_strength:.0%} strength)")
            if nutted_count > 0:
                reasoning_parts.append(f"but {nutted_count} opponent(s) may be strong too")
                recommended_action = "call" if risk_level < 50 else "raise_33"
            else:
                recommended_action = "raise_75" if risk_level > 40 else "raise_33"
                reasoning_parts.append("bet for value")

        # Medium hand
        elif hand_strength > 0.4:
            reasoning_parts.append(f"You have a medium-strength hand")
            if nutted_count > 0:
                recommended_action = "fold" if facing_bb > 3 else "call"
                reasoning_parts.append("be cautious with opponent showing strength")
            elif air_count > 0 and facing_bet == 0:
                recommended_action = "raise_33"
                reasoning_parts.append("opponents look weak, consider a bluff")
            else:
                recommended_action = "call" if pot_odds < 0.3 else "fold"

        # Weak hand
        else:
            reasoning_parts.append("Your hand is weak")
            if facing_bet > 0:
                recommended_action = "fold"
                reasoning_parts.append("fold to the bet")
            elif air_count >= 2 and risk_level > 60:
                recommended_action = "raise_33"
                reasoning_parts.append("but opponents look weak - consider a bluff")
            else:
                recommended_action = "call"  # check

    # Board texture warnings
    if board_flush and "flush" not in hero_cards.lower():
        reasoning_parts.append("watch for flush on board")
    if board_paired:
        reasoning_parts.append("paired board means full house possible")

    # SPR considerations
    if spr < 3:
        reasoning_parts.append(f"Low SPR ({spr:.1f}) - you're pot committed")
        if hand_strength > 0.4:
            recommended_action = "shove" if risk_level > 50 else "call"

    # Adjust for risk level
    if risk_level < 30:
        if recommended_action in ["raise_75", "shove"] and hand_strength < 0.7:
            recommended_action = "call"
            reasoning_parts.append("playing safe given conservative preference")
    elif risk_level > 70:
        if recommended_action == "call" and profit_bb > 0.5 and air_count > 0:
            recommended_action = "raise_33"
            reasoning_parts.append("taking aggressive line as you prefer")

    # Format the action text with BB sizing
    min_raise = game_state.get("minRaise", 0)
    pot_raise = pot_size + facing_bet

    if recommended_action == "fold":
        action_text = "fold"
    elif recommended_action == "call":
        if facing_bet == 0:
            action_text = "check"
        else:
            action_text = f"call {facing_bb:.1f} BB"
    elif recommended_action == "raise_33":
        raise_amount = (pot_raise * 0.33 + facing_bet) / bb
        action_text = f"raise to {raise_amount:.1f} BB (33% pot)"
    elif recommended_action == "raise_75":
        raise_amount = (pot_raise * 0.75 + facing_bet) / bb
        action_text = f"raise to {raise_amount:.1f} BB (75% pot)"
    elif recommended_action == "shove":
        action_text = f"go all-in ({stack_bb:.0f} BB)"
    else:
        action_text = recommended_action

    # Build the advice
    reasoning = ". ".join(reasoning_parts).capitalize() if reasoning_parts else ""
    profit_note = f" ML expects {profit_bb:+.1f} BB profit." if abs(profit_bb) > 0.5 else ""

    return f"I recommend you {action_text}. {reasoning}.{profit_note}".strip()
