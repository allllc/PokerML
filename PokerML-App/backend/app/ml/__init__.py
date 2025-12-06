"""
ML module for poker model integration.

This module provides:
- EndpointRegistry: Configuration management for Databricks model endpoints
- DatabricksModelClient: Async client for calling model serving endpoints
- FeatureEngine: Real-time feature calculation from game state
- MLService: High-level orchestration of ML predictions
- GameSession: Session management with player statistics
"""

from .endpoint_config import EndpointRegistry, EndpointConfig, DatabricksAuth
from .model_client import DatabricksModelClient, PredictionResult
from .feature_engine import FeatureEngine
from .game_session import (
    GameSession,
    GameSessionManager,
    PlayerSessionStats,
    HandTracker,
    HandSummary,
    MLPredictionRecord,
)
from .service import (
    MLService,
    MLPredictionsResult,
    OpponentPrediction,
    ActionRecommendation,
    ProfitPrediction,
)

__all__ = [
    # Endpoint configuration
    "EndpointRegistry",
    "EndpointConfig",
    "DatabricksAuth",
    # Model client
    "DatabricksModelClient",
    "PredictionResult",
    # Feature engine
    "FeatureEngine",
    # Session management
    "GameSession",
    "GameSessionManager",
    "PlayerSessionStats",
    "HandTracker",
    "HandSummary",
    "MLPredictionRecord",
    # ML Service
    "MLService",
    "MLPredictionsResult",
    "OpponentPrediction",
    "ActionRecommendation",
    "ProfitPrediction",
]
