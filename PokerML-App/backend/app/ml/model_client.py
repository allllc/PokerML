"""
Databricks Model Serving client.

Provides async HTTP client for calling Databricks model endpoints.
Handles authentication, retries, timeouts, and fallback predictions.
(Updated with debug logging)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import asyncio
import logging
import time

import httpx

from .endpoint_config import EndpointRegistry, EndpointConfig
from .feature_engine import (
    PROFIT_FEATURES_ORDER,
    OPPONENT_FEATURES_ORDER_PREFLOP,
    OPPONENT_FEATURES_ORDER_POSTFLOP,
    POLICY_FEATURES_ORDER,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from a model prediction call."""
    model_type: str
    street: str
    prediction: Dict[str, Any]
    latency_ms: float
    from_fallback: bool = False
    error: Optional[str] = None
    endpoint_url: Optional[str] = None

    def is_successful(self) -> bool:
        """Check if prediction succeeded (not from fallback)."""
        return not self.from_fallback and self.error is None


class DatabricksModelClient:
    """
    Async client for calling Databricks model serving endpoints.

    Usage:
        registry = EndpointRegistry()
        client = DatabricksModelClient(registry)

        # Single prediction
        result = await client.predict_opponent_strength("flop", features)

        # Multiple opponents in parallel
        results = await client.predict_all_opponents("flop", [feat1, feat2, feat3])
    """

    def __init__(self, registry: EndpointRegistry):
        """
        Initialize the model client.

        Args:
            registry: EndpointRegistry with endpoint configurations
        """
        self.registry = registry
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def predict_opponent_strength(
        self,
        street: str,
        features: Dict[str, float],
    ) -> PredictionResult:
        """
        Call opponent strength classifier.

        Args:
            street: One of 'preflop', 'flop', 'turn', 'river'
            features: Feature dictionary matching model input schema

        Returns:
            PredictionResult with predicted_class and probabilities
        """
        endpoint = self.registry.get("opponent", street)
        return await self._call_endpoint(endpoint, features, "opponent", street)

    async def predict_profit(
        self,
        street: str,
        features: Dict[str, float],
    ) -> PredictionResult:
        """
        Call profit predictor.

        Args:
            street: One of 'preflop', 'flop', 'turn', 'river'
            features: Feature dictionary matching model input schema

        Returns:
            PredictionResult with predicted_profit_bb
        """
        endpoint = self.registry.get("profit", street)
        return await self._call_endpoint(endpoint, features, "profit", street)

    async def predict_policy(
        self,
        street: str,
        features: Dict[str, float],
    ) -> PredictionResult:
        """
        Call policy action classifier.

        Args:
            street: One of 'preflop', 'flop', 'turn', 'river'
            features: Feature dictionary matching model input schema

        Returns:
            PredictionResult with predicted_action and probabilities
        """
        endpoint = self.registry.get("policy", street)
        return await self._call_endpoint(endpoint, features, "policy", street)

    async def predict_all_opponents(
        self,
        street: str,
        opponent_features: List[Dict[str, float]],
    ) -> List[PredictionResult]:
        """
        Predict strength for all opponents in parallel.

        Args:
            street: Current street
            opponent_features: List of feature dicts, one per opponent

        Returns:
            List of PredictionResults in same order as input
        """
        if not opponent_features:
            return []

        tasks = [
            self.predict_opponent_strength(street, features)
            for features in opponent_features
        ]
        return await asyncio.gather(*tasks)

    async def _call_endpoint(
        self,
        endpoint: Optional[EndpointConfig],
        features: Dict[str, float],
        model_type: str,
        street: str,
    ) -> PredictionResult:
        """
        Make HTTP call to Databricks endpoint.

        Args:
            endpoint: Endpoint configuration
            features: Input features
            model_type: Type of model
            street: Current street

        Returns:
            PredictionResult with prediction or fallback
        """
        start_time = time.perf_counter()

        # Check if endpoint is configured and enabled
        if endpoint is None:
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=self._get_fallback(model_type),
                latency_ms=0,
                from_fallback=True,
                error=f"No endpoint configured for {model_type}_{street}",
            )

        if not endpoint.enabled:
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=endpoint.fallback_prediction or self._get_fallback(model_type),
                latency_ms=0,
                from_fallback=True,
                error="Endpoint disabled",
            )

        if not endpoint.is_configured():
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=endpoint.fallback_prediction or self._get_fallback(model_type),
                latency_ms=0,
                from_fallback=True,
                error="Endpoint URL not configured",
            )

        if not self.registry.auth.is_configured():
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=endpoint.fallback_prediction or self._get_fallback(model_type),
                latency_ms=0,
                from_fallback=True,
                error="Databricks authentication not configured",
            )

        try:
            client = await self._get_client()

            # Databricks serving expects 'dataframe_split' format (like pandas to_dict(orient='split'))
            # This format has 'columns' and 'data' keys
            # CRITICAL: Use canonical feature order to match training data order
            columns = self._get_feature_order(model_type, street)
            if columns:
                # Use canonical order - extract features in that order
                data = [[features.get(col, 0.0) for col in columns]]
            else:
                # Fallback for unknown model types - use dict order
                columns = list(features.keys())
                data = [[features[col] for col in columns]]
            payload = {
                "dataframe_split": {
                    "columns": columns,
                    "data": data,
                }
            }

            response = await client.post(
                endpoint.url,
                json=payload,
                headers=self.registry.auth.get_headers(),
                timeout=endpoint.timeout_ms / 1000,
            )
            response.raise_for_status()

            result = response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log raw response for debugging
            logger.info(f"Raw Databricks response for {model_type}_{street}: {result}")

            # Parse response - Databricks returns predictions in 'predictions' key
            # The predictions is usually a list with one element per input row
            prediction = self._parse_prediction(result, model_type)

            logger.info(
                f"Parsed prediction {model_type}_{street}: {prediction} (latency: {latency_ms:.1f}ms)"
            )

            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=prediction,
                latency_ms=latency_ms,
                endpoint_url=endpoint.url,
            )

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Timeout calling {model_type}_{street} after {latency_ms:.1f}ms")
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=endpoint.fallback_prediction or self._get_fallback(model_type),
                latency_ms=latency_ms,
                from_fallback=True,
                error="Request timeout",
                endpoint_url=endpoint.url,
            )

        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_text = e.response.text
            logger.error(f"HTTP error calling {model_type}_{street}: {e.response.status_code}")
            logger.error(f"Error response body: {error_text}")
            logger.error(f"Features sent: {list(features.keys())}")
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=endpoint.fallback_prediction or self._get_fallback(model_type),
                latency_ms=latency_ms,
                from_fallback=True,
                error=f"HTTP {e.response.status_code}: {error_text[:500]}",
                endpoint_url=endpoint.url,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error calling {model_type}_{street}: {e}")
            return PredictionResult(
                model_type=model_type,
                street=street,
                prediction=endpoint.fallback_prediction or self._get_fallback(model_type),
                latency_ms=latency_ms,
                from_fallback=True,
                error=str(e),
                endpoint_url=endpoint.url,
            )

    def _parse_prediction(self, result: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        Parse Databricks model response into standardized prediction format.

        Databricks can return predictions in various formats depending on model type:
        - Classification models: may return class labels or probabilities
        - Regression models: return numeric values
        - Custom models: may return structured dicts

        Args:
            result: Raw JSON response from Databricks
            model_type: 'opponent', 'profit', or 'policy'

        Returns:
            Standardized prediction dict
        """
        predictions = result.get("predictions", [])

        # Handle empty predictions
        if not predictions:
            logger.warning(f"Empty predictions from {model_type} model: {result}")
            return self._get_fallback(model_type)

        pred = predictions[0]  # We only send 1 row

        if model_type == "opponent":
            # Opponent model: classification into 'air', 'middle', 'nutted'
            # Expected format: {"prediction": "air"} or {"prediction": 0, "probabilities": [...]}
            if isinstance(pred, dict):
                predicted_class = pred.get("prediction", pred.get("predicted_class", "middle"))
                probs = pred.get("probabilities", pred.get("probs", None))
            elif isinstance(pred, str):
                predicted_class = pred
                probs = None
            elif isinstance(pred, (int, float)):
                # Model returned class index
                classes = ["air", "middle", "nutted"]
                predicted_class = classes[int(pred)] if 0 <= int(pred) < 3 else "middle"
                probs = None
            else:
                predicted_class = "middle"
                probs = None

            # Convert probs list to dict if needed
            if isinstance(probs, list) and len(probs) >= 3:
                probs = {"air": float(probs[0]), "middle": float(probs[1]), "nutted": float(probs[2])}
            elif not isinstance(probs, dict):
                # No probabilities from model - create synthetic probs based on predicted class
                # Give the predicted class highest probability (60%), rest distributed
                classes = ["air", "middle", "nutted"]
                probs = {c: 0.2 for c in classes}
                if predicted_class in probs:
                    probs[predicted_class] = 0.6

            return {"predicted_class": predicted_class, "probabilities": probs}

        elif model_type == "profit":
            # Profit model: regression predicting profit in BB
            if isinstance(pred, dict):
                profit_bb = pred.get("prediction", pred.get("predicted_profit_bb", 0.0))
            elif isinstance(pred, (int, float)):
                profit_bb = float(pred)
            else:
                profit_bb = 0.0

            return {"predicted_profit_bb": profit_bb}

        elif model_type == "policy":
            # Policy model: 3-class classification (alphabetical order from LabelEncoder)
            actions = ["bet", "call", "fold"]

            if isinstance(pred, dict):
                predicted_action = pred.get("prediction", pred.get("predicted_action", "call"))
                probs = pred.get("probabilities", pred.get("probs", None))
            elif isinstance(pred, str):
                predicted_action = pred
                probs = None
            elif isinstance(pred, (int, float)):
                # Model returned class index
                predicted_action = actions[int(pred)] if 0 <= int(pred) < 3 else "call"
                probs = None
            else:
                predicted_action = "call"
                probs = None

            # Convert probs list to dict if needed
            if isinstance(probs, list) and len(probs) >= 3:
                probs = {a: float(p) for a, p in zip(actions, probs)}
            elif not isinstance(probs, dict):
                # No probabilities from model - create synthetic probs based on predicted action
                # Give the predicted action highest probability (60%), rest distributed
                probs = {a: 0.2 for a in actions}
                if predicted_action in probs:
                    probs[predicted_action] = 0.6

            return {"predicted_action": predicted_action, "probabilities": probs}

        else:
            return pred if isinstance(pred, dict) else {"prediction": pred}

    def _get_fallback(self, model_type: str) -> Dict[str, Any]:
        """Get default fallback prediction for a model type."""
        fallbacks = {
            "opponent": {
                "predicted_class": "middle",
                "probabilities": {"air": 0.33, "middle": 0.34, "nutted": 0.33},
            },
            "profit": {
                "predicted_profit_bb": 0.0,
                "interpretation": "unknown",
            },
            "policy": {
                "predicted_action": "call",
                "probabilities": {
                    "fold": 0.2,
                    "call": 0.4,
                    "raise_33": 0.2,
                    "raise_75": 0.1,
                    "shove": 0.1,
                },
            },
        }
        return fallbacks.get(model_type, {})

    def _get_feature_order(self, model_type: str, street: str) -> Optional[List[str]]:
        """
        Get the canonical feature order for a model type.

        CRITICAL: Features MUST be sent in this exact order to match training data.

        Args:
            model_type: 'opponent', 'profit', or 'policy'
            street: 'preflop', 'flop', 'turn', or 'river'

        Returns:
            List of feature names in canonical order, or None if unknown model type
        """
        if model_type == "profit":
            return PROFIT_FEATURES_ORDER
        elif model_type == "opponent":
            if street == "preflop":
                return OPPONENT_FEATURES_ORDER_PREFLOP
            else:
                return OPPONENT_FEATURES_ORDER_POSTFLOP
        elif model_type == "policy":
            return POLICY_FEATURES_ORDER
        else:
            return None
