"""
Databricks endpoint configuration and registry.

Manages ML model endpoint URLs, authentication, and configuration.
Endpoints can be configured via:
1. Environment variables
2. Config file (config/ml_endpoints.json)
3. Admin API at runtime
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EndpointConfig:
    """Configuration for a single Databricks model endpoint."""
    name: str
    url: str = ""
    model_version: str = "1"
    enabled: bool = False
    timeout_ms: int = 5000
    fallback_prediction: Optional[Dict[str, Any]] = None

    def is_configured(self) -> bool:
        """Check if endpoint has a valid URL configured."""
        return bool(self.url and self.url.strip())


@dataclass
class DatabricksAuth:
    """Authentication configuration for Databricks."""
    workspace_url: str = ""
    token: str = ""
    # Optional: For Service Principal auth (production)
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    def is_configured(self) -> bool:
        """Check if authentication is properly configured."""
        return bool(self.workspace_url and self.token)

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Databricks API calls."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }


# Model type definitions
STREETS = ["preflop", "flop", "turn", "river"]
MODEL_TYPES = ["opponent", "profit", "policy"]

# Fallback predictions when endpoints are unavailable
DEFAULT_FALLBACKS = {
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
            "bet": 0.3,
            "call": 0.4,
            "fold": 0.3,
        },
    },
}


class EndpointRegistry:
    """
    Registry for all ML model endpoints with easy switching.

    Supports 12 endpoints (3 model types × 4 streets):
    - opponent_preflop, opponent_flop, opponent_turn, opponent_river
    - profit_preflop, profit_flop, profit_turn, profit_river
    - policy_preflop, policy_flop, policy_turn, policy_river
    """

    CONFIG_FILE = Path("config/ml_endpoints.json")

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the endpoint registry.

        Args:
            config_path: Optional custom path for config file
        """
        if config_path:
            self.CONFIG_FILE = config_path

        # Load auth from Secret Manager or environment
        self.auth = self._load_auth()

        self._endpoints: Dict[str, EndpointConfig] = {}
        self._load_config()

        logger.info(
            f"EndpointRegistry initialized with {len(self._endpoints)} endpoints. "
            f"Auth configured: {self.auth.is_configured()}"
        )

    def _load_auth(self) -> DatabricksAuth:
        """Load Databricks auth from Secret Manager or environment variables."""
        workspace_url = ""
        token = ""
        client_id = None
        client_secret = None

        # Try Secret Manager first
        try:
            from ..secrets import get_secrets
            secrets = get_secrets()
            databricks_config = secrets.get_databricks_config()
            workspace_url = databricks_config.get("workspace_url") or ""
            token = databricks_config.get("token") or ""
            if workspace_url and token:
                logger.info("Loaded Databricks credentials from Secret Manager")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Could not load from Secret Manager: {e}")

        # Fall back to environment variables
        if not workspace_url:
            workspace_url = os.getenv("DATABRICKS_WORKSPACE_URL", "")
        if not token:
            token = os.getenv("DATABRICKS_TOKEN", "")
        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")

        return DatabricksAuth(
            workspace_url=workspace_url,
            token=token,
            client_id=client_id,
            client_secret=client_secret,
        )

    def _load_config(self) -> None:
        """Load endpoint config from file or initialize defaults."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE) as f:
                    config = json.load(f)
                    for key, data in config.get("endpoints", {}).items():
                        # Ensure fallback is set
                        model_type = key.split("_")[0]
                        if "fallback_prediction" not in data:
                            data["fallback_prediction"] = DEFAULT_FALLBACKS.get(model_type, {})
                        self._endpoints[key] = EndpointConfig(**data)
                logger.info(f"Loaded endpoint config from {self.CONFIG_FILE}")
            except Exception as e:
                logger.error(f"Error loading config: {e}. Initializing defaults.")
                self._init_default_endpoints()
        else:
            self._init_default_endpoints()

    def _init_default_endpoints(self) -> None:
        """Initialize with placeholder endpoints for all model types and streets."""
        for model_type in MODEL_TYPES:
            for street in STREETS:
                key = f"{model_type}_{street}"
                self._endpoints[key] = EndpointConfig(
                    name=key,
                    url="",
                    enabled=False,
                    fallback_prediction=DEFAULT_FALLBACKS.get(model_type, {}),
                )
        self._save_config()
        logger.info("Initialized default endpoint configuration")

    def _save_config(self) -> None:
        """Persist endpoint config to file."""
        try:
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            config = {
                "endpoints": {
                    key: {
                        "name": ep.name,
                        "url": ep.url,
                        "model_version": ep.model_version,
                        "enabled": ep.enabled,
                        "timeout_ms": ep.timeout_ms,
                    }
                    for key, ep in self._endpoints.items()
                }
            }
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            logger.debug(f"Saved endpoint config to {self.CONFIG_FILE}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, model_type: str, street: str) -> Optional[EndpointConfig]:
        """
        Get endpoint config for a model type and street.

        Args:
            model_type: One of 'opponent', 'profit', 'policy'
            street: One of 'preflop', 'flop', 'turn', 'river'

        Returns:
            EndpointConfig or None if not found
        """
        key = f"{model_type}_{street}"
        return self._endpoints.get(key)

    def get_by_key(self, key: str) -> Optional[EndpointConfig]:
        """Get endpoint by its key (e.g., 'opponent_preflop')."""
        return self._endpoints.get(key)

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all endpoint configurations."""
        return [
            {
                "key": key,
                "name": ep.name,
                "url": ep.url,
                "enabled": ep.enabled,
                "configured": ep.is_configured(),
                "model_version": ep.model_version,
                "timeout_ms": ep.timeout_ms,
            }
            for key, ep in sorted(self._endpoints.items())
        ]

    def update_endpoint(
        self,
        key: str,
        url: Optional[str] = None,
        enabled: Optional[bool] = None,
        model_version: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> EndpointConfig:
        """
        Update endpoint configuration.

        Args:
            key: Endpoint key (e.g., 'opponent_preflop')
            url: New endpoint URL
            enabled: Enable/disable endpoint
            model_version: Model version string
            timeout_ms: Request timeout in milliseconds

        Returns:
            Updated EndpointConfig

        Raises:
            KeyError: If endpoint key not found
        """
        if key not in self._endpoints:
            raise KeyError(f"Unknown endpoint: {key}")

        ep = self._endpoints[key]
        if url is not None:
            ep.url = url
        if enabled is not None:
            ep.enabled = enabled
        if model_version is not None:
            ep.model_version = model_version
        if timeout_ms is not None:
            ep.timeout_ms = timeout_ms

        self._save_config()
        logger.info(f"Updated endpoint {key}: url={ep.url}, enabled={ep.enabled}")
        return ep

    def update_auth(
        self,
        workspace_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        """
        Update Databricks authentication credentials.

        Note: These are stored in memory only, not persisted to file.
        For production, use environment variables or Secret Manager.
        """
        if workspace_url:
            self.auth.workspace_url = workspace_url
        if token:
            self.auth.token = token
        logger.info(f"Updated auth. Configured: {self.auth.is_configured()}")

    def get_enabled_endpoints(self, model_type: Optional[str] = None) -> List[str]:
        """Get list of enabled endpoint keys, optionally filtered by model type."""
        keys = []
        for key, ep in self._endpoints.items():
            if ep.enabled and ep.is_configured():
                if model_type is None or key.startswith(f"{model_type}_"):
                    keys.append(key)
        return keys

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of endpoint status for health checks."""
        total = len(self._endpoints)
        configured = sum(1 for ep in self._endpoints.values() if ep.is_configured())
        enabled = sum(1 for ep in self._endpoints.values() if ep.enabled)

        return {
            "total_endpoints": total,
            "configured": configured,
            "enabled": enabled,
            "auth_configured": self.auth.is_configured(),
            "by_model_type": {
                mt: {
                    "configured": sum(
                        1 for k, ep in self._endpoints.items()
                        if k.startswith(f"{mt}_") and ep.is_configured()
                    ),
                    "enabled": sum(
                        1 for k, ep in self._endpoints.items()
                        if k.startswith(f"{mt}_") and ep.enabled
                    ),
                }
                for mt in MODEL_TYPES
            },
        }
