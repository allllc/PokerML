"""
GCP Secret Manager integration for secure credential management.

Loads secrets from GCP Secret Manager when running in GCP,
falls back to environment variables for local development.
"""

import os
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import GCP Secret Manager
try:
    from google.cloud import secretmanager
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logger.info("google-cloud-secret-manager not installed - using env vars only")


class SecretManager:
    """
    Manages secrets from GCP Secret Manager with env var fallback.

    Usage:
        secrets = SecretManager(project_id="mlpoker")
        db_password = secrets.get("db_password")
    """

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize secret manager.

        Args:
            project_id: GCP project ID. If None, reads from GCP_PROJECT env var.
        """
        self._project_id = project_id or os.getenv("GCP_PROJECT", "mlpoker")
        self._client = None
        self._cache: dict[str, str] = {}

        if GCP_AVAILABLE:
            try:
                self._client = secretmanager.SecretManagerServiceClient()
                logger.info(f"Secret Manager client initialized for project: {self._project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Secret Manager client: {e}")
                self._client = None

    def get(self, secret_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.

        First checks cache, then GCP Secret Manager, then environment variables.

        Args:
            secret_name: Name of the secret (e.g., "db_password")
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # Check cache first
        if secret_name in self._cache:
            return self._cache[secret_name]

        # Try GCP Secret Manager
        if self._client:
            try:
                value = self._get_from_gcp(secret_name)
                if value:
                    self._cache[secret_name] = value
                    return value
            except Exception as e:
                logger.debug(f"Secret '{secret_name}' not in GCP: {e}")

        # Fall back to environment variable
        # Try both formats: SECRET_NAME and secret_name
        env_value = os.getenv(secret_name.upper()) or os.getenv(secret_name)
        if env_value:
            self._cache[secret_name] = env_value
            return env_value

        return default

    def _get_from_gcp(self, secret_name: str) -> Optional[str]:
        """Fetch secret from GCP Secret Manager."""
        if not self._client:
            return None

        name = f"projects/{self._project_id}/secrets/{secret_name}/versions/latest"

        try:
            response = self._client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.debug(f"Could not access secret {secret_name}: {e}")
            return None

    def get_database_url(self) -> Optional[str]:
        """
        Build PostgreSQL connection URL from secrets.

        Supports both Cloud SQL (Unix socket) and direct TCP connections.
        Prefers direct TCP (db_host) when available, falls back to Cloud SQL socket.

        Returns:
            PostgreSQL connection string or None if not configured
        """
        from urllib.parse import quote

        # Check for pre-built DATABASE_URL first
        db_url = self.get("DATABASE_URL")
        if db_url:
            return db_url

        # Build from components
        db_user = self.get("db_user")
        db_password = self.get("db_password")
        db_name = self.get("db_name", "poker_ml")

        if not db_user or not db_password:
            logger.info("Database credentials not configured")
            return None

        # URL-encode user and password to handle special characters
        db_user_encoded = quote(db_user, safe='')
        db_password_encoded = quote(db_password, safe='')

        # Prefer direct TCP connection when db_host is set
        db_host = self.get("db_host")
        if db_host:
            db_port = self.get("db_port", "5432")
            logger.info(f"Using TCP connection to {db_host}:{db_port}/{db_name}")
            return f"postgresql://{db_user_encoded}:{db_password_encoded}@{db_host}:{db_port}/{db_name}"

        # Fall back to Cloud SQL Unix socket connection
        cloud_sql_instance = self.get("CLOUD_SQL_CONNECTION_NAME")
        if cloud_sql_instance:
            logger.info(f"Using Cloud SQL socket: {cloud_sql_instance}")
            return f"postgresql://{db_user_encoded}:{db_password_encoded}@/{db_name}?host=/cloudsql/{cloud_sql_instance}"

        logger.info("No database host configured")
        return None

    def get_databricks_config(self) -> dict:
        """
        Get Databricks configuration from secrets.

        Returns:
            Dict with workspace_url and token (may be None if not configured)
        """
        return {
            "workspace_url": self.get("databricks_workspace_url"),
            "token": self.get("databricks_token"),
            "user": self.get("databricks_user"),
        }

    def is_gcp_available(self) -> bool:
        """Check if GCP Secret Manager is available."""
        return self._client is not None


# Global instance
_secrets: Optional[SecretManager] = None


def get_secrets() -> SecretManager:
    """Get the global SecretManager instance."""
    global _secrets
    if _secrets is None:
        _secrets = SecretManager()
    return _secrets
