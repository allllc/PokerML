"""
Initialize the PostgreSQL database schema.

Run this script to create all tables in Cloud SQL:
    python scripts/init_db.py

Requires either:
- GCP credentials (gcloud auth application-default login)
- GOOGLE_APPLICATION_CREDENTIALS environment variable pointing to service account key
- Or DATABASE_URL environment variable for direct connection
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import Database, SCHEMA_SQL
from app.secrets import get_secrets


async def main():
    print("=" * 60)
    print("Aurora Poker - Database Initialization")
    print("=" * 60)

    # Get database URL from secrets
    secrets = get_secrets()
    print(f"\nGCP Secret Manager available: {secrets.is_gcp_available()}")

    db_url = secrets.get_database_url()
    if not db_url:
        print("\nERROR: Could not build database URL.")
        print("Make sure these secrets are configured:")
        print("  - db_user")
        print("  - db_password")
        print("  - db_name (or defaults to 'poker_ml')")
        print("  - db_host (for TCP) or CLOUD_SQL_CONNECTION_NAME (for socket)")
        sys.exit(1)

    # Mask password in output
    display_url = db_url
    if "@" in db_url:
        parts = db_url.split("@")
        creds = parts[0].split("://")[1] if "://" in parts[0] else parts[0]
        if ":" in creds:
            user = creds.split(":")[0]
            display_url = db_url.replace(creds, f"{user}:****")

    print(f"\nDatabase URL: {display_url}")

    # Connect and initialize
    db = Database(database_url=db_url)

    if not db.is_enabled:
        print("\nERROR: Database is not enabled. Check asyncpg installation.")
        sys.exit(1)

    print("\nConnecting to database...")
    connected = await db.connect()

    if not connected:
        print("\nERROR: Failed to connect to database.")
        print("Check that:")
        print("  1. The database server is running")
        print("  2. Your IP is authorized in Cloud SQL")
        print("  3. Credentials are correct")
        sys.exit(1)

    print("Connected successfully!")

    print("\nInitializing schema...")
    success = await db.initialize_schema()

    if success:
        print("\nSchema initialized successfully!")
        print("\nTables created:")
        print("  - sessions")
        print("  - hands")
        print("  - actions")
        print("  - predictions")
        print("  - player_session_stats")
    else:
        print("\nERROR: Failed to initialize schema.")
        sys.exit(1)

    await db.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
