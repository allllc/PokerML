# Backend Dockerfile for GCP Cloud Run
# NOTE: Build from project root with: docker build -f backend/Dockerfile .
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the engine module (required by backend)
COPY engine /app/engine

# Copy the agents module (required for AI advisor)
COPY agents /app/agents

# Copy the backend application
COPY backend/app /app/app

# Copy the config folder (ml_endpoints.json)
COPY backend/config /app/config

# Set Python path to include engine
ENV PYTHONPATH=/app:/app/engine

# Cloud Run sets PORT environment variable
ENV PORT=8080

# Run the application
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
