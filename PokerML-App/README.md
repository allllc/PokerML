# PokerML App

A full-stack poker application with machine learning integration for training and prediction. This app connects to ML models trained in Databricks (see [PokerML-Databricks](../PokerML-Databricks/)) to provide intelligent poker recommendations and opponent modeling.

## Project Structure

```
PokerML-App/
├── engine/                 # Pure Python poker game engine
│   └── aurora_poker_engine/
│       ├── cards.py        # Card + deterministic deck
│       ├── config.py       # GameConfig schema + presets
│       ├── engine.py       # AuroraPokerEngine (reset/step/observe)
│       ├── evaluator.py    # Universal 5-card evaluator
│       ├── player.py       # PlayerState helpers
│       ├── pot.py          # Pot + side-pot models
│       └── types.py        # Observation/action dataclasses
├── backend/                # FastAPI REST API
│   └── app/
│       ├── main.py         # FastAPI entrypoint
│       ├── models.py       # Pydantic schemas
│       ├── session.py      # Table manager + bot loop
│       ├── ml/             # ML service integration
│       │   ├── service.py       # ML orchestration
│       │   ├── model_client.py  # Databricks endpoint client
│       │   └── feature_engine.py # Feature extraction
│       └── db/             # Database operations
├── frontend/               # React/Vite client
│   └── poker-client/
│       └── src/
│           └── components/ # UI components
├── agents/                 # Bot implementations
│   ├── bot_agents.py       # Heuristic bots
│   └── poker_advisor.py    # Recommendation system
└── cloudbuild.yaml         # GCP deployment config
```

## Features

### Poker Engine
- **Config-driven** – Supports NLHE, Leduc, Kuhn, and other variants via `GameConfig`
- **Gym-like API** – `reset()` and `step(bet)` interface for RL compatibility
- **Universal evaluator** – Scores arbitrary deck configurations (≤4 suits, ≤13 ranks)
- **Pot management** – Main pot + side pots with proper showdown logic

### ML Integration
This app integrates with ML models trained in Databricks:
- **Hand Strength Prediction** – Win probability based on hole cards and board
- **Opponent Modeling** – Predicts opponent action tendencies
- **Recommended Action** – Suggests optimal play based on game state

See the [PokerML-Databricks](../PokerML-Databricks/) folder for model training notebooks.

### Bot Opponents
- **Heuristic bots** – Always available, rule-based play
- **LLM bots** – Optional LangChain + OpenAI integration

## Quick Start

### Backend Setup
```bash
cd PokerML-App/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup
```bash
cd PokerML-App/frontend/poker-client
npm install
npm run dev  # default http://localhost:5173
```

### Environment Variables
Copy `.env.example` to `.env` and configure:
```
DATABRICKS_HOST=<your-databricks-host>
DATABRICKS_TOKEN=<your-token>
OPENAI_API_KEY=<optional-for-llm-bots>
ENABLE_OPENAI_BOTS=false
```

## API Reference

| Endpoint | Description |
|----------|-------------|
| `POST /api/tables` | Create a new table |
| `POST /api/tables/{id}/hand` | Start a new hand |
| `GET /api/tables/{id}/state` | Get current game state |
| `POST /api/tables/{id}/action` | Submit player action (integer chips) |
| `POST /api/tables/{id}/settings` | Toggle training mode (show all cards) |

Visit `http://localhost:8000/docs` for interactive API documentation.

## Technology Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, Pydantic, SQLAlchemy |
| Frontend | TypeScript, React, Vite |
| ML | Databricks ML Serving, LangChain |
| Deployment | Docker, Google Cloud Build |

## Deployment

The app is configured for Google Cloud Platform:
- `cloudbuild.yaml` – Full deployment pipeline
- `cloudbuild-backend.yaml` – Backend-only deployment

## License

CIS 508 Machine Learning in Business – Course Project
