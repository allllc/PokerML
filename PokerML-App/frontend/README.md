# Poker Client (React + Vite)

This folder now contains a working React UI that talks to the FastAPI backend and renders the oval poker table, seat panels, action bar, and training controls described in the spec.

## Getting Started

```bash
cd AuroraPokerApp/frontend/poker-client
npm install          # already done once, but safe to repeat
npm run dev          # starts Vite dev server (default http://localhost:5173)
```

Set `VITE_API_BASE` in a `.env` file if your backend is not running on `http://localhost:8000`:

```
VITE_API_BASE=http://127.0.0.1:8000
```

With both servers running:

1. `npm run dev` (frontend) → open the printed localhost URL.
2. Click **Create New Table** to call `POST /api/tables`. The app auto-seats bots and shows the hero action bar once it is your turn.
3. Use the Fold/Check/Call/Raise controls. Bets are routed to `POST /api/tables/{id}/action`; bots auto-play via the backend’s `autoAdvance=true` flag.
4. When the hand ends the Action Bar swaps to a **Start Next Hand** button (calls `POST /hand`).
5. The Training panel’s show-all toggle persists via `POST /settings`.

## UI Overview

- **Table felt** – circular canvas hosting six seat panels. Each panel shows name, stack, status badges, and hole cards (face-up for hero/showdown/show-all).
- **Board + Pot** – center row of community cards and pot breakdown (main + side pots).
- **Action bar** – hero-only controls with Fold/Check/Call and a raise slider bounded by the backend’s min/max raise values. When waiting on bots, it displays a “Waiting for opponents…” banner.
- **Training panel** – show-all toggle, hand summary, and quick list of seat statuses.

The layout is desktop-first but collapses vertically on smaller widths.

## Extending the Client

- Wire in a real coach overlay by adding another panel section fed from a `/coach` endpoint.
- Store snapshots of `PublicGameState` in a log to build replay controls.
- Replace the placeholder seat positioning with a responsive polar layout to support up to 9 seats or multiple presets.
