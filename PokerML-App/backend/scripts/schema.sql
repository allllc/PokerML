-- Aurora Poker ML Database Schema
-- Run this in Cloud SQL Studio to create all tables

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

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id);
CREATE INDEX IF NOT EXISTS idx_actions_session_hand ON actions(session_id, hand_number);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id, hand_number);
CREATE INDEX IF NOT EXISTS idx_player_stats_session ON player_session_stats(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(prediction_type, street);

-- Verify tables were created
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
