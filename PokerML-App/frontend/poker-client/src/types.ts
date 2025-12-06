export type Card = {
  rank: string
  suit: string
  id: number
}

export type SidePot = {
  amount: number
  eligibleSeats: number[]
}

export type PotState = {
  mainPot: number
  sidePots: SidePot[]
}

export type LegalAction =
  | { type: 'FOLD' }
  | { type: 'CHECK' }
  | { type: 'CALL'; amount: number }
  | { type: 'RAISE'; min: number; max: number }

export type PublicPlayerView = {
  seatIndex: number
  name: string
  stack: number
  betThisStreet: number
  totalCommitted: number
  isFolded: boolean
  isAllIn: boolean
  hasButton: boolean
  visibility: 'HIDDEN' | 'FACE_UP'
  holeCards?: Card[] | null
  styleLabel?: string | null
  agentDebug?: string | null
}

export type AgentHistoryEntry = {
  handId: number
  seatIndex: number
  streetIndex: number
  streetName?: string | null
  debug: Record<string, unknown>
}

export type PublicGameState = {
  tableId: string
  handId: number
  phase: 'PRE_HAND' | 'BETTING' | 'SHOWDOWN' | 'FINISHED'
  streetName: 'PREFLOP' | 'FLOP' | 'TURN' | 'RIVER' | null
  heroSeat: number
  activeSeat: number | null
  players: PublicPlayerView[]
  boardCards: Card[]
  pot: PotState
  callAmount: number | null
  minRaise: number | null
  maxRaise: number | null
  legalActions: LegalAction[]
  showAllCards: boolean
  agentHistory: AgentHistoryEntry[]
  winners?: number[] | null
  payouts?: number[] | null
}

export type CreateTableResponse = {
  tableId: string
  state: PublicGameState
}

// Legacy types (kept for backwards compatibility)
export type HandQuality = 'Air' | 'Medium' | 'Nuts'
export type PlayStyle = 'By the Book' | 'Range' | 'Aggressive'

export type PlayerMLPrediction = {
  seatIndex: number
  handQuality: HandQuality
  expectedValue: number
}

export type MLRecommendation = {
  action: 'Check' | 'Fold' | 'Call' | 'Raise'
  raiseAmount?: number
  confidence: number
  reasoning: string
}

export type MLPredictions = {
  players: PlayerMLPrediction[]
  heroRecommendation: MLRecommendation | null
  playStyle: PlayStyle
}

// ==================== New ML Types (Databricks Integration) ====================

export type OpponentStrength = 'air' | 'middle' | 'nutted'
export type PolicyAction = 'fold' | 'call' | 'bet'

export type OpponentPrediction = {
  seatIndex: number
  playerName: string
  predictedClass: OpponentStrength
  probabilities: Record<OpponentStrength, number>
  confidence: number
  fromFallback: boolean
}

export type ProfitPrediction = {
  predictedProfitBB: number
  interpretation: string
  fromFallback: boolean
}

export type ActionRecommendation = {
  action: PolicyAction
  chipAmount: number
  probabilities: Record<PolicyAction, number>
  confidence: number
  reasoning: string
  fromFallback: boolean
}

export type OpponentDebugInfo = {
  playerName: string
  features: {
    action_type: string
    bet_pct_pot: number
    pot_size: number
    amount: number
    pot_contribution: number
    starting_stack: number
    stack_vs_table_median: number
    raises_so_far: number
    calls_so_far: number
    vpip_last5: number
    pfr_last5: number
    agg_factor_last5: number
    isAllIn: boolean
  }
  rawOutput: Record<string, unknown> | null
}

export type MLDebugInfo = {
  opponentInputs: OpponentDebugInfo[]
  profitInput: Record<string, unknown>
  profitRawOutput: Record<string, unknown>
  policyInput: Record<string, unknown>
  policyRawOutput: Record<string, unknown>
}

export type MLPredictionsResponse = {
  handId: number
  street: string
  heroSeat: number
  actionIndex: number
  opponentPredictions: OpponentPrediction[]
  profitPrediction: ProfitPrediction | null
  actionRecommendation: ActionRecommendation | null
  latenciesMs: Record<string, number>
  anyFromFallback: boolean
  timestamp: string
  debugInfo?: MLDebugInfo
}

// Prediction history for review
export type PredictionHistoryEntry = {
  handNumber: number
  street: string
  actionIndex: number
  predictionType: 'opponent' | 'profit' | 'policy'
  targetSeat: number | null
  prediction: Record<string, unknown>
  latencyMs: number
  fromFallback: boolean
  timestamp: string
}

export type PredictionHistoryResponse = {
  session_id: string
  predictions: PredictionHistoryEntry[]
}

// ML Admin types
export type MLEndpointConfig = {
  key: string
  name: string
  url: string
  enabled: boolean
  configured: boolean
  model_version: string
  timeout_ms: number
}

export type MLStatusResponse = {
  ml_service_ready: boolean
  database_enabled: boolean
  endpoints: {
    total_endpoints: number
    configured: number
    enabled: number
    auth_configured: boolean
  }
  active_sessions: number
}

// ==================== ML Stats Types (Debug) ====================

export type PlayerStatsEntry = {
  seatIndex: number
  handsPlayed: number
  vpip: number
  pfr: number
  aggressionFactor: number
  stickiness: number
  cumulativeStats: {
    voluntaryPreflop: number
    preflopRaise: number
    aggressiveActions: number
    passiveActions: number
    sawFlop: number
    sawTurn: number
    sawRiver: number
    showdownCount: number
  }
  recentHandsCount: number
  currentHand?: {
    voluntaryPreflop: boolean
    preflopRaise: boolean
    sawFlop: boolean
    sawTurn: boolean
    sawRiver: boolean
    aggressiveActions: number
    passiveActions: number
  } | null
}

export type CurrentHandInfo = {
  handNumber: number
  actionCount: number
  raiseCount: number
  callCount: number
  currentStreet: string
}

export type MLFeaturesSnapshot = {
  handStrength: number
  potOdds: number
  opponentsActive: number
  potSize: number
  facingBet: number
}

export type MLStatsResponse = {
  tableId: string
  sessionId: string
  street: string
  handNumber: number
  currentHand: CurrentHandInfo | null
  playerStats: Record<string, PlayerStatsEntry>
  features: MLFeaturesSnapshot | null
  allFeatures: Record<string, number>
  featureCount: number
  sessionActive: boolean
}
