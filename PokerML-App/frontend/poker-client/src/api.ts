import type {
  CreateTableResponse,
  PublicGameState,
  MLPredictionsResponse,
  PredictionHistoryResponse,
  MLStatusResponse,
  MLEndpointConfig,
  MLStatsResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || response.statusText)
  }
  return (await response.json()) as T
}

export async function createTable(): Promise<CreateTableResponse> {
  return request<CreateTableResponse>('/api/tables', {
    method: 'POST',
    body: JSON.stringify({}),
  })
}

export async function startHand(tableId: string): Promise<PublicGameState> {
  return request<PublicGameState>(`/api/tables/${tableId}/hand`, {
    method: 'POST',
    body: JSON.stringify({ autoAdvance: true }),
  })
}

export async function act(tableId: string, viewerSeat: number, betAmount: number): Promise<PublicGameState> {
  return request<PublicGameState>(`/api/tables/${tableId}/action`, {
    method: 'POST',
    body: JSON.stringify({ viewerSeat, betAmount, autoAdvance: true }),
  })
}

export async function toggleShowAll(tableId: string, showAllCards: boolean): Promise<PublicGameState> {
  return request<PublicGameState>(`/api/tables/${tableId}/settings`, {
    method: 'POST',
    body: JSON.stringify({ showAllCards }),
  })
}

export async function fetchState(tableId: string): Promise<PublicGameState> {
  return request<PublicGameState>(`/api/tables/${tableId}/state`)
}

export async function advanceBots(tableId: string): Promise<PublicGameState> {
  return request<PublicGameState>(`/api/tables/${tableId}/advance`, { method: 'POST' })
}

// ==================== ML Prediction API ====================

export async function getMLPredictions(tableId: string): Promise<MLPredictionsResponse> {
  return request<MLPredictionsResponse>(`/api/tables/${tableId}/ml/predict`, {
    method: 'POST',
  })
}

export async function getPredictionHistory(tableId: string): Promise<PredictionHistoryResponse> {
  return request<PredictionHistoryResponse>(`/api/tables/${tableId}/ml/history`)
}

export async function getMLStats(tableId: string): Promise<MLStatsResponse> {
  return request<MLStatsResponse>(`/api/tables/${tableId}/ml/stats`)
}

// ==================== AI Advisor API ====================

export interface AIAdvisorRequest {
  tableId: string
  riskLevel: number  // 0-100
  playStyle: number  // 0-100
  gameState?: Record<string, unknown>
  mlPredictions?: Record<string, unknown>
  rawFeatures?: Record<string, unknown>  // Raw ML features with explanations
  handHistory?: Array<Record<string, unknown>>  // Betting actions this hand
}

export interface AIAdvisorResponse {
  advice: string
  model: string
  riskLevel: number
  playStyle: number
  promptUsed: string
}

export async function getAIAdvice(req: AIAdvisorRequest): Promise<AIAdvisorResponse> {
  return request<AIAdvisorResponse>('/api/ai-advisor', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export async function endSession(sessionId: string): Promise<void> {
  await request<{ status: string }>(`/api/sessions/${sessionId}`, {
    method: 'DELETE',
  })
}

// ==================== ML Admin API ====================

export async function getMLStatus(): Promise<MLStatusResponse> {
  return request<MLStatusResponse>('/api/admin/ml/status')
}

export async function getMLEndpoints(): Promise<MLEndpointConfig[]> {
  return request<MLEndpointConfig[]>('/api/admin/ml/endpoints')
}

export async function updateMLEndpoint(
  key: string,
  config: { url?: string; enabled?: boolean; timeout_ms?: number }
): Promise<MLEndpointConfig> {
  return request<MLEndpointConfig>(`/api/admin/ml/endpoints/${key}`, {
    method: 'PUT',
    body: JSON.stringify(config),
  })
}

export async function updateMLAuth(auth: {
  workspace_url?: string
  token?: string
}): Promise<{ status: string }> {
  return request<{ status: string }>('/api/admin/ml/auth', {
    method: 'PUT',
    body: JSON.stringify(auth),
  })
}
