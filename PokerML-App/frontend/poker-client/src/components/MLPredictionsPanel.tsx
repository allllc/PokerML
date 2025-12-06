import { useState, useEffect, useCallback } from 'react'
import type {
  PublicGameState,
  MLPredictionsResponse,
  OpponentStrength,
  PolicyAction,
  PredictionHistoryEntry,
  MLStatsResponse,
} from '../types'
import { getMLStats, getAIAdvice, type AIAdvisorResponse } from '../api'
import './MLPredictionsPanel.css'

type Props = {
  state: PublicGameState
  predictions: MLPredictionsResponse | null
  predictionHistory: PredictionHistoryEntry[]
  isLoading: boolean
  showAllCards: boolean
  onToggleShowAll: (enabled: boolean) => void
  showStyles: boolean
  onToggleStyles: (enabled: boolean) => void
  showAgentDebug: boolean
  onToggleAgentDebug: (enabled: boolean) => void
}

type ViewMode = 'predictions' | 'debug' | 'ai' | 'info'

export function MLPredictionsPanel({
  state,
  predictions,
  predictionHistory: _predictionHistory,
  isLoading,
  showAllCards,
  onToggleShowAll,
  showStyles,
  onToggleStyles,
  showAgentDebug,
  onToggleAgentDebug,
}: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>('predictions')
  const [_mlStats, setMlStats] = useState<MLStatsResponse | null>(null)
  const [_statsLoading, setStatsLoading] = useState(false)
  const [_showAllFeatures, _setShowAllFeatures] = useState(false)

  // AI Advisor state
  const [riskLevel, setRiskLevel] = useState(50)  // 0-100: Low Risk to High Risk
  const [playStyle, setPlayStyle] = useState(50)  // 0-100: By The Book to Variable Pro
  const [aiAdvice, setAiAdvice] = useState<AIAdvisorResponse | null>(null)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiError, setAiError] = useState<string | null>(null)
  const [showPromptModal, setShowPromptModal] = useState(false)
  const [selectedBotPrompt, setSelectedBotPrompt] = useState<{name: string, prompt: string} | null>(null)

  // Fetch stats when switching to stats tab or when game state changes
  const fetchStats = useCallback(async () => {
    if (!state.tableId) return
    setStatsLoading(true)
    try {
      const stats = await getMLStats(state.tableId)
      setMlStats(stats)
    } catch (err) {
      console.error('Failed to fetch ML stats:', err)
    } finally {
      setStatsLoading(false)
    }
  }, [state.tableId])

  // Fetch AI advice
  const fetchAIAdvice = useCallback(async () => {
    if (!state.tableId) return
    setAiLoading(true)
    setAiError(null)
    try {
      // Build game state info for the AI
      const heroPlayer = state.players.find(p => p.seatIndex === state.heroSeat)
      const bb = 2  // Big blind size (could be dynamic if game config is exposed)

      const gameStateInfo = {
        street: state.streetName || 'preflop',
        potSize: state.pot?.mainPot || 0,
        facingBet: state.callAmount || 0,
        heroStack: heroPlayer?.stack || 0,
        boardCards: state.boardCards?.map(c => `${c.rank}${c.suit}`).join(' ') || 'none',
        heroCards: heroPlayer?.holeCards?.map(c => `${c.rank}${c.suit}`).join(' ') || 'unknown',
        position: heroPlayer ? (heroPlayer.seatIndex - (state.players.find(p => p.hasButton)?.seatIndex || 0) + state.players.length) % state.players.length : 0,
        bb: bb,
        minRaise: state.minRaise || 0,
        maxRaise: state.maxRaise || heroPlayer?.stack || 0,
      }

      // Build ML predictions info
      const mlPredictionsInfo = predictions ? {
        opponentPredictions: predictions.opponentPredictions,
        profitPrediction: predictions.profitPrediction,
        actionRecommendation: predictions.actionRecommendation,
      } : {}

      // Extract raw features from debug info (policy features are the most comprehensive)
      const rawFeatures = predictions?.debugInfo?.policyInput || predictions?.debugInfo?.profitInput || {}

      // Build hand history from agent history
      const handHistory = state.agentHistory?.map(entry => ({
        streetName: entry.streetName,
        seatIndex: entry.seatIndex,
        playerName: state.players.find(p => p.seatIndex === entry.seatIndex)?.name || `Seat ${entry.seatIndex}`,
        action: (entry.debug as { final_action?: string })?.final_action || 'unknown',
        amount: (entry.debug as { final_bet?: number })?.final_bet || 0,
      })) || []

      const advice = await getAIAdvice({
        tableId: state.tableId,
        riskLevel,
        playStyle,
        gameState: gameStateInfo,
        mlPredictions: mlPredictionsInfo,
        rawFeatures: rawFeatures as Record<string, unknown>,
        handHistory: handHistory,
      })
      setAiAdvice(advice)
    } catch (err) {
      console.error('Failed to fetch AI advice:', err)
      setAiError(err instanceof Error ? err.message : 'Failed to get AI advice')
    } finally {
      setAiLoading(false)
    }
  }, [state.tableId, state.players, state.heroSeat, state.streetName, state.pot, state.callAmount, state.boardCards, state.minRaise, state.maxRaise, state.agentHistory, predictions, riskLevel, playStyle])

  // Auto-refresh stats when on stats tab and game state changes (keeping for compatibility)
  useEffect(() => {
    // No longer auto-fetch stats since we removed the stats view
    // Stats are still available via the ML Stats API if needed
  }, [viewMode, state.handId, state.streetName, state.activeSeat, fetchStats])

  // Auto-trigger AI advice when it's the hero's turn and predictions are available
  useEffect(() => {
    const isHeroTurn = state.activeSeat === state.heroSeat && state.phase === 'BETTING'
    const hasPredictions = predictions && predictions.opponentPredictions.length > 0

    if (isHeroTurn && hasPredictions && !aiLoading && !aiAdvice) {
      // Auto-fetch AI advice when hero's turn starts
      fetchAIAdvice()
    }
  }, [state.activeSeat, state.heroSeat, state.phase, predictions, aiLoading, aiAdvice, fetchAIAdvice])

  // Clear AI advice when the situation changes (new street, new hand, new table, or hero acts)
  useEffect(() => {
    setAiAdvice(null)
    setAiError(null)
  }, [state.handId, state.streetName, state.tableId])

  // Re-trigger AI advice when risk level or play style changes (only if we already have advice)
  useEffect(() => {
    const isHeroTurn = state.activeSeat === state.heroSeat && state.phase === 'BETTING'
    const hasPredictions = predictions && predictions.opponentPredictions.length > 0

    if (isHeroTurn && hasPredictions && aiAdvice && !aiLoading) {
      // Re-fetch with new settings
      fetchAIAdvice()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [riskLevel, playStyle])

  const getStrengthColor = (strength: OpponentStrength) => {
    switch (strength) {
      case 'nutted':
        return '#4caf50'
      case 'middle':
        return '#ffd166'
      case 'air':
        return '#ff6b6b'
      default:
        return '#8c93b8'
    }
  }

  const getActionColor = (action: PolicyAction | string) => {
    switch (action) {
      case 'bet':
        return '#4caf50'  // Green for aggressive actions
      case 'call':
        return '#4bd4ff'  // Blue for passive actions
      case 'fold':
        return '#ff6b6b'  // Red for fold
      default:
        return '#8c93b8'
    }
  }

  const formatAction = (action: PolicyAction): string => {
    switch (action) {
      case 'bet':
        return 'Bet/Raise'  // Any aggressive action
      case 'call':
        return 'Call/Check'  // Passive action
      case 'fold':
        return 'Fold'
      default:
        return action
    }
  }

  const formatStrength = (strength: OpponentStrength): string => {
    switch (strength) {
      case 'nutted':
        return 'Strong'
      case 'middle':
        return 'Medium'
      case 'air':
        return 'Weak'
      default:
        return strength
    }
  }

  return (
    <div className="ml-predictions-panel">
      <div className="panel-header">
        <div className="view-toggle">
          <button
            className={viewMode === 'predictions' ? 'active' : ''}
            onClick={() => setViewMode('predictions')}
          >
            ML Advisor
          </button>
          <button
            className={viewMode === 'ai' ? 'active' : ''}
            onClick={() => setViewMode('ai')}
          >
            AI Advisor
          </button>
          <button
            className={viewMode === 'debug' ? 'active' : ''}
            onClick={() => setViewMode('debug')}
          >
            Debug
          </button>
          <button className={viewMode === 'info' ? 'active' : ''} onClick={() => setViewMode('info')}>
            Info
          </button>
        </div>
      </div>

      {viewMode === 'predictions' ? (
        <div className="predictions-view">
          {isLoading && (
            <div className="loading-indicator">
              <span className="spinner" />
              Loading predictions...
            </div>
          )}

          {/* Action Recommendation */}
          {predictions?.actionRecommendation && (
            <div className="panel-section hero-recommendation">
              <h4>Recommended Action</h4>
              <div
                className="action-badge"
                style={{ borderColor: getActionColor(predictions.actionRecommendation.action) }}
              >
                <div
                  className="action-name"
                  style={{ color: getActionColor(predictions.actionRecommendation.action) }}
                >
                  {formatAction(predictions.actionRecommendation.action)}
                  {predictions.actionRecommendation.chipAmount > 0 &&
                    ` (${predictions.actionRecommendation.chipAmount})`}
                </div>
              </div>

              {/* Action Probability Boxes */}
              {predictions.debugInfo?.policyRawOutput && (
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem', marginBottom: '0.75rem' }}>
                  {(['fold', 'call', 'bet'] as const).map((action) => {
                    const probs = (predictions.debugInfo?.policyRawOutput as {probabilities?: Record<string, number>})?.probabilities || {}
                    const prob = probs[action] || 0
                    const isSelected = predictions.actionRecommendation?.action === action
                    const bgColor = action === 'fold' ? 'rgba(255, 107, 107, 0.2)' :
                                   action === 'call' ? 'rgba(75, 212, 255, 0.2)' :
                                   'rgba(76, 175, 80, 0.2)'
                    const borderColor = action === 'fold' ? '#ff6b6b' :
                                       action === 'call' ? '#4bd4ff' :
                                       '#4caf50'
                    return (
                      <div
                        key={action}
                        style={{
                          flex: 1,
                          padding: '0.5rem',
                          borderRadius: '6px',
                          background: bgColor,
                          border: isSelected ? `2px solid ${borderColor}` : '1px solid rgba(140, 147, 184, 0.3)',
                          textAlign: 'center',
                        }}
                      >
                        <div style={{
                          fontSize: '0.7rem',
                          color: borderColor,
                          fontWeight: isSelected ? 'bold' : 'normal',
                          textTransform: 'uppercase',
                          marginBottom: '0.25rem'
                        }}>
                          {action}
                        </div>
                        <div style={{
                          fontSize: '1rem',
                          fontWeight: 'bold',
                          color: isSelected ? borderColor : '#ddd'
                        }}>
                          {(prob * 100).toFixed(0)}%
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}

              <p className="reasoning">{predictions.actionRecommendation.reasoning}</p>
              {predictions.actionRecommendation.fromFallback && (
                <span className="fallback-badge">Fallback</span>
              )}
            </div>
          )}

          {/* Profit Prediction */}
          {predictions?.profitPrediction && (
            <div className="panel-section profit-prediction">
              <h4>Expected Profit</h4>
              <div
                className={`profit-value ${predictions.profitPrediction.predictedProfitBB >= 0 ? 'positive' : 'negative'}`}
              >
                {predictions.profitPrediction.predictedProfitBB >= 0 ? '+' : ''}
                {predictions.profitPrediction.predictedProfitBB.toFixed(2)} BB
              </div>
              <p className="interpretation">{predictions.profitPrediction.interpretation}</p>
              {predictions.profitPrediction.fromFallback && (
                <span className="fallback-badge">Fallback</span>
              )}
            </div>
          )}

          {/* Opponent Predictions */}
          <div className="panel-section">
            <h4>Opponent Hand Strength</h4>
            <div className="player-predictions">
              {predictions?.opponentPredictions.map((pred) => {
                const player = state.players.find((p) => p.seatIndex === pred.seatIndex)
                if (!player) return null
                return (
                  <div key={pred.seatIndex} className="player-pred-row">
                    <span className="player-name">{pred.playerName}</span>
                    <span
                      className="hand-quality-badge"
                      style={{
                        background: `${getStrengthColor(pred.predictedClass)}33`,
                        borderColor: getStrengthColor(pred.predictedClass),
                        color: getStrengthColor(pred.predictedClass),
                      }}
                    >
                      {formatStrength(pred.predictedClass)}
                    </span>
                    {pred.fromFallback && <span className="fallback-dot" title="Fallback prediction" />}
                  </div>
                )
              })}
              {(!predictions || predictions.opponentPredictions.length === 0) && !isLoading && (
                <p className="placeholder">No predictions available yet. Start a hand to see ML insights.</p>
              )}
            </div>
          </div>

          {/* Latency Info */}
          {predictions && (
            <div className="panel-section latency-info">
              <h4>Model Latencies</h4>
              <div className="latency-grid">
                {Object.entries(predictions.latenciesMs).map(([model, ms]) => (
                  <div key={model} className="latency-item">
                    <span className="latency-model">{model}</span>
                    <span className="latency-value">{ms.toFixed(0)}ms</span>
                  </div>
                ))}
              </div>
              {predictions.anyFromFallback && (
                <p className="fallback-notice">Some predictions used fallback models</p>
              )}
            </div>
          )}
        </div>
      ) : viewMode === 'debug' ? (
        <div className="debug-view">
          {predictions?.debugInfo ? (
            <>
              {/* Opponent Model Debug */}
              <div className="panel-section">
                <h4>Opponent Model</h4>
                {predictions.debugInfo.opponentInputs && predictions.debugInfo.opponentInputs.length > 0 ? (
                  <div className="debug-list">
                    {predictions.debugInfo.opponentInputs.map((opp, idx) => (
                      <div key={idx} style={{ marginBottom: '0.75rem' }}>
                        <div className="debug-header" style={{ marginBottom: '0.25rem' }}>{opp.playerName}</div>
                        {/* Input Box */}
                        <div className="debug-entry" style={{ marginBottom: '0.25rem' }}>
                          <div className="debug-header" style={{ color: '#8c93b8', fontSize: '0.65rem' }}>INPUT</div>
                          <div className="debug-row">
                            <span className="debug-label">action_type:</span>
                            <span className="debug-value" style={{
                              color: opp.features.action_type === 'bet_or_raise_to' ? '#4caf50' : '#999'
                            }}>
                              {opp.features.action_type}
                            </span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">amount:</span>
                            <span className="debug-value">{opp.features.amount}</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">bet_pct_pot:</span>
                            <span className="debug-value">{((opp.features.bet_pct_pot || 0) * 100).toFixed(1)}%</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">pot_size:</span>
                            <span className="debug-value">{opp.features.pot_size}</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">pot_contribution:</span>
                            <span className="debug-value">{opp.features.pot_contribution}</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">starting_stack:</span>
                            <span className="debug-value">{opp.features.starting_stack}</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">stack_vs_median:</span>
                            <span className="debug-value">{opp.features.stack_vs_table_median}x</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">raises/calls:</span>
                            <span className="debug-value">{opp.features.raises_so_far}/{opp.features.calls_so_far}</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">vpip_last5:</span>
                            <span className="debug-value">{(opp.features.vpip_last5 || 0).toFixed(0)}%</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">pfr_last5:</span>
                            <span className="debug-value">{(opp.features.pfr_last5 || 0).toFixed(0)}%</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">agg_factor:</span>
                            <span className="debug-value">{opp.features.agg_factor_last5}</span>
                          </div>
                          <div className="debug-row">
                            <span className="debug-label">isAllIn:</span>
                            <span className="debug-value">{opp.features.isAllIn ? 'Yes' : 'No'}</span>
                          </div>
                        </div>
                        {/* Output Box */}
                        <div className="debug-entry" style={{ background: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
                          <div className="debug-header" style={{ color: '#4caf50', fontSize: '0.65rem' }}>OUTPUT</div>
                          <div className="debug-row">
                            <span className="debug-label">predicted_class:</span>
                            <span className="debug-value" style={{
                              fontWeight: 'bold',
                              color: (opp.rawOutput as {predicted_class?: string})?.predicted_class === 'nutted' ? '#f44336' :
                                     (opp.rawOutput as {predicted_class?: string})?.predicted_class === 'air' ? '#4caf50' : '#ffc107'
                            }}>
                              {(opp.rawOutput as {predicted_class?: string})?.predicted_class || 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="placeholder">No opponent data</p>
                )}
              </div>

              {/* Policy Model Debug */}
              <div className="panel-section">
                <h4>Policy Model</h4>
                {/* Input Box - Policy model uses different features than Profit model */}
                <div className="debug-entry" style={{ marginBottom: '0.5rem' }}>
                  <div className="debug-header" style={{ color: '#8c93b8', fontSize: '0.7rem' }}>INPUT</div>
                  <div className="debug-row">
                    <span className="debug-label">hand_equity:</span>
                    <span className="debug-value">
                      {(((predictions.debugInfo.policyInput as {hand_equity?: number})?.hand_equity ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">starting_stack:</span>
                    <span className="debug-value">
                      {(predictions.debugInfo.policyInput as {starting_stack?: number})?.starting_stack?.toFixed(1) || 0} BB
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">position:</span>
                    <span className="debug-value">
                      {(predictions.debugInfo.policyInput as {position_from_button?: number})?.position_from_button ?? 'N/A'}
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">num_players:</span>
                    <span className="debug-value">
                      {(predictions.debugInfo.policyInput as {num_players?: number})?.num_players || 0}
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">opp_strength_mean:</span>
                    <span className="debug-value">
                      {(((predictions.debugInfo.policyInput as {opponent_strength_mean?: number})?.opponent_strength_mean ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">opp_nutted_count:</span>
                    <span className="debug-value" style={{
                      color: (predictions.debugInfo.policyInput as {opponent_nutted_count?: number})?.opponent_nutted_count ? '#f44336' : '#999'
                    }}>
                      {(predictions.debugInfo.policyInput as {opponent_nutted_count?: number})?.opponent_nutted_count || 0}
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">strength_adv:</span>
                    <span className="debug-value">
                      {(((predictions.debugInfo.policyInput as {strength_advantage?: number})?.strength_advantage ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                {/* Output Box */}
                <div className="debug-entry" style={{ background: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
                  <div className="debug-header" style={{ color: '#4caf50', fontSize: '0.7rem' }}>OUTPUT</div>
                  <div className="debug-row">
                    <span className="debug-label">predicted_action:</span>
                    <span className="debug-value" style={{ fontWeight: 'bold', fontSize: '0.85rem' }}>
                      {(predictions.debugInfo.policyRawOutput as {predicted_action?: string})?.predicted_action || 'N/A'}
                    </span>
                  </div>
                  {/* Action Probability Boxes */}
                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                    {(['fold', 'call', 'bet'] as const).map((action) => {
                      const probs = (predictions.debugInfo?.policyRawOutput as {probabilities?: Record<string, number>})?.probabilities || {}
                      const prob = probs[action] || 0
                      const isSelected = (predictions.debugInfo?.policyRawOutput as {predicted_action?: string})?.predicted_action === action
                      const bgColor = action === 'fold' ? 'rgba(255, 107, 107, 0.2)' :
                                     action === 'call' ? 'rgba(75, 212, 255, 0.2)' :
                                     'rgba(76, 175, 80, 0.2)'
                      const borderColor = action === 'fold' ? '#ff6b6b' :
                                         action === 'call' ? '#4bd4ff' :
                                         '#4caf50'
                      return (
                        <div
                          key={action}
                          style={{
                            flex: 1,
                            padding: '0.4rem',
                            borderRadius: '4px',
                            background: bgColor,
                            border: isSelected ? `2px solid ${borderColor}` : '1px solid rgba(140, 147, 184, 0.3)',
                            textAlign: 'center',
                          }}
                        >
                          <div style={{
                            fontSize: '0.65rem',
                            color: borderColor,
                            fontWeight: isSelected ? 'bold' : 'normal',
                            textTransform: 'uppercase',
                            marginBottom: '0.2rem'
                          }}>
                            {action}
                          </div>
                          <div style={{
                            fontSize: '0.9rem',
                            fontWeight: 'bold',
                            color: isSelected ? borderColor : '#ddd'
                          }}>
                            {(prob * 100).toFixed(0)}%
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>

              {/* Profit Model Debug */}
              <div className="panel-section">
                <h4>Profit Model</h4>
                {/* Input Box */}
                <div className="debug-entry" style={{ marginBottom: '0.5rem' }}>
                  <div className="debug-header" style={{ color: '#8c93b8', fontSize: '0.7rem' }}>INPUT</div>
                  <div className="debug-row">
                    <span className="debug-label">hand_equity:</span>
                    <span className="debug-value">
                      {(((predictions.debugInfo.profitInput as {hand_equity?: number})?.hand_equity ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">pot_size:</span>
                    <span className="debug-value">
                      {(predictions.debugInfo.profitInput as {pot_size?: number})?.pot_size || 0}
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">facing_call:</span>
                    <span className="debug-value">
                      {(predictions.debugInfo.profitInput as {facing_call?: number})?.facing_call || 0}
                    </span>
                  </div>
                  <div className="debug-row">
                    <span className="debug-label">opp_strength_mean:</span>
                    <span className="debug-value">
                      {(((predictions.debugInfo.profitInput as {opponent_strength_mean?: number})?.opponent_strength_mean ?? 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                {/* Output Box */}
                <div className="debug-entry" style={{ background: 'rgba(76, 175, 80, 0.1)', border: '1px solid rgba(76, 175, 80, 0.3)' }}>
                  <div className="debug-header" style={{ color: '#4caf50', fontSize: '0.7rem' }}>OUTPUT</div>
                  <div className="debug-row">
                    <span className="debug-label">predicted_profit_bb:</span>
                    <span className="debug-value" style={{
                      fontWeight: 'bold',
                      fontSize: '0.85rem',
                      color: ((predictions.debugInfo.profitRawOutput as {predicted_profit_bb?: number})?.predicted_profit_bb || 0) >= 0 ? '#4caf50' : '#f44336'
                    }}>
                      {((predictions.debugInfo.profitRawOutput as {predicted_profit_bb?: number})?.predicted_profit_bb || 0).toFixed(2)} BB
                    </span>
                  </div>
                </div>
              </div>
            </>
          ) : predictions ? (
            <div className="panel-section">
              <p className="placeholder">
                Debug info key: {predictions.debugInfo ? 'exists' : 'missing'},
                Type: {typeof predictions.debugInfo},
                Keys: {predictions.debugInfo ? Object.keys(predictions.debugInfo).join(', ') : 'N/A'}
              </p>
              <details open>
                <summary>Raw Response (debugInfo section)</summary>
                <pre style={{ fontSize: '10px', overflow: 'auto', maxHeight: '300px', background: '#1a1a2e', padding: '0.5rem', borderRadius: '4px' }}>
                  {JSON.stringify(predictions.debugInfo, null, 2)}
                </pre>
              </details>
              <details>
                <summary>Full Response</summary>
                <pre style={{ fontSize: '10px', overflow: 'auto', maxHeight: '200px' }}>
                  {JSON.stringify(predictions, null, 2)}
                </pre>
              </details>
            </div>
          ) : (
            <p className="placeholder">No predictions loaded. Make a prediction first.</p>
          )}
        </div>
      ) : viewMode === 'ai' ? (
        <div className="ai-advisor-view">
          {/* Risk Level Slider */}
          <div className="panel-section slider-section">
            <h4>Risk Level</h4>
            <div className="slider-container">
              <span className="slider-label-left">Low</span>
              <input
                type="range"
                min="0"
                max="100"
                value={riskLevel}
                onChange={(e) => setRiskLevel(Number(e.target.value))}
                className="advice-slider"
              />
              <span className="slider-label-right">High</span>
            </div>
            <div className="slider-value">{riskLevel}%</div>
          </div>

          {/* Play Style Slider */}
          <div className="panel-section slider-section">
            <h4>Play Style</h4>
            <div className="slider-container">
              <span className="slider-label-left">Standard</span>
              <input
                type="range"
                min="0"
                max="100"
                value={playStyle}
                onChange={(e) => setPlayStyle(Number(e.target.value))}
                className="advice-slider"
              />
              <span className="slider-label-right">Variable</span>
            </div>
            <div className="slider-value">{playStyle}%</div>
          </div>

          {/* AI Advice Display */}
          {aiLoading && (
            <div className="loading-indicator">
              <span className="spinner" />
              Consulting AI advisor...
            </div>
          )}

          {aiError && (
            <div className="panel-section error-section">
              <p className="error-message">{aiError}</p>
            </div>
          )}

          {!aiAdvice && !aiLoading && !aiError && (
            <div className="panel-section">
              <p className="placeholder">
                AI advice will appear automatically when it's your turn to act.
              </p>
            </div>
          )}

          {aiAdvice && !aiLoading && (
            <div className="panel-section advice-section">
              <h4>AI Recommendation</h4>
              <div className="advice-content">
                <p className="advice-text">{aiAdvice.advice}</p>
              </div>
              <div className="advice-meta">
                <span className="advice-model">Model: {aiAdvice.model}</span>
                <span className="advice-settings">
                  Risk: {aiAdvice.riskLevel}% | Style: {aiAdvice.playStyle}%
                </span>
              </div>
            </div>
          )}

          {/* View AI Prompt Button */}
          {aiAdvice?.promptUsed && (
            <div className="panel-section">
              <button
                className="show-prompt-btn"
                onClick={() => setShowPromptModal(true)}
              >
                View AI Prompt
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="info-view">
          <div className="panel-section">
            <h4>Hand Snapshot</h4>
            <ul>
              <li>Hand #{state.handId}</li>
              <li>Phase: {state.phase}</li>
              <li>Street: {state.streetName ?? '—'}</li>
              <li>Active Seat: {state.activeSeat ?? 'None'}</li>
            </ul>
          </div>
          <div className="panel-section">
            <h4>Players Remaining</h4>
            <ul className="player-list">
              {state.players.map((player) => (
                <li key={player.seatIndex} className={player.isFolded ? 'folded' : ''}>
                  Seat {player.seatIndex}: {player.name} ({player.stack})
                </li>
              ))}
            </ul>
          </div>
          <div className="toggle-controls">
            <h4>Training Tools</h4>
            <label className="toggle">
              <input type="checkbox" checked={showAllCards} onChange={(e) => onToggleShowAll(e.target.checked)} />
              <span>Show all players' cards</span>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={showStyles} onChange={(e) => onToggleStyles(e.target.checked)} />
              <span>Show bot play styles</span>
            </label>
            <label className="toggle">
              <input type="checkbox" checked={showAgentDebug} onChange={(e) => onToggleAgentDebug(e.target.checked)} />
              <span>Enable AI reasoning</span>
            </label>
          </div>
          {/* Bot Agent Prompts */}
          <div className="panel-section">
            <h4>Bot Agent Prompts</h4>
            <div className="bot-prompts-list">
              {state.players
                .filter(p => p.seatIndex !== state.heroSeat && p.agentDebug)
                .map((player) => (
                  <button
                    key={player.seatIndex}
                    className="bot-prompt-btn"
                    onClick={() => setSelectedBotPrompt({
                      name: player.name,
                      prompt: player.agentDebug || ''
                    })}
                  >
                    {player.name}
                  </button>
                ))}
              {state.players.filter(p => p.seatIndex !== state.heroSeat && p.agentDebug).length === 0 && (
                <p className="placeholder" style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
                  Enable "AI reasoning" above to see bot prompts
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Advisor Prompt Modal */}
      {showPromptModal && aiAdvice?.promptUsed && (
        <div className="modal-overlay" onClick={() => setShowPromptModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>AI Advisor Prompt</h3>
              <button className="modal-close" onClick={() => setShowPromptModal(false)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <pre className="prompt-text">{aiAdvice.promptUsed}</pre>
            </div>
          </div>
        </div>
      )}

      {/* Bot Agent Prompt Modal */}
      {selectedBotPrompt && (
        <div className="modal-overlay" onClick={() => setSelectedBotPrompt(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{selectedBotPrompt.name} - Agent Reasoning</h3>
              <button className="modal-close" onClick={() => setSelectedBotPrompt(null)}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <pre className="prompt-text">{selectedBotPrompt.prompt}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
