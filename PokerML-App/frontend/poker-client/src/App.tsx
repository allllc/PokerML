import { useCallback, useEffect, useMemo, useState, useRef } from 'react'
import { act, advanceBots, createTable, startHand, toggleShowAll, getMLPredictions, getPredictionHistory } from './api'
import type { PublicGameState, MLPredictionsResponse, PredictionHistoryEntry } from './types'
import { SeatPanel } from './components/SeatPanel'
import { Board } from './components/Board'
import { ActionBar } from './components/ActionBar'
import { AgentDebugPanel } from './components/AgentDebugPanel'
import { MLPredictionsPanel } from './components/MLPredictionsPanel'
import './App.css'

const seatCoordinates = [
  { top: '6%', left: '50%' },
  { top: '22%', left: '85%' },
  { top: '70%', left: '85%' },
  { top: '92%', left: '50%' },
  { top: '70%', left: '15%' },
  { top: '22%', left: '15%' },
]

function App() {
  const [tableId, setTableId] = useState<string | null>(null)
  const [state, setState] = useState<PublicGameState | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showStyles, setShowStyles] = useState(false)
  const [showAgentDebug, setShowAgentDebug] = useState(false)
  const [agentPanelOpen, setAgentPanelOpen] = useState(false)
  const [isAdvancing, setIsAdvancing] = useState(false)

  // ML Predictions state
  const [mlPredictions, setMlPredictions] = useState<MLPredictionsResponse | null>(null)
  const [predictionHistory, setPredictionHistory] = useState<PredictionHistoryEntry[]>([])
  const [mlLoading, setMlLoading] = useState(false)

  // Track the last state for which we fetched predictions
  const lastPredictionKey = useRef<string | null>(null)

  const initialize = async () => {
    setLoading(true)
    setError(null)
    // Clear predictions when creating a new table (new session)
    setMlPredictions(null)
    setPredictionHistory([])
    lastPredictionKey.current = null
    try {
      const data = await createTable()
      setTableId(data.tableId)
      setState(data.state)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create table')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!state) {
      initialize()
    }
  }, [])

  const heroSeat = state?.heroSeat ?? 0

  const handleStartHand = async () => {
    if (!tableId) return
    setLoading(true)
    // Clear predictions for new hand
    setMlPredictions(null)
    lastPredictionKey.current = null
    try {
      const newState = await startHand(tableId)
      setState(newState)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start hand')
    } finally {
      setLoading(false)
    }
  }

  const handleAction = async (bet: number) => {
    if (!tableId || !state) return
    setLoading(true)
    try {
      const newState = await act(tableId, heroSeat, bet)
      setState(newState)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Action failed')
    } finally {
      setLoading(false)
    }
  }

  const handleToggleShowAll = async (enabled: boolean) => {
    if (!tableId) return
    setLoading(true)
    try {
      const newState = await toggleShowAll(tableId, enabled)
      setState(newState)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Toggle failed')
    } finally {
      setLoading(false)
    }
  }

  // Fetch ML predictions when state changes during betting
  const fetchPredictions = useCallback(async () => {
    if (!tableId || !state) return
    if (state.phase !== 'BETTING') return
    if (state.activeSeat !== state.heroSeat) return // Only predict when it's hero's turn

    // Create a key to track if we already fetched for this state
    // Include pot and callAmount to detect when opponent raises after hero acts
    const predictionKey = `${state.handId}-${state.streetName}-${state.activeSeat}-${state.pot}-${state.callAmount}`
    if (predictionKey === lastPredictionKey.current) return

    setMlLoading(true)
    try {
      const predictions = await getMLPredictions(tableId)
      setMlPredictions(predictions)
      lastPredictionKey.current = predictionKey

      // Also update history
      const history = await getPredictionHistory(tableId)
      setPredictionHistory(history.predictions)
    } catch (err) {
      console.error('Failed to fetch ML predictions:', err)
      // Don't show error to user - predictions are optional
    } finally {
      setMlLoading(false)
    }
  }, [tableId, state])

  // Trigger predictions when it's hero's turn or when betting situation changes
  useEffect(() => {
    if (state?.phase === 'BETTING' && state.activeSeat === state.heroSeat) {
      fetchPredictions()
    }
  }, [state?.phase, state?.activeSeat, state?.handId, state?.streetName, state?.pot, state?.callAmount, fetchPredictions])

  // Clear predictions when hand ends
  useEffect(() => {
    if (state?.phase === 'SHOWDOWN' || state?.phase === 'FINISHED') {
      // Keep predictions visible for review at showdown
      // Clear the key so new hand will fetch fresh predictions
      lastPredictionKey.current = null
    }
  }, [state?.phase])

  const tableSeats = useMemo(() => {
    if (!state) return []
    return state.players.map((player, idx) => {
      const coords = seatCoordinates[idx % seatCoordinates.length]
      const isWinner = state.winners?.includes(player.seatIndex) ?? false
      const winAmount = isWinner && state.payouts ? state.payouts[player.seatIndex] : undefined
      return (
        <SeatPanel
          key={player.seatIndex}
          player={player}
          isHero={player.seatIndex === state.heroSeat}
          isActive={player.seatIndex === state.activeSeat}
          positionStyle={coords}
          showStyle={showStyles}
          isWinner={isWinner}
          winAmount={winAmount}
        />
      )
    })
  }, [state, showStyles])

  const requestAdvance = useCallback(() => {
    if (!tableId || !state || isAdvancing || loading) return
    if (state.phase !== 'BETTING') return
    if (state.activeSeat === state.heroSeat) return
    const hero = state.players.find((p) => p.seatIndex === state.heroSeat)
    if (!hero || hero.isAllIn || hero.isFolded) return

    setIsAdvancing(true)
    advanceBots(tableId)
      .then((newState) => setState(newState))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to advance bots'))
      .finally(() => setIsAdvancing(false))
  }, [tableId, state, isAdvancing, loading])

  useEffect(() => {
    if (!tableId || !state) return
    if (loading || isAdvancing) return
    if (state.phase !== 'BETTING') return
    if (state.activeSeat === state.heroSeat) return
    const hero = state.players.find((p) => p.seatIndex === state.heroSeat)
    if (!hero || hero.isFolded || hero.isAllIn) return

    const timer = setTimeout(() => {
      requestAdvance()
    }, 500)
    return () => clearTimeout(timer)
  }, [tableId, state, loading, isAdvancing, requestAdvance])

  return (
    <div className="app-shell">
      <header>
        <div>
          <h1>PokerML</h1>
          <p>Single-table No-Limit Hold'em with bots</p>
        </div>
        <div className="header-actions">
          <button onClick={initialize}>Create New Table</button>
          <button onClick={handleStartHand}>Start Hand</button>
        </div>
      </header>

      <div className="banner-container">
        {error && <div className="banner error">{error}</div>}
        {loading && <div className="banner info">Working…</div>}
      </div>

      {!state ? (
        <div className="empty-state">
          <p>Click "Create New Table" to begin.</p>
        </div>
      ) : (
        <div className="table-layout">
          <div className="table-wrapper">
            <div className="table-felt">
              <Board cards={state.boardCards} pot={state.pot} />
              {tableSeats}
            </div>
            <ActionBar
              state={state}
              onAction={handleAction}
              onStartHand={handleStartHand}
              onAdvanceBots={requestAdvance}
              isAdvancing={isAdvancing}
            />
          </div>
          <MLPredictionsPanel
            state={state}
            predictions={mlPredictions}
            predictionHistory={predictionHistory}
            isLoading={mlLoading}
            showAllCards={state.showAllCards}
            onToggleShowAll={handleToggleShowAll}
            showStyles={showStyles}
            onToggleStyles={setShowStyles}
            showAgentDebug={showAgentDebug}
            onToggleAgentDebug={setShowAgentDebug}
          />
        </div>
      )}
      {agentPanelOpen && state && (
        <AgentDebugPanel players={state.players} history={state.agentHistory} onClose={() => setAgentPanelOpen(false)} />
      )}
    </div>
  )
}

export default App
