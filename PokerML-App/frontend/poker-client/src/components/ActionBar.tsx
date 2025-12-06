import { useMemo, useState } from 'react'
import type { PublicGameState } from '../types'
import './ActionBar.css'

type Props = {
  state: PublicGameState
  onAction: (bet: number) => Promise<void>
  onStartHand: () => Promise<void>
  onAdvanceBots: () => void
  isAdvancing: boolean
}

export function ActionBar({ state, onAction, onStartHand, onAdvanceBots, isAdvancing }: Props) {
  const [raiseAmount, setRaiseAmount] = useState<number | null>(null)
  const heroActions = useMemo(() => state.legalActions ?? [], [state])

  const callAction = heroActions.find((a) => a.type === 'CALL')
  const raiseAction = heroActions.find((a) => a.type === 'RAISE')
  const canCheck = heroActions.some((a) => a.type === 'CHECK')
  const canFold = heroActions.some((a) => a.type === 'FOLD')

  const handleRaise = async () => {
    if (!raiseAction) return
    const chosen = raiseAmount ?? raiseAction.min
    await onAction(chosen)
  }

  if (state.phase === 'SHOWDOWN' || state.phase === 'FINISHED') {
    return (
      <div className="action-bar">
        <div className="action-status">Hand complete</div>
        <button onClick={onStartHand} className="primary">
          Start Next Hand
        </button>
      </div>
    )
  }

  if (state.activeSeat !== state.heroSeat) {
    return (
      <div className="action-bar waiting">
        <div className="action-status">Waiting for opponents...</div>
        <button onClick={onAdvanceBots} disabled={isAdvancing} className="ghost">
          {isAdvancing ? 'Advancing…' : 'Advance Opponents'}
        </button>
      </div>
    )
  }

  return (
    <div className="action-bar">
      <div className="action-status">Your action — {state.streetName ?? 'Pre-Hand'}</div>
      <div className="action-buttons">
        {/* Check/Fold button - enabled when can check OR can fold */}
        <button disabled={!canCheck && !canFold} onClick={() => onAction(0)}>
          {canCheck ? 'Check' : 'Fold'}
        </button>
        {/* Call button - only show when there's a call action */}
        {callAction && (
          <button onClick={() => onAction(callAction.amount)}>
            Call {callAction.amount}
          </button>
        )}
        <div className="raise-group">
          <button disabled={!raiseAction} onClick={handleRaise}>
            Raise
          </button>
          {raiseAction && (
            <div className="raise-controls">
              <div className="raise-input-group">
                <button
                  className="raise-increment"
                  onClick={() => {
                    const current = raiseAmount ?? raiseAction.min
                    const newAmount = Math.max(raiseAction.min, current - 2)
                    setRaiseAmount(newAmount)
                  }}
                >
                  -2
                </button>
                <input
                  type="number"
                  className="raise-number-input"
                  min={raiseAction.min}
                  max={raiseAction.max}
                  value={raiseAmount ?? raiseAction.min}
                  onChange={(e) => {
                    const val = Number(e.target.value)
                    if (!isNaN(val)) {
                      setRaiseAmount(Math.min(raiseAction.max, Math.max(raiseAction.min, val)))
                    }
                  }}
                />
                <button
                  className="raise-increment"
                  onClick={() => {
                    const current = raiseAmount ?? raiseAction.min
                    const newAmount = Math.min(raiseAction.max, current + 2)
                    setRaiseAmount(newAmount)
                  }}
                >
                  +2
                </button>
              </div>
              <input
                type="range"
                className="raise-slider"
                min={raiseAction.min}
                max={raiseAction.max}
                value={raiseAmount ?? raiseAction.min}
                onChange={(e) => setRaiseAmount(Number(e.target.value))}
              />
              <div className="raise-values">
                Min: {raiseAction.min} | Max: {raiseAction.max}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
