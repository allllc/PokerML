import type { PublicGameState } from '../types'
import './TrainingPanel.css'

type Props = {
  state: PublicGameState
  onToggleShowAll: (enabled: boolean) => Promise<void>
  showStyles: boolean
  onToggleStyles: (enabled: boolean) => void
  showAgentDebug: boolean
  onToggleAgentDebug: (enabled: boolean) => void
  onOpenAgentPanel: () => void
  hasAgentHistory: boolean
}

export function TrainingPanel({
  state,
  onToggleShowAll,
  showStyles,
  onToggleStyles,
  showAgentDebug,
  onToggleAgentDebug,
  onOpenAgentPanel,
  hasAgentHistory,
}: Props) {
  return (
    <div className="training-panel">
      <div className="panel-section">
        <h3>Training Tools</h3>
        <label className="toggle">
          <input
            type="checkbox"
            checked={state.showAllCards}
            onChange={(e) => onToggleShowAll(e.target.checked)}
          />
          <span>Show all players’ cards</span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showStyles} onChange={(e) => onToggleStyles(e.target.checked)} />
          <span>Show bot play styles</span>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={showAgentDebug} onChange={(e) => onToggleAgentDebug(e.target.checked)} />
          <span>Enable agent reasoning panel</span>
        </label>
        <button
          className="open-agent-panel"
          disabled={!showAgentDebug || !hasAgentHistory}
          onClick={onOpenAgentPanel}
        >
          View agent reasoning
        </button>
      </div>
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
              {showStyles && player.styleLabel && <div className="style-tag">{player.styleLabel}</div>}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
