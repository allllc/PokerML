import type { AgentHistoryEntry, PublicPlayerView } from '../types'
import './AgentDebugPanel.css'

type Props = {
  players: PublicPlayerView[]
  history: AgentHistoryEntry[]
  onClose: () => void
}

export function AgentDebugPanel({ players, history, onClose }: Props) {
  const seatMap = new Map(players.map((p) => [p.seatIndex, p.name]))
  const grouped = history.reduce<Record<number, AgentHistoryEntry[]>>((acc, entry) => {
    if (!acc[entry.seatIndex]) acc[entry.seatIndex] = []
    acc[entry.seatIndex].push(entry)
    return acc
  }, {})

  return (
    <div className="agent-panel-backdrop">
      <div className="agent-panel">
        <div className="agent-panel-header">
          <h3>Agent Reasoning</h3>
          <button onClick={onClose}>Close</button>
        </div>
        <div className="agent-panel-body">
          {Object.keys(grouped).length === 0 && <div>No agent data yet. Play a hand to capture reasoning.</div>}
          {Object.entries(grouped).map(([seatIndex, entries]) => (
            <div key={seatIndex} className="agent-entry">
              <div className="agent-entry-title">
                Seat {seatIndex}: {seatMap.get(Number(seatIndex))}
              </div>
              {entries.map((entry, idx) => (
                <div key={idx} className="agent-entry-block">
                  <div className="agent-entry-meta">
                    Street: {entry.streetName ?? entry.streetIndex} | Hand #{entry.handId}
                    {(() => {
                      const debug = entry.debug as Record<string, unknown> | undefined
                      const source = debug?.source
                      return source ? <span className="agent-entry-source">Source: {String(source)}</span> : null
                    })()}
                  </div>
                  <pre>{JSON.stringify(entry.debug, null, 2)}</pre>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
