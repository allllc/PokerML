import type { PublicPlayerView } from '../types'
import './SeatPanel.css'

type Props = {
  player: PublicPlayerView
  isHero: boolean
  isActive: boolean
  positionStyle: React.CSSProperties
  showStyle: boolean
  isWinner?: boolean
  winAmount?: number
}

export function SeatPanel({ player, isHero, isActive, positionStyle, showStyle, isWinner, winAmount }: Props) {
  const getSuitClass = (suit: string) => {
    if (suit === '♠') return 'suit-spades'
    if (suit === '♥') return 'suit-hearts'
    if (suit === '♦') return 'suit-diamonds'
    if (suit === '♣') return 'suit-clubs'
    return ''
  }

  return (
    <div
      className={`seat-panel ${isHero ? 'hero' : ''} ${isActive ? 'active' : ''} ${
        player.isFolded ? 'folded' : ''
      } ${player.isAllIn ? 'all-in' : ''} ${isWinner ? 'winner' : ''}`}
      style={positionStyle}
    >
      {isWinner && winAmount && (
        <div className="win-badge">+{winAmount}</div>
      )}
      <div className="seat-name">
        {player.name}
        {player.hasButton && <span className="button-chip">B</span>}
      </div>
      <div className="seat-stack">Stack: {player.stack}</div>
      {showStyle && player.styleLabel && <div className="seat-style">{player.styleLabel}</div>}
      <div className={`seat-bet ${player.betThisStreet > 0 ? 'has-bet' : ''}`}>
        {player.betThisStreet > 0 ? `Bet: ${player.betThisStreet}` : player.totalCommitted > 0 ? `Committed: ${player.totalCommitted}` : 'Waiting'}
      </div>
      <div className="seat-cards">
        {player.holeCards && player.holeCards.length > 0 ? (
          player.holeCards.map((card) => (
            <div key={card.id} className="card face-up">
              <span>{card.rank}</span>
              <span className={getSuitClass(card.suit)}>{card.suit}</span>
            </div>
          ))
        ) : (
          <>
            <div className="card back" />
            <div className="card back" />
          </>
        )}
      </div>
    </div>
  )
}
