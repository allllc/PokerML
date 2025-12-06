import type { Card, PotState } from '../types'
import './Board.css'

type Props = {
  cards: Card[]
  pot: PotState
}

export function Board({ cards, pot }: Props) {
  const getSuitClass = (suit: string) => {
    if (suit === '♠') return 'suit-spades'
    if (suit === '♥') return 'suit-hearts'
    if (suit === '♦') return 'suit-diamonds'
    if (suit === '♣') return 'suit-clubs'
    return ''
  }

  return (
    <div className="board">
      <div className="community-cards">
        {cards.length === 0 && <span className="placeholder">Board pending…</span>}
        {cards.map((card, index) => (
          <div
            key={card.id}
            className="card face-up"
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <span className="card-rank">{card.rank}</span>
            <span className={`card-suit ${getSuitClass(card.suit)}`}>{card.suit}</span>
          </div>
        ))}
      </div>
      <div className="pot-info">
        <div className="pot-main">Pot: {pot.mainPot}</div>
        {pot.sidePots.length > 0 && (
          <div className="side-pots">
            {pot.sidePots.map((sp, idx) => (
              <span key={idx} className="side-pot-badge">
                Side {idx + 1}: {sp.amount}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
