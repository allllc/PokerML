"""
Poker Advisor AI Prompts.

Contains prompt templates for the AI poker advisor that uses ML predictions
to provide strategic advice based on the player's risk tolerance and play style.
"""

from typing import Dict, Any, List
import random

# 100 neutral player names that don't indicate playing style
PLAYER_NAMES: List[str] = [
    "Alex", "Jordan", "Casey", "Morgan", "Taylor", "Riley", "Quinn", "Avery", "Blake", "Cameron",
    "Dakota", "Drew", "Emerson", "Finley", "Harper", "Hayden", "Jamie", "Jesse", "Kendall", "Lane",
    "Logan", "Marley", "Parker", "Peyton", "Reagan", "Reese", "River", "Rowan", "Sage", "Sam",
    "Sawyer", "Skyler", "Spencer", "Sydney", "Tatum", "Charlie", "Frankie", "Jackie", "Kerry", "Kim",
    "Lee", "Leslie", "Lynn", "Marion", "Pat", "Robin", "Sandy", "Shannon", "Terry", "Tracy",
    "Adrian", "Angel", "Arden", "Aubrey", "Bailey", "Brett", "Brook", "Carmen", "Carroll", "Cody",
    "Corey", "Dana", "Daryl", "Devon", "Dominique", "Ellis", "Gene", "Glenn", "Gray", "Jade",
    "Jean", "Jody", "Jules", "Justice", "Kelly", "Lake", "Laurie", "Loren", "Mackenzie", "Micah",
    "Milan", "Monroe", "Murphy", "Noel", "Ocean", "Phoenix", "Presley", "Raven", "Remy", "Rory",
    "Scout", "Shawn", "Shelby", "Storm", "Sunny", "Timber", "Tristan", "Winter", "Wren", "Zion",
]


def get_random_player_names(count: int = 5, exclude: List[str] = None) -> List[str]:
    """Get random player names from the pool."""
    exclude = exclude or []
    available = [n for n in PLAYER_NAMES if n not in exclude]
    return random.sample(available, min(count, len(available)))


POKER_ADVISOR_SYSTEM_PROMPT = """You are an expert poker advisor providing strategic recommendations. You receive comprehensive game data including ML model predictions, raw features, hand history, and board texture analysis.

IMPORTANT GUIDELINES:

1. USE ML PREDICTIONS AS ONE INPUT, NOT THE ONLY INPUT
   - The ML models provide predictions but they are not infallible
   - Consider the predictions alongside your own analysis of the cards, board texture, and betting patterns
   - If the predictions conflict with obvious strategic logic (e.g., folding the nuts), use your judgment

2. ANALYZE THE FULL PICTURE
   - Your hole cards and their strength relative to the board
   - Board texture: flush draws, straight draws, paired boards, high cards
   - Betting patterns: who has been aggressive, who is passive, bet sizing tells
   - Position: early position requires stronger hands, late position allows more flexibility
   - Stack depths: SPR (stack-to-pot ratio) affects commitment decisions

3. PROVIDE ACTIONABLE ADVICE
   - Recommend a specific action: fold, check, call, or raise
   - For raises, specify the size in Big Blinds (BB)
   - Consider pot odds and implied odds
   - Factor in the player's risk tolerance and play style preferences

4. RESPONSE FORMAT
   Provide a concise 2-4 sentence recommendation that:
   - States your recommended action with bet sizing in BB if applicable
   - Explains the key reasoning (cards, board, opponents, predictions)
   - Notes any important considerations or alternative lines

Do not use bullet points or lists. Write in a conversational but professional tone."""


# Feature explanations for the prompt
FEATURE_EXPLANATIONS = {
    # Position and context
    "street_rank": "Street (0=preflop, 1=flop, 2=turn, 3=river)",
    "position_bucket": "Table position (early/middle/late/blinds/button)",
    "players_active": "Players still in the hand",
    "opponents_active": "Number of opponents remaining",

    # Stack and pot metrics
    "pot_size": "Current pot size in chips",
    "pot_contribution": "Your total chips committed this hand",
    "starting_stack": "Your stack at start of hand",
    "bb": "Big blind size",
    "facing_call": "Amount you need to call",
    "pot_odds_call": "Pot odds for calling (bet / (pot + bet))",

    # SPR (Stack-to-Pot Ratio)
    "spr": "Stack-to-Pot Ratio (stack / pot) - low SPR means committed",
    "spr_low": "SPR < 3 (committed, call/fold decisions)",
    "spr_medium": "SPR 3-7 (moderate commitment level)",
    "spr_high": "SPR 7-15 (room to maneuver)",
    "spr_very_deep": "SPR > 15 (deep stacked, can play speculative hands)",

    # Hand strength
    "hand_strength": "Your hand strength (0-1 scale, higher = stronger)",
    "strength_advantage": "Your strength vs avg opponent strength",

    # Board texture
    "board_monotone": "All board cards same suit (flush present/possible)",
    "board_paired": "Board has a pair (full house/quads possible)",
    "board_straighty": "Board has connected cards (straight possible)",
    "board_flush_pressure": "3+ cards of same suit on board",
    "board_straight_pressure": "Connected board enabling straights",

    # Opponent predictions from ML
    "opponent_strength_mean": "Average predicted opponent strength",
    "opponent_strength_max": "Strongest predicted opponent",
    "opponent_air_count": "Opponents predicted to have weak hands",
    "opponent_middle_count": "Opponents predicted to have medium hands",
    "opponent_nutted_count": "Opponents predicted to have strong hands",

    # Raise sizing info
    "can_afford_raise_33": "Can afford 33% pot raise",
    "can_afford_raise_75": "Can afford 75% pot raise",
    "can_afford_raise_100": "Can afford pot-sized raise",

    # Historical patterns
    "vpip_last10_hist": "Your voluntary play rate (higher = looser)",
    "pfr_last10_hist": "Your preflop raise rate (higher = more aggressive)",
    "agg_factor_last10_hist": "Your aggression factor (bets+raises / calls)",
}


def format_hand_history(hand_history: List[Dict[str, Any]]) -> str:
    """Format hand history into readable text."""
    if not hand_history:
        return "No actions recorded yet this hand."

    lines = []
    current_street = None

    for entry in hand_history:
        street = entry.get("streetName", "Unknown")
        player = entry.get("playerName", f"Seat {entry.get('seatIndex', '?')}")
        action = entry.get("action", "unknown")
        amount = entry.get("amount", 0)

        # Add street header if changed
        if street != current_street:
            if lines:
                lines.append("")  # Blank line between streets
            lines.append(f"[{street}]")
            current_street = street

        # Format action
        if action in ("bet", "raise", "bet_or_raise_to"):
            action_str = f"{player} raises to {amount}"
        elif action in ("call", "call_or_check"):
            if amount > 0:
                action_str = f"{player} calls {amount}"
            else:
                action_str = f"{player} checks"
        elif action == "fold":
            action_str = f"{player} folds"
        else:
            action_str = f"{player} {action} {amount}" if amount else f"{player} {action}"

        lines.append(action_str)

    return "\n".join(lines)


def analyze_board_texture(board_cards: List[str]) -> Dict[str, Any]:
    """Analyze board texture for draws and made hands."""
    if not board_cards:
        return {"texture": "preflop", "description": "No community cards yet."}

    # Parse cards
    ranks = []
    suits = []
    rank_values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                   "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}

    for card in board_cards:
        if len(card) >= 2:
            rank = card[0].upper()
            suit = card[1].lower()
            ranks.append(rank)
            suits.append(suit)

    # Suit analysis
    suit_counts = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    max_suit = max(suit_counts.values()) if suit_counts else 0

    flush_draw = max_suit == 3
    flush_complete = max_suit >= 4
    monotone = max_suit == len(board_cards) and len(board_cards) >= 3

    # Rank analysis
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    paired = any(c >= 2 for c in rank_counts.values())
    trips_on_board = any(c >= 3 for c in rank_counts.values())

    # Connectedness
    values = sorted(set(rank_values.get(r, 0) for r in ranks))
    if 14 in values:  # Add ace-low for wheel
        values = sorted(set(values) | {1})

    connected = False
    for i in range(len(values) - 2):
        if values[i + 2] - values[i] <= 4:
            connected = True
            break

    # High cards
    high_cards = sum(1 for r in ranks if rank_values.get(r, 0) >= 10)

    # Build description
    descriptions = []
    board_str = " ".join(board_cards)

    if monotone:
        descriptions.append("monotone (flush possible)")
    elif flush_complete:
        descriptions.append("4-flush on board")
    elif flush_draw:
        descriptions.append("flush draw possible")

    if trips_on_board:
        descriptions.append("trips on board")
    elif paired:
        descriptions.append("paired board")

    if connected:
        descriptions.append("connected/straighty")

    if high_cards >= 2:
        descriptions.append("broadway-heavy")

    texture = "wet" if (flush_draw or connected) else "dry"
    if paired:
        texture = "paired " + texture

    return {
        "board": board_str,
        "texture": texture,
        "description": ", ".join(descriptions) if descriptions else "dry board",
        "flush_possible": flush_draw or flush_complete,
        "straight_possible": connected,
        "paired": paired,
        "high_cards": high_cards,
    }


def format_features_for_prompt(
    raw_features: Dict[str, Any],
    include_all: bool = False
) -> str:
    """Format raw ML features into human-readable text with explanations."""
    if not raw_features:
        return "No feature data available."

    # Key features to always include
    key_features = [
        "pot_size", "facing_call", "pot_odds_call", "bb",
        "hand_strength", "strength_advantage",
        "spr", "opponents_active",
        "opponent_strength_mean", "opponent_air_count", "opponent_middle_count", "opponent_nutted_count",
        "board_flush_pressure", "board_straight_pressure", "board_paired",
    ]

    lines = []

    for key in key_features:
        if key in raw_features:
            value = raw_features[key]
            explanation = FEATURE_EXPLANATIONS.get(key, key)

            # Format value nicely
            if isinstance(value, float):
                if abs(value) < 0.01:
                    formatted_value = "0"
                elif abs(value) < 1:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.1f}"
            else:
                formatted_value = str(value)

            lines.append(f"- {key}: {formatted_value} ({explanation})")

    return "\n".join(lines)


def build_poker_advisor_prompt(
    game_state: Dict[str, Any],
    ml_predictions: Dict[str, Any],
    risk_level: int,  # 0-100
    play_style: int,  # 0-100
    raw_features: Dict[str, Any] = None,
    hand_history: List[Dict[str, Any]] = None,
) -> str:
    """
    Build the comprehensive user prompt for the poker advisor.

    Args:
        game_state: Current game state information
        ml_predictions: ML model predictions
        risk_level: 0 (conservative) to 100 (aggressive)
        play_style: 0 (by the book) to 100 (variable/exploitative)
        raw_features: Raw feature dictionary from feature engine
        hand_history: List of actions taken this hand

    Returns:
        Formatted prompt string
    """
    # === SECTION 1: YOUR CARDS AND THE BOARD ===
    hero_cards = game_state.get('heroCards', 'unknown')
    board_cards_raw = game_state.get('boardCards', [])
    if isinstance(board_cards_raw, str):
        board_cards = board_cards_raw.split() if board_cards_raw else []
    else:
        board_cards = board_cards_raw or []

    board_analysis = analyze_board_texture(board_cards)

    cards_section = f"""YOUR HAND: {hero_cards}
BOARD: {board_analysis['board'] if board_cards else 'No community cards (preflop)'}
BOARD TEXTURE: {board_analysis['description']}"""

    # === SECTION 2: GAME SITUATION ===
    street = game_state.get('street', 'unknown')
    pot_size = game_state.get('potSize', 0)
    facing_bet = game_state.get('facingBet', 0)
    hero_stack = game_state.get('heroStack', 0)
    position = game_state.get('position', 'unknown')
    bb = game_state.get('bb', 10)  # Default BB

    # Calculate values in BB
    pot_bb = pot_size / bb if bb else pot_size
    facing_bb = facing_bet / bb if bb else facing_bet
    stack_bb = hero_stack / bb if bb else hero_stack

    situation_section = f"""STREET: {street}
POT: {pot_size} chips ({pot_bb:.1f} BB)
FACING BET: {facing_bet} chips ({facing_bb:.1f} BB)
YOUR STACK: {hero_stack} chips ({stack_bb:.1f} BB)
POSITION: {position}"""

    # === SECTION 3: HAND HISTORY ===
    history_section = "BETTING ACTION THIS HAND:\n"
    if hand_history:
        history_section += format_hand_history(hand_history)
    else:
        history_section += "No actions recorded yet."

    # === SECTION 4: RAW FEATURES FROM ML PIPELINE ===
    features_section = "KEY FEATURES (from ML pipeline):\n"
    if raw_features:
        features_section += format_features_for_prompt(raw_features)
    else:
        features_section += "No raw features available."

    # === SECTION 5: ML MODEL PREDICTIONS ===
    # Format opponent predictions
    opponent_summary = ""
    if ml_predictions.get("opponentPredictions"):
        opp_strs = []
        for opp in ml_predictions["opponentPredictions"]:
            strength = opp.get("predictedClass", "unknown")
            name = opp.get("playerName", "Opponent")
            confidence = opp.get("confidence", 0)
            opp_strs.append(f"{name}: {strength} ({confidence:.0%})")
        opponent_summary = "\n  ".join(opp_strs)
    else:
        opponent_summary = "No opponent predictions available"

    # Format profit prediction
    profit_info = ""
    if ml_predictions.get("profitPrediction"):
        profit_bb = ml_predictions["profitPrediction"].get("predictedProfitBB", 0)
        interpretation = ml_predictions["profitPrediction"].get("interpretation", "")
        from_fallback = ml_predictions["profitPrediction"].get("fromFallback", False)
        profit_info = f"{profit_bb:+.2f} BB ({interpretation})"
        if from_fallback:
            profit_info += " [fallback estimate]"
    else:
        profit_info = "Not available"

    # Format action recommendation
    action_info = ""
    if ml_predictions.get("actionRecommendation"):
        action = ml_predictions["actionRecommendation"].get("action", "unknown")
        chip_amount = ml_predictions["actionRecommendation"].get("chipAmount", 0)
        confidence = ml_predictions["actionRecommendation"].get("confidence", 0)
        from_fallback = ml_predictions["actionRecommendation"].get("fromFallback", False)

        action_bb = chip_amount / bb if bb and chip_amount else 0
        action_info = f"{action}"
        if chip_amount > 0:
            action_info += f" to {chip_amount} chips ({action_bb:.1f} BB)"
        action_info += f" (confidence: {confidence:.0%})"
        if from_fallback:
            action_info += " [fallback]"
    else:
        action_info = "Not available"

    predictions_section = f"""ML MODEL PREDICTIONS:
(Use these as guidance, not absolute rules. Evaluate against the actual cards and situation.)

Opponent Hand Strengths:
  {opponent_summary}

Expected Profit: {profit_info}

Model's Suggested Action: {action_info}"""

    # === SECTION 6: PLAYER PREFERENCES ===
    risk_desc = "very conservative" if risk_level < 20 else \
                "conservative" if risk_level < 40 else \
                "balanced" if risk_level < 60 else \
                "aggressive" if risk_level < 80 else "very aggressive"

    style_desc = "strict GTO/textbook" if play_style < 20 else \
                 "mostly by the book" if play_style < 40 else \
                 "balanced approach" if play_style < 60 else \
                 "exploitative" if play_style < 80 else "highly variable/creative"

    preferences_section = f"""PLAYER PREFERENCES:
Risk Level: {risk_level}/100 ({risk_desc})
Play Style: {play_style}/100 ({style_desc})"""

    # === SECTION 7: SIZING CONTEXT ===
    min_raise = game_state.get('minRaise', 0)
    max_raise = game_state.get('maxRaise', hero_stack)

    min_raise_bb = min_raise / bb if bb else min_raise
    max_raise_bb = max_raise / bb if bb else max_raise

    # Common bet sizes in BB
    half_pot_bb = (pot_size * 0.5) / bb if bb else pot_size * 0.5
    two_thirds_pot_bb = (pot_size * 0.67) / bb if bb else pot_size * 0.67
    pot_bb_size = pot_size / bb if bb else pot_size

    sizing_section = f"""BET SIZING REFERENCE (in Big Blinds):
- 1 BB = {bb} chips
- Minimum raise: {min_raise_bb:.1f} BB
- Half pot: {half_pot_bb:.1f} BB
- 2/3 pot: {two_thirds_pot_bb:.1f} BB
- Full pot: {pot_bb_size:.1f} BB
- Maximum (all-in): {max_raise_bb:.1f} BB"""

    # === COMBINE ALL SECTIONS ===
    prompt = f"""{cards_section}

{situation_section}

{history_section}

{features_section}

{predictions_section}

{preferences_section}

{sizing_section}

QUESTION: Based on all this information, what action should I take? If raising, specify the size in BB."""

    return prompt
