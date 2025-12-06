"""
Agent modules for Aurora Poker App.

Contains bot agents (heuristic and OpenAI-powered) and AI advisor prompts.
"""

from .poker_advisor import (
    POKER_ADVISOR_SYSTEM_PROMPT,
    build_poker_advisor_prompt,
    PLAYER_NAMES,
    get_random_player_names,
    FEATURE_EXPLANATIONS,
    format_hand_history,
    analyze_board_texture,
    format_features_for_prompt,
)

from .bot_agents import (
    BotPolicy,
    HeuristicBotPolicy,
    OpenAIAgentPolicy,
    create_bot_policy,
    PERSONA_ROTATION,
    PERSONA_LABELS,
    PersonaKey,
)

__all__ = [
    # Poker Advisor
    "POKER_ADVISOR_SYSTEM_PROMPT",
    "build_poker_advisor_prompt",
    "PLAYER_NAMES",
    "get_random_player_names",
    "FEATURE_EXPLANATIONS",
    "format_hand_history",
    "analyze_board_texture",
    "format_features_for_prompt",
    # Bot Agents
    "BotPolicy",
    "HeuristicBotPolicy",
    "OpenAIAgentPolicy",
    "create_bot_policy",
    "PERSONA_ROTATION",
    "PERSONA_LABELS",
    "PersonaKey",
]
