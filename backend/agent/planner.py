"""
Planner agent for deciding the next action in the conversation.

The PlannerAgent:
- Looks at high-level user intent.
- Uses conversation memory.
- Inspects missing fields in the structured user profile.

It outputs **structured plans only** (Python dicts), never user-facing text.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional

from backend.memory.conversation_memory import ConversationMemory
from backend.memory.user_profile import UserProfileMemory


ActionType = Literal[
    "ASK_QUESTION",
    "COLLECT_PROFILE_FIELD",
    "RUN_ELIGIBILITY_CHECK",
    "RETRIEVE_SCHEMES",
    "END_CONVERSATION",
    "NO_OP",
]


@dataclass
class Plan:
    """
    Structured representation of the planner's decision.

    Attributes:
        action: High-level action type to perform next.
        field: Optional profile field that this plan is about
            (e.g., "income", "age") when asking for more information.
        metadata: Optional extra information required by the executor
            (for example, target language, flags, or scoring hints).
    """

    action: ActionType
    field: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the plan to a plain dictionary."""
        data = asdict(self)
        # Normalise None metadata to an empty dict for downstream consumers.
        if data.get("metadata") is None:
            data["metadata"] = {}
        return data


class PlannerAgent:
    """
    Planner agent that decides the next action in the flow.

    This agent does not generate natural language. It only produces
    structured plans describing what should happen next.
    """

    def __init__(self) -> None:
        """Initialise the PlannerAgent."""
        # Placeholder for future configuration (e.g., thresholds, strategies).
        self._profile_confidence_threshold: float = 0.6

    def plan(
        self,
        user_intent: str,
        conversation_memory: ConversationMemory,
        user_profile: UserProfileMemory,
    ) -> Dict[str, Any]:
        """
        Decide the next action based on intent, memory, and profile completeness.

        Args:
            user_intent: High-level intent label inferred elsewhere
                (e.g., "START", "COLLECT_PROFILE", "CHECK_ELIGIBILITY").
            conversation_memory: ConversationMemory instance containing
                the multi-turn dialogue so far.
            user_profile: UserProfileMemory instance with structured attributes.

        Returns:
            A dictionary representing the plan, e.g.:
                {
                    "action": "ASK_QUESTION",
                    "field": "income",
                    "metadata": {...}
                }
        """
        normalized_intent = (user_intent or "").strip().upper()
        missing_fields = user_profile.get_missing_fields()

        # 1. If we are in a profile collection phase and there are missing fields,
        #    ask about the next most important missing field.
        if normalized_intent in {"START", "COLLECT_PROFILE", "APPLY_SCHEME"} and missing_fields:
            next_field = self._prioritise_field(missing_fields)
            plan = Plan(
                action="ASK_QUESTION",
                field=next_field,
                metadata={
                    "reason": "missing_profile_field",
                    "all_missing_fields": missing_fields,
                },
            )
            return plan.to_dict()

        # 2. If user wants eligibility or schemes and the profile looks complete,
        #    proceed to eligibility evaluation.
        if normalized_intent in {"CHECK_ELIGIBILITY", "APPLY_SCHEME"} and not missing_fields:
            plan = Plan(
                action="RUN_ELIGIBILITY_CHECK",
                field=None,
                metadata={
                    "reason": "profile_complete",
                },
            )
            return plan.to_dict()

        # 3. If the user intent is to just see available schemes (without eligibility),
        #    we can trigger retrieval directly when some basic fields are present.
        if normalized_intent in {"BROWSE_SCHEMES", "DISCOVER"}:
            plan = Plan(
                action="RETRIEVE_SCHEMES",
                field=None,
                metadata={
                    "reason": "user_wants_discovery",
                    "missing_fields": missing_fields,
                },
            )
            return plan.to_dict()

        # 4. If the conversation context suggests we are done (very few recent turns
        #    and no clear intent), we may end the conversation.
        recent_context = conversation_memory.get_context(last_n_turns=3)
        if not recent_context:
            plan = Plan(
                action="NO_OP",
                field=None,
                metadata={"reason": "no_conversation_history"},
            )
            return plan.to_dict()

        # Default: do nothing specific, leave it to higher-level orchestration.
        plan = Plan(
            action="NO_OP",
            field=None,
            metadata={"reason": "no_specific_plan_for_intent", "intent": normalized_intent},
        )
        return plan.to_dict()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _prioritise_field(self, missing_fields: list[str]) -> str:
        """
        Select which profile field to ask for next.

        Simple heuristic:
        - Prioritise fields that are typically required for eligibility:
          income > age > category > state > gender
        - Fallback to the first missing field if none of the above match.
        """
        priority_order = ["income", "age", "category", "state", "gender"]
        lower_missing = [f.lower() for f in missing_fields]

        for field in priority_order:
            if field in lower_missing:
                return field

        # Fallback: return the first as given.
        return missing_fields[0] if missing_fields else ""


