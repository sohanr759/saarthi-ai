"""
Evaluator agent for checking whether the agent response satisfies the user goal.

Responsibilities:
- Inspect internal state (conversation, profile, tool confidences) to decide
  if the last step was successful.
- Detect failure modes:
  - Missing information (e.g., required profile fields not collected).
  - Unresolved contradictions in user facts.
  - Low-confidence outputs from tools (e.g., STT).

Output (internal, non user-facing):
- status: one of {"SUCCESS", "RETRY", "FAIL"}
- feedback: optional machine-readable feedback for replanning.

This module does NOT generate any user-visible natural language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from backend.memory.conversation_memory import ConversationMemory
from backend.memory.user_profile import UserProfileMemory


EvaluationStatus = Literal["SUCCESS", "RETRY", "FAIL"]


@dataclass
class EvaluationResult:
    """
    Structured result of an evaluation.

    Attributes:
        status: Overall evaluation outcome (SUCCESS / RETRY / FAIL).
        feedback: Optional machine-readable hints for the planner, such as:
            - missing_fields: list of profile fields that must be collected.
            - unresolved_contradictions: list of fact keys with conflicts.
            - low_confidence: boolean flags for low-confidence operations.
            - notes: short, internal-only explanation strings.
    """

    status: EvaluationStatus
    feedback: Dict[str, Any]


class EvaluatorAgent:
    """
    EvaluatorAgent inspects the current state and decides if the last step
    moved the conversation closer to the user's goal.

    It does not generate user-facing text; instead it returns structured
    signals for the planner/executor loop.
    """

    def __init__(
        self,
        conversation_memory: ConversationMemory,
        user_profile: UserProfileMemory,
        stt_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialise the evaluator with memory modules and thresholds.

        Args:
            conversation_memory: ConversationMemory instance.
            user_profile: UserProfileMemory instance.
            stt_confidence_threshold: Minimum acceptable STT confidence.
        """
        self._conversation_memory = conversation_memory
        self._user_profile = user_profile
        self._stt_conf_threshold = max(0.0, min(1.0, float(stt_confidence_threshold)))

        # Optional cache for latest tool signals (e.g., from executor).
        self._last_stt_confidence: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_stt_confidence(self, confidence: Optional[float]) -> None:
        """
        Record the latest STT confidence for evaluation purposes.
        """
        if confidence is None:
            self._last_stt_confidence = None
        else:
            self._last_stt_confidence = max(0.0, min(1.0, float(confidence)))

    def evaluate_step(
        self,
        user_goal: str,
        last_action: str,
        last_agent_response_internal_tag: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate whether the last agent step satisfies or advances the user goal.

        Args:
            user_goal: High-level goal label (e.g., "CHECK_ELIGIBILITY",
                "COLLECT_PROFILE", "DISCOVER_SCHEMES").
            last_action: Planner action that was executed
                (e.g., "ASK_QUESTION", "RUN_ELIGIBILITY_CHECK").
            last_agent_response_internal_tag: Optional internal tag describing
                the response template used, if the executor provides it.

        Returns:
            EvaluationResult with status and feedback.
        """
        goal = (user_goal or "").strip().upper()
        action = (last_action or "").strip().upper()

        feedback: Dict[str, Any] = {
            "goal": goal,
            "last_action": action,
            "issues": [],
        }

        # 1. Check for missing profile information when profile is needed.
        missing_fields = self._user_profile.get_missing_fields()
        feedback["missing_fields"] = missing_fields

        # 2. Check for contradictions in conversation memory facts.
        contradictions = self._detect_unresolved_contradictions()
        feedback["unresolved_contradictions"] = contradictions

        # 3. Check low-confidence STT (if any).
        low_confidence = self._is_low_confidence_stt()
        feedback["low_confidence_stt"] = low_confidence

        # Aggregate issues for planner.
        if missing_fields:
            feedback["issues"].append("missing_profile_fields")
        if contradictions:
            feedback["issues"].append("unresolved_contradictions")
        if low_confidence:
            feedback["issues"].append("low_confidence_stt")

        # Determine overall status based on goal and issues.
        status = self._decide_status(goal, action, feedback)

        return EvaluationResult(status=status, feedback=feedback)

    # ------------------------------------------------------------------ #
    # Internal checks
    # ------------------------------------------------------------------ #

    def _detect_unresolved_contradictions(self) -> list[str]:
        """
        Detect if the conversation memory contains conflicting user facts.

        Approach:
        - Compare the latest user message facts with stored facts using
          ConversationMemory.detect_contradiction.
        - If any contradiction is found, return the list of affected keys.
        """
        if not self._conversation_memory.messages:
            return []

        # Look at the last user message (if any) and check for contradictions
        # between its facts and the global fact store.
        last_user_facts = {}
        for msg in reversed(self._conversation_memory.messages):
            if msg.role == "user" and msg.facts:
                last_user_facts = msg.facts
                break

        if not last_user_facts:
            return []

        conflicted_keys: list[str] = []
        for key, value in last_user_facts.items():
            result = self._conversation_memory.detect_contradiction({key: value})
            if result is not None:
                conflicted_keys.append(key)

        return conflicted_keys

    def _is_low_confidence_stt(self) -> bool:
        """
        Determine if the most recent STT output was low-confidence.
        """
        if self._last_stt_confidence is None:
            return False
        return self._last_stt_confidence < self._stt_conf_threshold

    def _decide_status(
        self,
        goal: str,
        action: str,
        feedback: Dict[str, Any],
    ) -> EvaluationStatus:
        """
        Combine goal, last action, and detected issues into a status.
        """
        issues = feedback.get("issues") or []
        missing_fields = feedback.get("missing_fields") or []

        # If we are trying to check eligibility but profile is incomplete,
        # we should RETRY with more information.
        if goal in {"CHECK_ELIGIBILITY", "APPLY_SCHEME"} and missing_fields:
            return "RETRY"

        # If contradictions are present, planner should resolve them.
        if "unresolved_contradictions" in issues:
            return "RETRY"

        # Low STT confidence suggests the user input may be unreliable.
        if "low_confidence_stt" in issues and action in {"ASK_QUESTION", "COLLECT_PROFILE_FIELD"}:
            return "RETRY"

        # If no issues remain and we have just run the main operation for the goal,
        # consider this a success.
        if not issues and goal in {"CHECK_ELIGIBILITY", "DISCOVER_SCHEMES", "BROWSE_SCHEMES"}:
            if action in {"RUN_ELIGIBILITY_CHECK", "RETRIEVE_SCHEMES"}:
                return "SUCCESS"

        # If there are issues that cannot be resolved by retry (e.g., structural
        # problems in configuration), they would be marked as FAIL. For now, we
        # conservatively default to RETRY when issues exist, otherwise SUCCESS.
        if issues:
            return "RETRY"

        return "SUCCESS"


