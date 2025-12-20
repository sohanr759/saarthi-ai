"""
Main agent loop wiring Planner, Executor, and Evaluator together.

This module demonstrates agentic behaviour:
- Planner decides the next action.
- Executor performs the action and interacts with tools/memory.
- Evaluator checks whether the step advanced or satisfied the user goal.

The loop:
- Repeats Planner → Executor → Evaluator.
- Stops when Evaluator returns SUCCESS or when max retries is reached.
- Logs plans, tool calls, and evaluation results for observability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.agent.planner import PlannerAgent
from backend.agent.executor import ExecutorAgent
from backend.agent.evaluator import EvaluatorAgent, EvaluationResult
from backend.memory.conversation_memory import ConversationMemory
from backend.memory.user_profile import UserProfileMemory
from backend.tools.eligibility_engine import EligibilityEngine
from backend.tools.scheme_retriever import SchemeRetriever
from backend.voice.stt import SpeechToText


logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    High-level orchestrator for the agent loop.

    This class wires together:
    - PlannerAgent
    - ExecutorAgent
    - EvaluatorAgent
    - Memory modules and tools
    """

    def __init__(
        self,
        stt_model: SpeechToText,
        native_language: str = "hi",
        max_retries: int = 5,
    ) -> None:
        """
        Initialise the orchestrator and underlying components.

        Args:
            stt_model: SpeechToText instance for audio transcription.
            native_language: ISO code for native language (used by ExecutorAgent).
            max_retries: Maximum evaluation retries before giving up.
        """
        self.conversation_memory = ConversationMemory()
        self.user_profile = UserProfileMemory()

        self.scheme_retriever = SchemeRetriever()
        self.eligibility_engine = EligibilityEngine()

        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(
            stt=stt_model,
            scheme_retriever=self.scheme_retriever,
            eligibility_engine=self.eligibility_engine,
            conversation_memory=self.conversation_memory,
            user_profile=self.user_profile,
            native_language=native_language,
        )
        self.evaluator = EvaluatorAgent(
            conversation_memory=self.conversation_memory,
            user_profile=self.user_profile,
        )

        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def run(
        self,
        user_goal: str,
        initial_intent: str,
        audio_input: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run the main agent loop until SUCCESS or max_retries is reached.

        Args:
            user_goal: High-level goal label (e.g., "CHECK_ELIGIBILITY").
            initial_intent: Initial intent passed to the planner
                (e.g., "START", "COLLECT_PROFILE").
            audio_input: Optional raw audio from the user for the first turn.

        Returns:
            Dictionary summary containing:
                - "status": final evaluation status ("SUCCESS" / "RETRY" / "FAIL")
                - "turns": number of loop iterations executed
        """
        turns = 0
        intent = initial_intent

        # Optional first audio ingestion.
        if audio_input is not None:
            transcript = self.executor.process_user_audio(audio_input)
            logger.info("ToolCall[STT]: transcript='%s'", transcript)

        while turns < self.max_retries:
            turns += 1

            # 1. PLAN
            plan = self.planner.plan(
                user_intent=intent,
                conversation_memory=self.conversation_memory,
                user_profile=self.user_profile,
            )
            logger.info("Plan[%d]: %s", turns, plan)

            # 2. EXECUTE
            response_text = self.executor.execute(plan)
            logger.info("ToolCall[Executor][%d]: response_text='%s'", turns, response_text)

            # 3. EVALUATE
            status_result: EvaluationResult = self.evaluator.evaluate_step(
                user_goal=user_goal,
                last_action=str(plan.get("action", "")),
            )
            logger.info("Evaluation[%d]: status=%s feedback=%s", turns, status_result.status, status_result.feedback)

            if status_result.status == "SUCCESS":
                return {"status": "SUCCESS", "turns": turns}

            if status_result.status == "FAIL":
                return {"status": "FAIL", "turns": turns}

            # Prepare for next iteration: update intent if needed.
            # Here we simply keep the original goal-aligned intent; in a more
            # advanced system, we could derive a new intent from feedback.
            intent = user_goal

        # If we exhausted retries without success, declare FAIL.
        logger.warning("Agent loop reached max_retries=%d without success.", self.max_retries)
        return {"status": "FAIL", "turns": turns}

    def process_turn(
        self,
        user_goal: str,
        audio_input: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Process a single conversational turn (user input → agent response).

        This method is designed for live conversational interactions where:
        - User provides audio input
        - Agent processes it and generates a response
        - Caller handles TTS and waits for next user input

        Args:
            user_goal: High-level goal label (e.g., "CHECK_ELIGIBILITY").
            audio_input: Optional raw audio from the user for this turn.
                If None, the agent will plan based on current state.

        Returns:
            Dictionary containing:
                - "response_text": Agent's response in native language
                - "status": Evaluation status ("SUCCESS" / "RETRY" / "FAIL")
                - "plan": The plan that was executed
                - "should_continue": Boolean indicating if conversation should continue
        """
        # Process user audio if provided
        if audio_input is not None:
            transcript = self.executor.process_user_audio(audio_input)
            logger.info("ToolCall[STT]: transcript='%s'", transcript)

        # Determine intent from current state
        # If we have recent conversation, use goal; otherwise start fresh
        recent_context = self.conversation_memory.get_context(last_n_turns=1)
        intent = user_goal if recent_context else "START"

        # 1. PLAN
        plan = self.planner.plan(
            user_intent=intent,
            conversation_memory=self.conversation_memory,
            user_profile=self.user_profile,
        )
        logger.info("Plan: %s", plan)

        # 2. EXECUTE
        response_text = self.executor.execute(plan)
        logger.info("ToolCall[Executor]: response_text='%s'", response_text)

        # 3. EVALUATE
        status_result: EvaluationResult = self.evaluator.evaluate_step(
            user_goal=user_goal,
            last_action=str(plan.get("action", "")),
        )
        logger.info("Evaluation: status=%s feedback=%s", status_result.status, status_result.feedback)

        # Determine if conversation should continue
        should_continue = status_result.status == "RETRY"

        return {
            "response_text": response_text,
            "status": status_result.status,
            "plan": plan,
            "should_continue": should_continue,
            "feedback": status_result.feedback,
        }


