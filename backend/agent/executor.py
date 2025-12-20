"""
Executor agent for carrying out planner actions.

Responsibilities:
- Execute structured plans from the PlannerAgent.
- Call tools such as STT, scheme retriever, and eligibility engine.
- Update conversation and user profile memory.
- Generate user-facing responses strictly in a native Indian language (here: Hindi).

Internal reasoning is in English only; all returned text is Hindi templates.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from backend.memory.conversation_memory import ConversationMemory
from backend.memory.user_profile import UserProfileMemory
from backend.tools.eligibility_engine import EligibilityEngine
from backend.tools.scheme_retriever import Scheme, SchemeRetriever
from backend.voice.stt import SpeechToText


class ExecutorAgent:
    """
    ExecutorAgent coordinates tool calls and side effects based on a plan.

    The public interface returns **only** native-language (Hindi) strings,
    while all decision logic and comments remain in English.
    """

    def __init__(
        self,
        stt: SpeechToText,
        scheme_retriever: SchemeRetriever,
        eligibility_engine: EligibilityEngine,
        conversation_memory: ConversationMemory,
        user_profile: UserProfileMemory,
        native_language: str = "hi",
    ) -> None:
        """
        Initialise the executor with its dependencies.

        Args:
            stt: Speech-to-text tool for converting raw audio to user text.
            scheme_retriever: Tool for fetching government schemes.
            eligibility_engine: Rule-based eligibility evaluator.
            conversation_memory: Conversation memory store.
            user_profile: Structured user profile memory.
            native_language: ISO code of the native language for responses
                (currently only Hindi templates are implemented).
        """
        self._stt = stt
        self._scheme_retriever = scheme_retriever
        self._eligibility_engine = eligibility_engine
        self._conversation_memory = conversation_memory
        self._user_profile = user_profile
        self._native_language = native_language

        # Cache of the last transcription for optional downstream use.
        self._last_transcript: Optional[str] = None
        self._last_transcript_confidence: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_user_audio(self, audio: Any) -> str:
        """
        Run STT on raw audio, update conversation memory, and return user text.

        This method is intended to be called before planning so that the
        planner and executor together operate on up-to-date conversation
        history.
        """
        result = self._stt.transcribe(audio)
        text = (result.get("text") or "").strip()
        confidence = result.get("confidence")

        self._last_transcript = text
        self._last_transcript_confidence = confidence

        if text:
            self._conversation_memory.add_user_message(text)

        return text

    def execute(self, plan: Dict[str, Any]) -> str:
        """
        Execute a planner plan and return a Hindi response for the user.

        Args:
            plan: Structured plan dictionary, typically from PlannerAgent.plan().

        Returns:
            response_text (Hindi string) suitable for TTS or display.
        """
        action = (plan.get("action") or "").upper()
        field = plan.get("field")
        metadata = plan.get("metadata") or {}

        if action == "ASK_QUESTION" or action == "COLLECT_PROFILE_FIELD":
            response = self._handle_ask_question(field)
        elif action == "RUN_ELIGIBILITY_CHECK":
            response = self._handle_run_eligibility(metadata)
        elif action == "RETRIEVE_SCHEMES":
            response = self._handle_retrieve_schemes(metadata)
        elif action == "END_CONVERSATION":
            response = self._handle_end_conversation()
        else:
            response = self._handle_no_op()

        # Store agent response in conversation memory.
        self._conversation_memory.add_agent_message(response)
        return response

    # ------------------------------------------------------------------ #
    # Action handlers (Hindi templates)
    # ------------------------------------------------------------------ #

    def _handle_ask_question(self, field: Optional[str]) -> str:
        """
        Generate a Hindi question for a specific profile field.
        """
        f = (field or "").lower()

        if f == "income":
            return "कृपया अपना मासिक या वार्षिक आय (राशि और इकाई के साथ) बता दीजिए।"
        if f == "age":
            return "कृपया अपनी आयु वर्षों में बता दीजिए।"
        if f == "state":
            return "आप वर्तमान में किस राज्य में रहते हैं, कृपया राज्य का नाम बताइए।"
        if f == "category":
            return "कृपया अपनी श्रेणी बताइए (जैसे SC, ST, OBC, General)।"
        if f == "gender":
            return "कृपया अपना लिंग/जेंडर बताइए।"

        return "कृपया अपने बारे में थोड़ी और जानकारी दीजिए ताकि मैं सही योजना चुन सकूँ।"

    def _handle_retrieve_schemes(self, metadata: Dict[str, Any]) -> str:
        """
        Retrieve schemes using SchemeRetriever and describe them in Hindi.
        """
        profile_snapshot = self._user_profile.get_profile()
        profile = profile_snapshot.get("profile")
        profile_dict = asdict(profile) if profile is not None else {}

        schemes: List[Scheme] = self._scheme_retriever.retrieve(profile_dict)

        if not schemes:
            return "अभी आपके प्रोफ़ाइल के आधार पर कोई उपयुक्त सरकारी योजना नहीं मिली। हम कुछ और जानकारी लेकर दोबारा कोशिश कर सकते हैं।"

        # List top few schemes by name and a short hint.
        top = schemes[:3]
        names = [s.name for s in top if s.name]
        if not names:
            return "कुछ योजनाएँ मिली हैं, पर उनके नाम उपलब्ध नहीं हैं। कृपया बाद में पुनः प्रयास करें।"

        if len(names) == 1:
            return f"आपके लिए एक संभावित योजना मिली है: {names[0]}."

        joined = " , ".join(names)
        return f"आपके प्रोफ़ाइल के अनुसार कुछ उपयुक्त योजनाएँ मिली हैं: {joined}."

    def _handle_run_eligibility(self, metadata: Dict[str, Any]) -> str:
        """
        Run eligibility checks for retrieved schemes and summarise in Hindi.
        """
        profile_snapshot = self._user_profile.get_profile()
        profile = profile_snapshot.get("profile")
        profile_dict = asdict(profile) if profile is not None else {}

        candidate_schemes: List[Scheme] = self._scheme_retriever.retrieve(profile_dict)
        if not candidate_schemes:
            return "अभी आपके प्रोफ़ाइल के आधार पर कोई भी योजना नहीं मिली, इसलिए पात्रता जाँच संभव नहीं है।"

        eligible_schemes: List[Scheme] = []
        ineligible_schemes: List[Scheme] = []

        for scheme in candidate_schemes:
            result = self._eligibility_engine.evaluate(profile_dict, scheme)
            if result.get("eligible"):
                eligible_schemes.append(scheme)
            else:
                ineligible_schemes.append(scheme)

        if not eligible_schemes:
            return "आपके द्वारा दी गई जानकारी के आधार पर अभी किसी भी उपलब्ध योजना के लिए आप पात्र नहीं दिख रहे हैं।"

        top_eligible = eligible_schemes[:3]
        names = [s.name for s in top_eligible if s.name]

        if len(names) == 1:
            return f"आप {names[0]} योजना के लिए पात्र दिख रहे हैं।"

        joined = " , ".join(names)
        return f"आप निम्न योजनाओं के लिए पात्र दिख रहे हैं: {joined}."

    def _handle_end_conversation(self) -> str:
        """Return a polite Hindi closing message."""
        return "धन्यवाद! यदि आपको आगे और सहायता चाहिए हो तो आप फिर से संपर्क कर सकते हैं।"

    def _handle_no_op(self) -> str:
        """
        Fallback response when no specific action is defined.
        """
        return "मैंने आपकी जानकारी ले ली है। कृपया यदि आप किसी योजना के बारे में जानना चाहते हैं तो थोड़ा और स्पष्ट बताएँ।"


