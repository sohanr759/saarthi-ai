"""
Decision explanation generator for eligibility results.

Input:
- Eligibility reasoning trace (field -> reason string).
- Native language code (e.g., "hi" for Hindi).

Output:
- Spoken-style explanation in simple, polite language.
- Bullet-structured verbal summary, reusable across different schemes.

This module is deterministic and does not rely on any LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


SUPPORTED_LANGUAGES = {"hi"}  # Extension point: add more Indian languages later.


@dataclass
class DecisionExplanationGenerator:
    """
    Generate a spoken explanation of eligibility decisions.

    The generator:
    - Consumes a reasoning trace (field -> internal reason string).
    - Produces a concise, polite verbal summary in a native Indian language.
    - Uses a bullet-style, numbered structure well-suited for TTS.
    """

    native_language: str = "hi"

    def __post_init__(self) -> None:
        # Fallback to Hindi if an unsupported language code is provided.
        if self.native_language not in SUPPORTED_LANGUAGES:
            self.native_language = "hi"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, reasoning_trace: Dict[str, str]) -> str:
        """
        Generate a spoken explanation from an eligibility reasoning trace.

        Args:
            reasoning_trace: Mapping from field name (e.g., "income", "age")
                to internal explanation string from the EligibilityEngine.

        Returns:
            A single string in the configured native language, structured
            as bullet-style spoken sentences.
        """
        # Normalise to deterministic order so explanations are stable.
        ordered_fields = ["income", "age", "state", "category", "gender"]
        seen_fields: List[Tuple[str, str]] = []

        for field in ordered_fields:
            if field in reasoning_trace:
                seen_fields.append((field, reasoning_trace[field]))

        # Include any additional fields at the end.
        for field, reason in reasoning_trace.items():
            if field not in ordered_fields:
                seen_fields.append((field, reason))

        if self.native_language == "hi":
            return self._generate_hindi(seen_fields)

        # At present we only support Hindi; other languages are future work.
        return self._generate_hindi(seen_fields)

    # ------------------------------------------------------------------ #
    # Language-specific generators
    # ------------------------------------------------------------------ #

    def _generate_hindi(self, items: List[Tuple[str, str]]) -> str:
        """
        Create a Hindi spoken explanation with bullet-style structure.
        """
        if not items:
            return "संक्षेप में कहें तो, अभी हमारे पास आपके निर्णय की विस्तृत जानकारी उपलब्ध नहीं है।"

        lines: List[str] = []
        lines.append("संक्षेप में मैं आपके लिए निर्णय इस तरह समझा रहा हूँ:")

        for idx, (field, raw_reason) in enumerate(items, start=1):
            status = self._classify_status(raw_reason)
            bullet = self._hindi_bullet_line(idx, field, status)
            if bullet:
                lines.append(bullet)

        return " ".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify_status(reason: str) -> str:
        """
        Classify a raw reason string into a compact status label.

        Returns one of:
        - "pass"
        - "fail"
        - "missing"
        - "unknown"
        """
        text = (reason or "").lower()
        if "pass" in text:
            return "pass"
        if "fail" in text:
            return "fail"
        if "missing" in text or "not provided" in text:
            return "missing"
        return "unknown"

    @staticmethod
    def _hindi_field_name(field: str) -> str:
        """Map internal field keys to Hindi-friendly labels."""
        mapping = {
            "income": "आय से जुड़ी शर्त",
            "age": "आयु से जुड़ी शर्त",
            "state": "राज्य से जुड़ी शर्त",
            "category": "श्रेणी से जुड़ी शर्त",
            "gender": "लिंग से जुड़ी शर्त",
        }
        return mapping.get(field, field)

    def _hindi_bullet_line(self, index: int, field: str, status: str) -> str:
        """
        Build a single bullet-style Hindi line for one field decision.
        """
        label = self._hindi_field_name(field)

        prefix = f"बिंदु {index}: "

        if status == "pass":
            return f"{prefix}{label} आप आराम से पूरी करते हैं।"
        if status == "fail":
            return f"{prefix}{label} आप इस समय पूरी नहीं करते हैं।"
        if status == "missing":
            return f"{prefix}{label} के बारे में पर्याप्त जानकारी अभी हमारे पास नहीं है।"

        # Unknown status: keep explanation generic.
        return f"{prefix}{label} के बारे में निर्णय सामान्य नियमों के आधार पर लिया गया है।"


