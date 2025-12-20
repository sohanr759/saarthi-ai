"""
User profile memory module for storing structured user attributes.

This module provides deterministic, in‑memory handling of a user's
profile details such as age, income, state, gender, and category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Literal, Any


AttributeKey = Literal["age", "income", "state", "gender", "category"]


class ProfileFieldStatus(TypedDict):
    """Represents value and confidence for a single profile field."""

    value: Optional[Any]
    confidence: float


@dataclass
class UserProfile:
    """
    Structured representation of the user's profile attributes.

    Attributes:
        age: User age in years (if known).
        income: User income (numeric, normalised representation if applicable).
        state: User's state / region name.
        gender: User's gender label.
        category: Socio‑economic / reservation category (e.g., SC, ST, OBC, General).
    """

    age: Optional[int] = None
    income: Optional[float] = None
    state: Optional[str] = None
    gender: Optional[str] = None
    category: Optional[str] = None


class UserProfileMemory:
    """
    In‑memory store for structured user profile attributes.

    This class:
    - Keeps track of key user attributes with an associated confidence score.
    - Allows incremental deterministic updates to each field.
    - Can report whether the profile is complete and which fields are missing
      or uncertain.

    Confidence is a float in the range [0.0, 1.0].
    """

    _FIELDS: List[AttributeKey] = ["age", "income", "state", "gender", "category"]

    def __init__(self) -> None:
        """Initialise an empty profile with zero confidence for all fields."""
        self._data: Dict[AttributeKey, ProfileFieldStatus] = {
            field: {"value": None, "confidence": 0.0} for field in self._FIELDS
        }

    def update(self, field: AttributeKey, value: Any, confidence: float) -> None:
        """
        Update a specific profile field with a new value and confidence.

        Args:
            field: The profile field to update (age, income, state, gender, category).
            value: The new value for the field.
            confidence: Confidence level for this value between 0.0 and 1.0.

        Notes:
            - Confidence is clamped to the range [0.0, 1.0].
            - If an existing value has higher confidence, it is retained.
            - If the new value contradicts the existing one but has higher
              confidence, it replaces the previous value.
        """
        if field not in self._FIELDS:
            raise ValueError(f"Unsupported field: {field}")

        clamped_confidence = max(0.0, min(1.0, float(confidence)))

        current = self._data[field]
        current_value = current["value"]
        current_conf = current["confidence"]

        # If we have no value yet, accept the update.
        if current_value is None or current_conf == 0.0:
            self._data[field] = {"value": self._normalise(field, value), "confidence": clamped_confidence}
            return

        # If new confidence is lower or equal and value is same, keep best.
        normalised_new = self._normalise(field, value)
        if clamped_confidence <= current_conf and self._values_equal(current_value, normalised_new):
            return

        # Prefer higher confidence values; in case of conflict, overwrite only
        # when new confidence is strictly higher.
        if clamped_confidence > current_conf:
            self._data[field] = {"value": normalised_new, "confidence": clamped_confidence}

    def is_complete(self) -> bool:
        """
        Check whether all required fields are present with reasonable confidence.

        Returns:
            True if every field has a non‑None value and confidence > 0.6,
            otherwise False.
        """
        for status in self._data.values():
            if status["value"] is None or status["confidence"] <= 0.6:
                return False
        return True

    def get_missing_fields(self) -> List[AttributeKey]:
        """
        Get list of fields that are missing or have low confidence.

        Returns:
            List of field names that should be confirmed or asked from the user.
        """
        missing: List[AttributeKey] = []
        for field, status in self._data.items():
            if status["value"] is None or status["confidence"] <= 0.6:
                missing.append(field)
        return missing

    def get_profile(self) -> Dict[str, Any]:
        """
        Get a structured snapshot of the current profile and confidences.

        Returns:
            Dictionary containing:
                - "profile": a UserProfile instance with current values.
                - "confidence": mapping from field name to confidence score.
        """
        profile = UserProfile(
            age=self._safe_cast_int(self._data["age"]["value"]),
            income=self._safe_cast_float(self._data["income"]["value"]),
            state=self._safe_cast_str(self._data["state"]["value"]),
            gender=self._safe_cast_str(self._data["gender"]["value"]),
            category=self._safe_cast_str(self._data["category"]["value"]),
        )
        confidence: Dict[str, float] = {
            field: status["confidence"] for field, status in self._data.items()
        }
        return {"profile": profile, "confidence": confidence}

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _normalise(self, field: AttributeKey, value: Any) -> Any:
        """
        Normalise raw values per field for consistent internal storage.

        Args:
            field: Field name being updated.
            value: Raw value passed into `update`.
        """
        if value is None:
            return None

        if field == "age":
            return self._safe_cast_int(value)

        if field == "income":
            return self._safe_cast_float(value)

        if field in ("state", "gender", "category"):
            text = self._safe_cast_str(value)
            return text.strip().title() if text is not None else None

        return value

    @staticmethod
    def _values_equal(a: Any, b: Any) -> bool:
        """Check equality for stored values in a tolerant way."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-6
        return a == b

    @staticmethod
    def _safe_cast_int(value: Any) -> Optional[int]:
        """Best‑effort cast of a value to int; returns None on failure."""
        if value is None:
            return None
        try:
            return int(float(str(value).replace(",", "").strip()))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_cast_float(value: Any) -> Optional[float]:
        """Best‑effort cast of a value to float; returns None on failure."""
        if value is None:
            return None
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_cast_str(value: Any) -> Optional[str]:
        """Best‑effort cast of a value to str; returns None on failure."""
        if value is None:
            return None
        try:
            text = str(value).strip()
            return text or None
        except Exception:
            return None


