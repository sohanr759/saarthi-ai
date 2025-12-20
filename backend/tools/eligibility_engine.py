"""
Eligibility engine for checking if a user is eligible for a given scheme.

Responsibilities:
- Deterministic, rule-based evaluation of eligibility.
- Human-readable reasoning trace per field (age, income, state, category, gender).

No LLM usage is involved; the logic is purely declarative and based on
the structured user profile and scheme metadata.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Union

from backend.tools.scheme_retriever import Scheme


UserProfileLike = Union[Dict[str, Any], Any]


class EligibilityEngine:
    """
    Rule-based eligibility engine.

    The engine evaluates a user profile against a ``Scheme`` and returns:
    - ``eligible``: overall boolean eligibility flag.
    - ``reasoning_trace``: mapping from field name to a human-readable string
      explaining why that field passed or failed.
    """

    def evaluate(self, user_profile: UserProfileLike, scheme: Scheme) -> Dict[str, Any]:
        """
        Evaluate eligibility of a user for a specific scheme.

        Args:
            user_profile: User profile object or dict. If it is a dataclass
                or has a ``__dict__``, it will be converted to a dict.
            scheme: The scheme to evaluate against.

        Returns:
            Dictionary with:
                - ``eligible`` (bool)
                - ``reasoning_trace`` (dict[field -> str])
        """
        profile = self._normalise_profile(user_profile)
        reasoning: Dict[str, str] = {}

        age_ok, age_reason = self._check_age(profile, scheme)
        reasoning["age"] = age_reason

        income_ok, income_reason = self._check_income(profile, scheme)
        reasoning["income"] = income_reason

        state_ok, state_reason = self._check_state(profile, scheme)
        reasoning["state"] = state_reason

        category_ok, category_reason = self._check_category(profile, scheme)
        reasoning["category"] = category_reason

        gender_ok, gender_reason = self._check_gender(profile, scheme)
        reasoning["gender"] = gender_reason

        eligible = all([age_ok, income_ok, state_ok, category_ok, gender_ok])

        return {
            "eligible": eligible,
            "reasoning_trace": reasoning,
        }

    # ------------------------------------------------------------------ #
    # Normalisation helpers
    # ------------------------------------------------------------------ #

    def _normalise_profile(self, user_profile: UserProfileLike) -> Dict[str, Any]:
        """
        Convert various user profile representations into a simple dict.
        """
        if isinstance(user_profile, dict):
            return user_profile

        # Handle dataclasses and objects with __dict__
        try:
            return asdict(user_profile)  # type: ignore[arg-type]
        except TypeError:
            if hasattr(user_profile, "__dict__"):
                return dict(user_profile.__dict__)

        # Fallback: return empty dict.
        return {}

    # ------------------------------------------------------------------ #
    # Field-wise checks
    # ------------------------------------------------------------------ #

    def _check_age(self, profile: Dict[str, Any], scheme: Scheme) -> tuple[bool, str]:
        age_raw = profile.get("age")
        age = self._safe_int(age_raw)

        meta = scheme.metadata or {}
        min_age = self._safe_int(self._nested_get(meta, ["min_age", "eligibility", "min_age"]))
        max_age = self._safe_int(self._nested_get(meta, ["max_age", "eligibility", "max_age"]))

        if age is None:
            if min_age is None and max_age is None:
                return True, "No age information required and none provided."
            return False, "Age is missing but the scheme defines age-based eligibility."

        if min_age is not None and age < min_age:
            return False, f"FAIL: age {age} is below the minimum required age {min_age}."

        if max_age is not None and age > max_age:
            return False, f"FAIL: age {age} is above the maximum allowed age {max_age}."

        if min_age is None and max_age is None:
            return True, f"PASS: age {age} provided; scheme does not restrict age."

        range_desc = []
        if min_age is not None:
            range_desc.append(f"≥ {min_age}")
        if max_age is not None:
            range_desc.append(f"≤ {max_age}")
        range_str = " and ".join(range_desc)
        return True, f"PASS: age {age} satisfies scheme age range ({range_str})."

    def _check_income(self, profile: Dict[str, Any], scheme: Scheme) -> tuple[bool, str]:
        income_raw = profile.get("income")
        income = self._safe_float(income_raw)

        meta = scheme.metadata or {}
        min_income = self._safe_float(self._nested_get(meta, ["min_income", "eligibility", "min_income"]))
        max_income = self._safe_float(self._nested_get(meta, ["max_income", "eligibility", "max_income"]))

        if income is None:
            if min_income is None and max_income is None:
                return True, "No income information required and none provided."
            return False, "Income is missing but the scheme defines income-based eligibility."

        if min_income is not None and income < min_income:
            return False, f"FAIL: income {income} is below the minimum required income {min_income}."

        if max_income is not None and income > max_income:
            return False, f"FAIL: income {income} is above the maximum allowed income {max_income}."

        if min_income is None and max_income is None:
            return True, f"PASS: income {income} provided; scheme does not restrict income."

        range_desc = []
        if min_income is not None:
            range_desc.append(f"≥ {min_income}")
        if max_income is not None:
            range_desc.append(f"≤ {max_income}")
        range_str = " and ".join(range_desc)
        return True, f"PASS: income {income} satisfies scheme income range ({range_str})."

    def _check_state(self, profile: Dict[str, Any], scheme: Scheme) -> tuple[bool, str]:
        state_raw = profile.get("state") or ""
        state = str(state_raw).strip().lower()

        if not scheme.states:
            return True, "PASS: scheme does not restrict state."

        scheme_states = [s.lower() for s in scheme.states]

        if "all" in scheme_states:
            if state:
                return True, f"PASS: scheme applies to all states, including '{state}'."
            return True, "PASS: scheme applies to all states."

        if not state:
            return False, "FAIL: state is missing, but the scheme is restricted to specific states."

        if state in scheme_states:
            return True, f"PASS: user state '{state}' is eligible for this scheme."

        return False, f"FAIL: user state '{state}' is not in the eligible states {scheme_states}."

    def _check_category(self, profile: Dict[str, Any], scheme: Scheme) -> tuple[bool, str]:
        category_raw = profile.get("category") or ""
        category = str(category_raw).strip().lower()

        if not scheme.categories:
            return True, "PASS: scheme does not restrict category."

        scheme_categories = [c.lower() for c in scheme.categories]

        if "all" in scheme_categories:
            if category:
                return True, f"PASS: scheme applies to all categories, including '{category}'."
            return True, "PASS: scheme applies to all categories."

        if not category:
            return False, "FAIL: category is missing, but the scheme is restricted to specific categories."

        if category in scheme_categories:
            return True, f"PASS: user category '{category}' is eligible for this scheme."

        return False, f"FAIL: user category '{category}' is not in the eligible categories {scheme_categories}."

    def _check_gender(self, profile: Dict[str, Any], scheme: Scheme) -> tuple[bool, str]:
        gender_raw = profile.get("gender")
        gender = str(gender_raw).strip().lower() if gender_raw is not None else ""

        meta = scheme.metadata or {}
        allowed_genders_raw = self._nested_get(meta, ["allowed_genders", "eligibility", "genders"])
        allowed_genders = self._normalise_str_list(allowed_genders_raw)

        if not allowed_genders:
            if gender:
                return True, f"PASS: gender '{gender}' provided; scheme does not restrict gender."
            return True, "PASS: scheme does not restrict gender."

        if not gender:
            return False, "FAIL: gender is missing, but the scheme is restricted to specific genders."

        if gender in allowed_genders:
            return True, f"PASS: user gender '{gender}' is eligible for this scheme."

        return False, f"FAIL: user gender '{gender}' is not in the eligible genders {allowed_genders}."

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def _nested_get(mapping: Dict[str, Any], keys: list[str]) -> Optional[Any]:
        """
        Try multiple nested key paths and return the first non-None value.

        Example:
            _nested_get(meta, ["min_age", "eligibility", "min_age"])
        """
        # Direct top-level key
        if keys and keys[0] in mapping:
            return mapping.get(keys[0])

        # Nested under a container key (e.g., "eligibility": {"min_age": 18})
        if len(keys) >= 2 and keys[1] in mapping:
            nested = mapping.get(keys[1])
            if isinstance(nested, dict):
                return nested.get(keys[0])

        return None

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(str(value).replace(",", "").strip()))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalise_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, (list, tuple, set)):
            items = [str(v) for v in value]
        else:
            items = [str(value)]

        return [item.strip().lower() for item in items if item and str(item).strip()]


