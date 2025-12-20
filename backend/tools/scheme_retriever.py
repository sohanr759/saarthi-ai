"""
Scheme retrieval tool for matching government schemes to a user profile.

Current behaviour:
- Loads scheme data from a local JSON file.
- Filters schemes based on user state and category.
- Returns a list of top matching schemes with metadata.

This module is intentionally designed so that vector embeddings or more
advanced ranking can be plugged in later without changing the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import logging


SCHEMES_FILE_NAME = "schemes.json"

logger = logging.getLogger(__name__)


@dataclass
class Scheme:
    """
    Represents a single government scheme with relevant metadata.

    Attributes:
        id: Unique identifier of the scheme (if available).
        name: Human-readable scheme name.
        description: Short description of the scheme.
        states: List of applicable state codes or names; may contain "ALL".
        categories: List of applicable user categories (e.g., SC, ST, OBC, General);
            may contain "ALL".
        metadata: Arbitrary additional information from the source JSON.
    """

    id: Optional[str]
    name: str
    description: str
    states: List[str]
    categories: List[str]
    metadata: Dict[str, Any]


class SchemeRetriever:
    """
    Tool for retrieving government schemes that match a given user profile.

    The retriever currently performs:
    - File-based loading of scheme data from ``data/schemes.json``.
    - Simple rule-based filtering on ``state`` and ``category``.

    Extension points for future embeddings / ranking:
    - ``_score_scheme`` can be enhanced to use similarity scores.
    - ``_load_schemes`` can be modified to include pre-computed embeddings
      from the JSON file when available.
    """

    def __init__(self, schemes_path: Optional[Path] = None, max_results: int = 10) -> None:
        """
        Initialise the SchemeRetriever.

        Args:
            schemes_path: Optional explicit path to the schemes JSON file.
                If not provided, ``data/schemes.json`` is resolved relative
                to the project root.
            max_results: Maximum number of schemes to return from ``retrieve``.
        """
        self._max_results = max_results
        self._schemes_path = schemes_path or self._default_schemes_path()
        self._schemes: List[Scheme] = self._load_schemes()

    def retrieve(self, user_profile: Dict[str, Any]) -> List[Scheme]:
        """
        Retrieve schemes matching the given user profile.

        Args:
            user_profile: Dictionary-like structure containing at least:
                - ``state``: User's state (string, case-insensitive).
                - ``category``: User's category (e.g., SC, ST, OBC, General),
                  case-insensitive.

        Returns:
            A list of ``Scheme`` objects ordered by a simple score based on
            state/category match. Length is limited by ``max_results``.
        """
        state = (user_profile.get("state") or "").strip().lower()
        category = (user_profile.get("category") or "").strip().lower()

        scored: List[tuple[float, Scheme]] = []

        for scheme in self._schemes:
            score = self._score_scheme(scheme, state=state, category=category)
            if score > 0:
                scored.append((score, scheme))

        # Order by score descending, keep top N.
        scored.sort(key=lambda item: item[0], reverse=True)
        top_schemes = [scheme for _, scheme in scored[: self._max_results]]
        return top_schemes

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _default_schemes_path(self) -> Path:
        """
        Resolve the default path for the schemes JSON file.

        The file is expected at ``<project_root>/data/schemes.json``.
        """
        # backend/tools/scheme_retriever.py -> project root is two levels up.
        return Path(__file__).resolve().parents[2] / "data" / SCHEMES_FILE_NAME

    def _load_schemes(self) -> List[Scheme]:
        """
        Load scheme definitions from the JSON file.

        Expected JSON structure (flexible but recommended):
        [
            {
                "id": "PM-001",
                "name": "Example Scheme",
                "description": "Short description...",
                "states": ["ALL"] or ["Karnataka", "KA"],
                "categories": ["ALL"] or ["SC", "ST", "OBC", "General"],
                ... any additional metadata ...
            },
            ...
        ]
        """
        if not self._schemes_path.is_file():
            # If the file is missing, return an empty list rather than crashing.
            logger.warning("Schemes file not found: %s", self._schemes_path)
            return []

        try:
            with self._schemes_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning("Schemes file is empty: %s", self._schemes_path)
                    return []
                raw_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse schemes JSON file %s: %s", self._schemes_path, e)
            return []
        except Exception as e:
            logger.error("Error loading schemes file %s: %s", self._schemes_path, e)
            return []

        schemes: List[Scheme] = []
        for item in raw_data:
            if not isinstance(item, dict):
                continue

            states = self._normalize_list(item.get("states"))
            categories = self._normalize_list(item.get("categories"))

            scheme = Scheme(
                id=item.get("id"),
                name=str(item.get("name", "")).strip(),
                description=str(item.get("description", "")).strip(),
                states=states,
                categories=categories,
                metadata={k: v for k, v in item.items() if k not in {"id", "name", "description", "states", "categories"}},
            )
            schemes.append(scheme)

        return schemes

    @staticmethod
    def _normalize_list(value: Any) -> List[str]:
        """
        Normalise a list-like field from the JSON into a lower-cased list.
        """
        if value is None:
            return []
        if isinstance(value, str):
            raw_items: Sequence[str] = [value]
        elif isinstance(value, (list, tuple, set)):
            raw_items = [str(v) for v in value]
        else:
            raw_items = [str(value)]

        return [item.strip().lower() for item in raw_items if item and str(item).strip()]

    def _score_scheme(self, scheme: Scheme, state: str, category: str) -> float:
        """
        Compute a simple relevance score for a scheme.

        Current logic:
        - +1.0 for state match (or if scheme applies to all states).
        - +1.0 for category match (or if scheme applies to all categories).

        This method can be extended later to incorporate semantic similarity
        using embeddings while keeping the same interface.
        """
        score = 0.0

        if state:
            if "all" in scheme.states:
                score += 0.5
            elif state in scheme.states:
                score += 1.0

        if category:
            if "all" in scheme.categories:
                score += 0.5
            elif category in scheme.categories:
                score += 1.0

        return score


