"""
Text-to-speech (TTS) abstractions for Indian languages.

This module defines an abstract ``TextToSpeech`` class that:
- Converts text to speech in a specified Indian language
- Returns playable audio output (engine-defined audio object)
- Supports slow and normal speech modes (e.g., for elderly users)

The implementation is intentionally engine-agnostic so that a concrete
TTS backend (e.g., on-device, cloud, or vendor SDK) can be plugged in
without changing the public interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal


SpeedMode = Literal["normal", "slow"]
AudioOutput = Any  # Concrete implementations should define a specific type (e.g. bytes).


@dataclass
class TTSConfig:
    """
    Configuration for a text-to-speech engine.

    Attributes:
        language: ISO language code for the target Indian language
            (e.g., 'hi' for Hindi, 'bn' for Bengali, 'te' for Telugu).
        normal_rate: Baseline speech rate multiplier (1.0 = engine default).
        slow_rate: Slower speech rate multiplier for elderly or low-literacy users.
    """

    language: str
    normal_rate: float = 1.0
    slow_rate: float = 0.7


class TextToSpeech(ABC):
    """
    Abstract base class for text-to-speech in Indian languages.

    This class encapsulates:
    - Language selection
    - Speed mode handling (normal / slow)
    - A stable public ``speak`` interface

    Concrete subclasses must implement the ``_synthesize`` method, which
    performs the actual TTS using a specific engine.
    """

    def __init__(self, config: TTSConfig) -> None:
        """
        Initialise a new TextToSpeech instance with the given configuration.

        Args:
            config: TTS configuration including language and rate multipliers.
        """
        self._config = config
        self._speed_rates: Dict[SpeedMode, float] = {
            "normal": float(config.normal_rate),
            "slow": float(config.slow_rate),
        }

    @property
    def language(self) -> str:
        """Return the target Indian language code used for synthesis."""
        return self._config.language

    def speak(self, text: str, speed: SpeedMode = "normal") -> AudioOutput:
        """
        Convert text to speech audio in the configured Indian language.

        Args:
            text: Input text to be spoken.
            speed: Speech mode, either ``\"normal\"`` or ``\"slow\"``.

        Returns:
            Engine-defined audio object that is directly playable or can be
            serialized (e.g., raw bytes, WAV buffer, or engine-specific handle).

        Raises:
            ValueError: If an unsupported speed mode is provided.
        """
        if speed not in self._speed_rates:
            raise ValueError(f"Unsupported speed mode: {speed}")

        rate_multiplier = self._speed_rates[speed]
        return self._synthesize(text=text, language=self.language, rate_multiplier=rate_multiplier)

    @abstractmethod
    def _synthesize(self, text: str, language: str, rate_multiplier: float) -> AudioOutput:
        """
        Synthesize speech for the given text and language.

        This method is implemented by concrete subclasses that wrap a real
        TTS engine. It must respect the provided language and rate multiplier.

        Args:
            text: Text to synthesize.
            language: ISO code of the Indian language to use.
            rate_multiplier: Relative speech rate (e.g., 1.0 for normal,
                < 1.0 for slower speech).

        Returns:
            Engine-defined playable audio object.
        """
        raise NotImplementedError


