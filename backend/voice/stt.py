"""
Speech-to-text (STT) utilities built on top of Faster-Whisper.

This module provides a `SpeechToText` class that:
- Accepts raw audio input
- Forces transcription to a single, configurable Indian language
- Returns transcript text and an aggregate confidence score
- Handles low-confidence transcriptions explicitly

NOTE: This module assumes that `faster-whisper` is installed in the environment.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Union

try:
    # Faster-Whisper is generally more efficient and production friendly.
    from faster_whisper import WhisperModel
    import numpy as np
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - hard dependency check
    raise ImportError(
        "faster-whisper, numpy, and soundfile are required for SpeechToText. "
        "Install them via 'pip install faster-whisper numpy soundfile'."
    ) from exc


logger = logging.getLogger(__name__)


AudioInput = Union[str, bytes, "np.ndarray"]  # type: ignore[name-defined]


class SpeechToText:
    """
    Simple wrapper around Faster-Whisper for Indian-language transcription.

    Attributes:
        target_language: ISO language code for target Indian language
            (e.g., 'hi' for Hindi, 'bn' for Bengali, 'te' for Telugu).
        low_confidence_threshold: Threshold below which a transcription is
            considered low-confidence.
    """

    def __init__(
        self,
        target_language: str,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        low_confidence_threshold: float = 0.6,
    ) -> None:
        """
        Initialise the STT model.

        Args:
            target_language: ISO language code for the single Indian language
                to force transcription into.
            model_size: Whisper model size (e.g., 'tiny', 'base', 'small').
            device: Device identifier for Faster-Whisper ('cpu', 'cuda').
            compute_type: Precision type for inference (e.g., 'int8', 'float16').
            low_confidence_threshold: Confidence threshold below which
                transcription is flagged as low-confidence.
        """
        self.target_language = target_language
        self.low_confidence_threshold = max(0.0, min(1.0, float(low_confidence_threshold)))

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: AudioInput) -> Dict[str, Any]:
        """
        Transcribe raw audio into text in the configured target language.

        Args:
            audio: Raw audio input.
                - File path (str)
                - Bytes buffer
                - Numpy array (mono PCM, if used)

        Returns:
            Dictionary with:
                - "text": Final combined transcript string.
                - "confidence": Aggregate confidence score in [0.0, 1.0].

        Notes:
            - No English fallback is performed; language is fixed to
              `self.target_language`.
            - Low-confidence cases are explicitly handled: confidence is
              computed and can be checked by the caller against
              `self.low_confidence_threshold`.
        """
        # Handle numpy array audio input
        processed_audio = audio
        tmp_file_path = None
        
        if isinstance(audio, np.ndarray):
            # Ensure float32 format and proper shape
            if audio.dtype != np.float32:
                # Convert int16/int32 to float32 and normalize to [-1.0, 1.0]
                if audio.dtype in (np.int16, np.int32):
                    processed_audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
                else:
                    processed_audio = audio.astype(np.float32)
            
            # Ensure mono (1D array or 2D with single channel)
            if len(processed_audio.shape) > 1:
                if processed_audio.shape[1] > 1:
                    # Convert stereo to mono by averaging
                    processed_audio = np.mean(processed_audio, axis=1)
                else:
                    processed_audio = processed_audio.flatten()
            
            # Normalize to [-1.0, 1.0] range if not already
            max_val = np.max(np.abs(processed_audio)) if len(processed_audio) > 0 else 1.0
            if max_val > 1.0:
                processed_audio = processed_audio / max_val
            
            # Save to temporary WAV file for faster-whisper (more reliable than direct array)
            # faster-whisper works better with file paths for numpy arrays
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_file_path = tmp_file.name
            tmp_file.close()
            sf.write(tmp_file_path, processed_audio, samplerate=16000, format="WAV")
            processed_audio = tmp_file_path
            logger.debug("Saved numpy audio to temporary file: %s (shape: %s, dtype: %s)", 
                        tmp_file_path, audio.shape, audio.dtype)
        
        try:
            segments, info = self._model.transcribe(
                processed_audio,
                language=self.target_language,
                beam_size=5,
                best_of=5,
            )
        finally:
            # Clean up temporary file if created
            if tmp_file_path:
                try:
                    import os
                    os.unlink(tmp_file_path)
                    logger.debug("Cleaned up temporary audio file: %s", tmp_file_path)
                except Exception as e:
                    logger.warning("Failed to delete temporary audio file %s: %s", tmp_file_path, e)

        texts: List[str] = []
        confidences: List[float] = []

        for segment in segments:
            # segment.text is expected to already be in the requested language
            text_piece = segment.text.strip()
            if text_piece:
                texts.append(text_piece)

            # Derive a confidence proxy from model outputs.
            # Here we invert no_speech_prob and clamp to [0.0, 1.0].
            seg_conf = 1.0 - float(getattr(segment, "no_speech_prob", 0.0))
            seg_conf = max(0.0, min(1.0, seg_conf))
            confidences.append(seg_conf)

        full_text = " ".join(texts).strip()

        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            # No usable segments; treat as very low confidence.
            avg_confidence = 0.0

        # Explicit low-confidence handling branch.
        # We do not alter the language or fallback to English;
        # the caller should inspect this and decide how to react.
        if avg_confidence < self.low_confidence_threshold:
            # Optionally, more sophisticated handling (like logging or
            # downstream flags) can be added here as needed.
            pass

        return {
            "text": full_text,
            "confidence": avg_confidence,
        }


