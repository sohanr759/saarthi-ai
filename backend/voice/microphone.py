"""
Microphone audio capture utilities for real-time voice input.

This module provides functionality to:
- Record audio from the default microphone
- Support configurable sample rates and durations
- Return audio as numpy arrays compatible with STT systems
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import sounddevice as sd
    import numpy as np
except ImportError as exc:
    raise ImportError(
        "sounddevice and numpy are required for microphone capture. "
        "Install them via 'pip install sounddevice numpy'."
    ) from exc


logger = logging.getLogger(__name__)


class MicrophoneCapture:
    """
    Real-time microphone audio capture for voice input.

    This class provides methods to record audio from the default microphone
    and return it in formats compatible with speech-to-text systems.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        """
        Initialize microphone capture settings.

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000, standard for STT).
            channels: Number of audio channels (1 = mono, 2 = stereo).
            dtype: Data type for audio samples ('float32' or 'int16').
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype

    def record(
        self,
        duration: Optional[float] = None,
        blocking: bool = True,
    ) -> np.ndarray:
        """
        Record audio from the microphone.

        Args:
            duration: Recording duration in seconds. If None, records until
                interrupted (non-blocking mode recommended).
            blocking: If True, blocks until recording completes. If False,
                returns immediately (requires duration to be set).

        Returns:
            NumPy array of shape (samples, channels) containing recorded audio.
            For mono: shape is (samples,), dtype matches self.dtype.

        Raises:
            ValueError: If duration is None and blocking is False.
            RuntimeError: If microphone access fails.
        """
        if duration is None and not blocking:
            raise ValueError("Non-blocking mode requires a duration to be specified.")

        try:
            logger.debug(
                "Recording audio: sample_rate=%d, channels=%d, duration=%s",
                self.sample_rate,
                self.channels,
                duration,
            )

            recording = sd.rec(
                int(duration * self.sample_rate) if duration else 0,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocking=blocking,
            )

            # Convert to mono if needed (STT typically expects mono)
            if self.channels == 1 and len(recording.shape) > 1:
                recording = recording[:, 0]
            elif self.channels == 1:
                recording = recording.flatten()

            logger.debug("Recorded audio shape: %s, dtype: %s", recording.shape, recording.dtype)
            return recording

        except Exception as e:
            logger.error("Failed to record audio: %s", e)
            raise RuntimeError(f"Microphone recording failed: {e}") from e

    def record_with_vad(
        self,
        max_duration: float = 10.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.0,
    ) -> np.ndarray:
        """
        Record audio with Voice Activity Detection (VAD).

        Records until silence is detected or max_duration is reached.

        Args:
            max_duration: Maximum recording duration in seconds.
            silence_threshold: Amplitude threshold below which audio is considered silence.
            silence_duration: Duration of silence (in seconds) before stopping recording.

        Returns:
            NumPy array containing recorded audio (trimmed to actual speech).
        """
        logger.debug("Recording with VAD: max_duration=%.1f, threshold=%.3f", max_duration, silence_threshold)

        # Record in chunks to detect silence
        chunk_duration = 0.1  # 100ms chunks
        chunks = []
        silence_start = None
        max_chunks = int(max_duration / chunk_duration)

        for i in range(max_chunks):
            chunk = self.record(duration=chunk_duration, blocking=True)
            max_amplitude = np.max(np.abs(chunk))

            if max_amplitude > silence_threshold:
                # Speech detected
                chunks.append(chunk)
                silence_start = None
            else:
                # Silence detected
                if silence_start is None:
                    silence_start = i * chunk_duration

                chunks.append(chunk)  # Still record silence

                # Check if we've had enough silence
                if (i * chunk_duration) - silence_start >= silence_duration:
                    logger.debug("Silence detected, stopping recording")
                    break

        if not chunks:
            return np.array([], dtype=self.dtype)

        # Concatenate all chunks
        recording = np.concatenate(chunks)

        # Trim trailing silence
        # Find last point where amplitude exceeds threshold
        for i in range(len(recording) - 1, -1, -1):
            if abs(recording[i]) > silence_threshold:
                recording = recording[: i + int(self.sample_rate * 0.2)]  # Add small buffer
                break

        logger.debug("VAD recording complete: %d samples", len(recording))
        return recording

    @staticmethod
    def list_devices() -> None:
        """Print list of available audio input/output devices."""
        try:
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            print("-" * 70)
            for i, device in enumerate(devices):
                device_type = []
                if device["max_input_channels"] > 0:
                    device_type.append("INPUT")
                if device["max_output_channels"] > 0:
                    device_type.append("OUTPUT")
                print(f"{i}: {device['name']} ({', '.join(device_type)})")
            print("-" * 70)
        except Exception as e:
            logger.error("Failed to list devices: %s", e)
            print(f"Error listing devices: {e}")

    @staticmethod
    def get_default_input_device() -> Optional[int]:
        """Get the default input device index."""
        try:
            return sd.default.device[0]  # Input device index
        except Exception:
            return None

