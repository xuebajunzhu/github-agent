"""Offline speech-to-text via faster-whisper (CPU, int8)."""
from __future__ import annotations

import io
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger


class SpeechError(Exception):
    pass


class SpeechToText(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio bytes into text. Raises SpeechError on failure."""

    def available(self) -> bool:
        return True


class FasterWhisperSTT(SpeechToText):
    """Multilingual (zh/en) ASR, fully offline once the model is downloaded."""

    name = "faster_whisper"

    def __init__(
        self,
        model_path: str,
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self._model_path = model_path
        self._compute_type = compute_type
        self._language = language
        self._model = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False
        return (Path(self._model_path) / "model.bin").exists()

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(f"Loading ASR model from '{self._model_path}' ({self._compute_type}) ...")
            self._model = WhisperModel(self._model_path, device="cpu", compute_type=self._compute_type)
            logger.info("ASR model ready.")
        return self._model

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        model = self._ensure_model()
        try:
            segments, info = model.transcribe(
                io.BytesIO(audio_bytes),
                language=self._language,
                vad_filter=True,
            )
            with self._lock:
                text = "".join(segment.text for segment in segments).strip()
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Transcription failed: {exc}") from exc
        if not text:
            raise SpeechError("No speech recognized in the audio")
        logger.debug(f"ASR language={info.language} prob={info.language_probability:.2f}")
        return text
