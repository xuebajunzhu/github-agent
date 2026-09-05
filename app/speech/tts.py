"""Offline text-to-speech via sherpa-onnx + Melo-TTS (Chinese/English mixed)."""
from __future__ import annotations

import io
import threading
import wave
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger


class TextToSpeech(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text into a complete WAV file (16-bit PCM)."""

    def available(self) -> bool:
        return True


def samples_to_wav(samples, sample_rate: int) -> bytes:
    """Convert float32 samples in [-1, 1] to a 16-bit mono WAV file."""
    import numpy as np

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class SherpaOnnxTTS(TextToSpeech):
    name = "sherpa_onnx"

    def __init__(self, model_dir: str, speaker_id: int = 0, speed: float = 1.0, num_threads: int = 2):
        self._model_dir = Path(model_dir)
        self._speaker_id = speaker_id
        self._speed = speed
        self._num_threads = num_threads
        self._tts = None
        self._sample_rate = 44100
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        return (self._model_dir / "model.onnx").exists() and (self._model_dir / "tokens.txt").exists()

    def _ensure_tts(self):
        if self._tts is None:
            import sherpa_onnx

            model_dir = self._model_dir
            logger.info(f"Loading TTS model from '{model_dir}' ...")
            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(model_dir / "model.onnx"),
                        lexicon=str(model_dir / "lexicon.txt")
                        if (model_dir / "lexicon.txt").exists()
                        else "",
                        tokens=str(model_dir / "tokens.txt"),
                        data_dir=str(model_dir / "espeak-ng-data")
                        if (model_dir / "espeak-ng-data").exists()
                        else "",
                        dict_dir=str(model_dir / "dict") if (model_dir / "dict").exists() else "",
                    ),
                    num_threads=self._num_threads,
                ),
                max_num_sentences=1,
            )
            if not config.validate():
                raise RuntimeError("Invalid sherpa-onnx TTS configuration")
            self._tts = sherpa_onnx.OfflineTts(config)
            self._sample_rate = self._tts.sample_rate
            logger.info(f"TTS model ready (sample_rate={self._sample_rate}).")
        return self._tts

    def synthesize(self, text: str) -> bytes:
        tts = self._ensure_tts()
        audio = tts.generate(text, sid=self._speaker_id, speed=self._speed)
        if len(audio.samples) == 0:
            raise RuntimeError("TTS produced no audio")
        with self._lock:
            return samples_to_wav(audio.samples, audio.sample_rate)
