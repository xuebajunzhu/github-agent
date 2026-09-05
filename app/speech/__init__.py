"""Speech package: offline voice in/out with pluggable providers."""
from __future__ import annotations

from app.config import Settings
from app.speech.asr import FasterWhisperSTT, SpeechError, SpeechToText
from app.speech.tts import SherpaOnnxTTS, TextToSpeech

__all__ = [
    "SpeechToText",
    "TextToSpeech",
    "SpeechError",
    "FasterWhisperSTT",
    "SherpaOnnxTTS",
    "build_stt",
    "build_tts",
]


def build_stt(settings: Settings) -> SpeechToText | None:
    """Build the configured STT; None when disabled/unavailable."""
    if settings.asr_provider == "none":
        return None
    if settings.asr_provider == "faster_whisper":
        stt = FasterWhisperSTT(
            model_path=settings.asr_model_path,
            compute_type=settings.asr_compute_type,
            language=settings.asr_language,
        )
        if stt.available():
            return stt
    return None


def build_tts(settings: Settings) -> TextToSpeech | None:
    if settings.tts_provider == "none":
        return None
    if settings.tts_provider == "sherpa_onnx":
        tts = SherpaOnnxTTS(
            model_dir=settings.tts_model_dir,
            speaker_id=settings.tts_speaker_id,
            speed=settings.tts_speed,
        )
        if tts.available():
            return tts
    return None
