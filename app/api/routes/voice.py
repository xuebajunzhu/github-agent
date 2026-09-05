"""Voice endpoints: speech in -> recommendations -> speech out (all offline)."""
from __future__ import annotations

import base64
import re
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from loguru import logger
from pydantic import BaseModel

from app.schemas.recommend import RecommendationItem
from app.speech.asr import SpeechError, SpeechToText
from app.speech.tts import TextToSpeech

router = APIRouter(prefix="/api/voice", tags=["voice"])

VOICE_PAGE = Path(__file__).resolve().parents[2] / "static" / "voice.html"

_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
_MARKDOWN_RE = re.compile(r"[#*`_\[\]()!<>|]")


class SpeakRequest(BaseModel):
    text: str


class VoiceAskResponse(BaseModel):
    query: str
    reply_text: str
    results: list[RecommendationItem]
    audio_base64: str
    audio_format: str = "wav"


def _require_stt(services) -> SpeechToText:
    if services.stt is None:
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text is not available. Run: python training/download_speech_models.py "
            "(or set ASR_PROVIDER=none to disable).",
        )
    return services.stt


def _require_tts(services) -> TextToSpeech:
    if services.tts is None:
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech is not available. Run: python training/download_speech_models.py "
            "(or set TTS_PROVIDER=none to disable).",
        )
    return services.tts


def _clean_for_speech(text: str, max_chars: int = 80) -> str:
    """Strip markdown/emoji and noise so TTS produces natural speech."""
    cleaned = _EMOJI_RE.sub("", text or "")
    cleaned = _MARKDOWN_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def compose_spoken_reply(query: str, hits: list) -> str:
    if not hits:
        return "没有找到相关的开源项目，请换个说法再试试。"
    parts = [f"为你找到{len(hits)}个相关项目。"]
    for index, hit in enumerate(hits, start=1):
        name = hit.project.full_name.replace("/", " ")
        summary = _clean_for_speech(
            hit.project.ai_summary or hit.project.description or "", max_chars=60
        )
        parts.append(f"第{index}个，{name}，{summary}。")
    reply = " ".join(parts)
    logger.debug(f"Spoken reply for {query!r}: {reply}")
    return reply


@router.get("/status")
def voice_status(request: Request) -> dict:
    services = request.app.state.services
    return {
        "asr_available": services.stt is not None,
        "tts_available": services.tts is not None,
        "asr_provider": services.stt.name if services.stt else None,
        "tts_provider": services.tts.name if services.tts else None,
    }


@router.post("/transcribe")
async def voice_transcribe(request: Request, file: UploadFile = File(...)) -> dict:
    services = request.app.state.services
    stt = _require_stt(services)
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    try:
        text = stt.transcribe(audio, file.filename or "audio.wav")
    except SpeechError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"text": text}


@router.post("/speak")
def voice_speak(payload: SpeakRequest, request: Request) -> Response:
    services = request.app.state.services
    tts = _require_tts(services)
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        wav = tts.synthesize(text)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"TTS failed: {exc}")
        raise HTTPException(status_code=503, detail="Text-to-speech synthesis failed") from exc
    return Response(content=wav, media_type="audio/wav")


@router.post("/ask", response_model=VoiceAskResponse)
async def voice_ask(request: Request, file: UploadFile = File(...)) -> VoiceAskResponse:
    """One-shot voice interaction: audio in -> recommendations -> spoken audio out."""
    services = request.app.state.services
    stt = _require_stt(services)
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    try:
        query = stt.transcribe(audio, file.filename or "audio.wav")
    except SpeechError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    hits = services.recommend.recommend(query, top_k=services.settings.voice_reply_max_results)
    reply_text = compose_spoken_reply(query, hits)
    tts = _require_tts(services)
    try:
        wav = tts.synthesize(reply_text)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"TTS failed: {exc}")
        raise HTTPException(status_code=503, detail="Text-to-speech synthesis failed") from exc

    return VoiceAskResponse(
        query=query,
        reply_text=reply_text,
        results=[
            RecommendationItem(
                project=hit.project,
                score=round(hit.score, 4),
                dependencies=hit.dependencies,
                reason=hit.reason,
            )
            for hit in hits
        ],
        audio_base64=base64.b64encode(wav).decode("ascii"),
    )


@router.get("/page", include_in_schema=False)
def voice_page() -> FileResponse:
    if not VOICE_PAGE.exists():
        raise HTTPException(status_code=404, detail="Voice page not found")
    return FileResponse(VOICE_PAGE)
