from __future__ import annotations

import base64
import io
import wave

from app.speech.tts import samples_to_wav


class FakeSTT:
    name = "fake"

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        return "testing tools"


class FakeTTS:
    name = "fake"

    def synthesize(self, text: str) -> bytes:
        return samples_to_wav([0.0, 0.5, -0.5, 0.0], 16000)


def make_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    return buffer.getvalue()


def test_voice_status_reports_unavailable(client):
    body = client.get("/api/voice/status").json()
    assert body["asr_available"] is False
    assert body["tts_available"] is False
    assert client.get("/api/voice/page").status_code == 200


def test_transcribe_returns_503_without_stt(client):
    response = client.post(
        "/api/voice/transcribe",
        files={"file": ("a.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 503


def test_ask_full_flow_with_fake_speech(client):
    from tests.test_api import seed_project_with_embedding

    services = client.app.state.services
    seed_project_with_embedding(services, 3, "acme/testkit", "a testing toolkit", ["testing"])
    services.stt = FakeSTT()
    services.tts = FakeTTS()

    response = client.post(
        "/api/voice/ask",
        files={"file": ("a.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "testing tools"
    assert body["results"], "expected recommendations"
    assert body["results"][0]["project"]["full_name"] == "acme/testkit"
    assert "为你找到" in body["reply_text"]
    wav = base64.b64decode(body["audio_base64"])
    assert wav[:4] == b"RIFF"  # valid WAV container


def test_ask_returns_503_without_tts(client):
    from tests.test_api import seed_project_with_embedding

    services = client.app.state.services
    seed_project_with_embedding(services, 4, "acme/t2", "another toolkit", ["testing"])
    services.stt = FakeSTT()
    services.tts = None

    response = client.post(
        "/api/voice/ask",
        files={"file": ("a.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 503


def test_speak_requires_text(client):
    services = client.app.state.services
    services.tts = FakeTTS()
    assert client.post("/api/voice/speak", json={"text": " "}).status_code == 400
    ok = client.post("/api/voice/speak", json={"text": "你好"})
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "audio/wav"
