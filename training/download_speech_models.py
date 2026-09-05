"""Download offline speech models: faster-whisper ASR + Melo-TTS (zh-en).

All downloads go through hf-mirror.com (huggingface.co is unreachable in some
networks). Run once:
    python training/download_speech_models.py
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WHISPER_REPO = "Systran/faster-whisper-small"
TTS_HF_URL = (
    "https://hf-mirror.com/csukuangfj/vits-melo-tts-zh_en/resolve/main/vits-melo-tts-zh_en.tar.bz2"
)
TTS_FALLBACK_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--whisper-model", default=WHISPER_REPO)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    return parser.parse_args()


def download_whisper(models_dir: Path, repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    target = models_dir / "whisper-small"
    if (target / "model.bin").exists():
        print(f"ASR model already present: {target}")
        return target
    print(f"Downloading ASR model {repo_id} via {os.environ['HF_ENDPOINT']} ...")
    path = snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"ASR model saved to {path}")
    return target


def download_tts(models_dir: Path) -> Path:
    target = models_dir / "tts-melo-zh-en"
    if (target / "MODEL_CARD").exists() or any(target.glob("*.onnx")):
        print(f"TTS model already present: {target}")
        return target
    archive = models_dir / "vits-melo-tts-zh_en.tar.bz2"
    for url in (TTS_HF_URL, TTS_FALLBACK_URL):
        try:
            print(f"Downloading TTS model from {url} ...")
            urllib.request.urlretrieve(url, archive)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  download failed: {exc}")
    else:
        raise SystemExit("All TTS download sources failed")
    print("Extracting ...")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(models_dir)
    extracted = models_dir / "vits-melo-tts-zh_en"
    if extracted.exists() and extracted != target:
        extracted.rename(target)
    archive.unlink(missing_ok=True)
    print(f"TTS model saved to {target}")
    return target


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_asr:
        download_whisper(models_dir, args.whisper_model)
    if not args.skip_tts:
        download_tts(models_dir)
    print("Done.")


if __name__ == "__main__":
    main()
