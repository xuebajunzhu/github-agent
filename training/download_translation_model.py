"""Download the offline zh->en translation model via the HF mirror.

  .venv/Scripts/python.exe training/download_translation_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = "Helsinki-NLP/opus-mt-zh-en"


def main() -> None:
    from huggingface_hub import snapshot_download

    target = Path("models/opus-mt-zh-en")
    if (target / "pytorch_model.bin").exists() or (target / "model.safetensors").exists():
        print(f"translation model already present: {target}")
        return
    print(f"downloading {REPO} via {os.environ['HF_ENDPOINT']} ...")
    path = snapshot_download(repo_id=REPO, local_dir=str(target))
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
