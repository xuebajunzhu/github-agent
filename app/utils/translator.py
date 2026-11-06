"""Offline zh->en neural translation (Helsinki-NLP opus-mt, CPU).

The zero-cost, zero-dependency leg of the translation chain: when no
chat-capable LLM is available (or it fails), CJK queries are still translated
locally. Runs fully offline once models/opus-mt-zh-en is downloaded via
training/download_translation_model.py.
"""
from __future__ import annotations

import threading

from loguru import logger


class OfflineTranslator:
    name = "opus-mt"

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._translator = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        from pathlib import Path

        path = Path(self._model_path)
        # the HF repo ships TF weights (tf_model.h5); transformers converts
        # them on first load via h5py, so any of these counts as usable
        return any(
            (path / name).exists()
            for name in ("pytorch_model.bin", "model.safetensors", "tf_model.h5")
        )

    def translate(self, text: str) -> str | None:
        try:
            translator = self._ensure()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Offline translator unavailable: {exc}")
            return None
        try:
            with self._lock:
                results = translator([text], max_length=128)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Offline translation failed for {text!r}: {exc}")
            return None
        translation = (results[0]["translation_text"] or "").strip() if results else ""
        return translation or None

    def _ensure(self):
        if self._translator is None:
            from transformers import MarianMTModel, MarianTokenizer

            logger.info(f"Loading offline translator from '{self._model_path}' ...")
            tokenizer = MarianTokenizer.from_pretrained(
                self._model_path, local_files_only=True
            )
            model = MarianMTModel.from_pretrained(self._model_path, local_files_only=True)
            model.eval()
            self._translator = lambda texts, **_: [  # noqa: E731
                {"translation_text": t}
                for t in tokenizer.batch_decode(
                    model.generate(**tokenizer(texts, return_tensors="pt", padding=True)),
                    skip_special_tokens=True,
                )
            ]
            logger.info("Offline translator ready.")
        return self._translator
