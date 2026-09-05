"""Tests for the self-evolution loop: promote / no-regression / canary rollback.

Training and golden evaluation are stubbed so these run in seconds; the real
subprocess + golden gates are exercised by the standalone cycle run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.evolution_service import EvolutionService
from app.ml.model_registry import current_version


class FakeEmbedder:
    def embed(self, text):
        return [0.0] * 32


class FakeVectors:
    def __init__(self):
        self.data = {}

    def upsert(self, ids, embeddings, documents=None, metadatas=None):
        for vector_id, vector in zip(ids, embeddings):
            self.data[vector_id] = vector

    def existing_ids(self, ids):
        return {i for i in ids if i in self.data}


class FakeAnalysis:
    pass


@pytest.fixture()
def evolution_env(tmp_path, monkeypatch):
    """A live model dir (copied from the real one if present, else tiny) plus
    an EvolutionService whose subprocess training copies the live model and
    writes a challenger metrics file we control per-test."""
    settings = Settings(
        _env_file=None,
        trained_model_path=str(tmp_path / "repo-analyzer"),
        evolution_min_new_samples=0,
        evolution_train_epochs=1,
        evolution_max_train_samples=500,
        evolution_min_topic_freq=2,
    )
    live = Path(settings.trained_model_path)
    live.mkdir(parents=True, exist_ok=True)
    (live / "analyzer_config.json").write_text("{}", encoding="utf-8")
    (live / "version.json").write_text(json.dumps({"version": "base", "data_size": 100}), encoding="utf-8")

    # dataset rows: enough topics to build a vocab
    rows = [
        {
            "github_id": i,
            "full_name": f"acme/repo{i}",
            "description": f"A deep learning toolkit number {i}",
            "language": "Python",
            "topics": ["deep-learning", f"topic{i % 5}"],
        }
        for i in range(30)
    ]

    service = EvolutionService(settings, None, FakeEmbedder(), FakeVectors(), FakeAnalysis())
    service._harvest_db = lambda: []
    service._collect_jsonl_rows = lambda: rows
    yield service, tmp_path, monkeypatch


def _fake_train(service, monkeypatch, challenger_f1, gates_pass=True):
    """Replace subprocess training: copy live model to staging + fake eval."""

    def fake_train(data_dir, model_dir, epochs):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        live = Path(service._settings.trained_model_path)
        for name in ("analyzer_config.json",):
            (model_dir / name).write_text((live / name).read_text(encoding="utf-8"))
        (model_dir / "training_done.marker").write_text(str(challenger_f1), encoding="utf-8")

    def fake_evaluate(model_dir):
        marker = Path(model_dir) / "training_done.marker"
        if marker.exists():
            f1 = float(marker.read_text(encoding="utf-8"))
        else:
            f1 = 0.5  # the live champion
        return {
            "micro_f1": f1,
            "recall": min(1.0, f1 + 0.1),
            "gate_failures": [] if gates_pass else ["use_cases F1"],
        }

    monkeypatch.setattr(service, "_train_staging", fake_train)
    monkeypatch.setattr(service, "_evaluate_model", fake_evaluate)
    # real _canary would load torch models from these fake dirs -> stub it out
    monkeypatch.setattr(service, "_canary", lambda model_dir: True)
    # avoid touching a real DB session in re-embed
    monkeypatch.setattr(service, "_reembed_all", lambda: 5)


def test_promotes_better_challenger(evolution_env, monkeypatch):
    service, tmp_path, monkeypatch = evolution_env
    _fake_train(service, monkeypatch, challenger_f1=0.9)

    report = service.run_cycle(force=True)

    assert report["outcome"] == "promoted"
    assert report["challenger_f1"] == 0.9
    assert current_version(service._settings.trained_model_path) is not None
    assert Path(service._settings.trained_model_path).exists()


def test_discards_worse_challenger(evolution_env, monkeypatch):
    service, tmp_path, monkeypatch = evolution_env
    _fake_train(service, monkeypatch, challenger_f1=0.1)

    report = service.run_cycle(force=True)

    assert report["outcome"] == "discarded"
    assert "did not beat" in report["reason"]
    # live model untouched, still the base version
    assert current_version(service._settings.trained_model_path) == "base"


def test_discards_gate_failures(evolution_env, monkeypatch):
    service, tmp_path, monkeypatch = evolution_env
    _fake_train(service, monkeypatch, challenger_f1=0.99, gates_pass=False)

    report = service.run_cycle(force=True)

    assert report["outcome"] == "discarded"
    assert report["gate_failures"]


def test_rolls_back_when_live_model_broken(evolution_env, monkeypatch):
    service, tmp_path, monkeypatch = evolution_env
    live = Path(service._settings.trained_model_path)
    # seed an archive that still works
    archive = live.parent / "archive" / "oldgood"
    archive.mkdir(parents=True)
    (archive / "analyzer_config.json").write_text("{}", encoding="utf-8")
    # break the live model: canary stub reports failure for the live path only
    (live / "model.pt").write_text("corrupted garbage")

    def fake_canary(model_dir):
        return Path(model_dir) != live

    monkeypatch.setattr(service, "_canary", fake_canary)

    report = service.run_cycle()

    assert report["outcome"] == "repaired"
    assert "rolled back" in report["reason"]
    # the archive content now lives at the live path
    assert (live / "analyzer_config.json").exists()
