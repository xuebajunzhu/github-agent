"""Regression gate: the hand-labeled golden eval must pass in CI.

Runs the full golden set through the real trained analyzer (~15s on CPU).
Skipped automatically when the trained model or torch is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from app.config import Settings  # noqa: E402

GOLDEN_MODULE_DIR = Path(__file__).resolve().parents[1] / "training"


def _load_golden_eval():
    import sys

    if str(GOLDEN_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(GOLDEN_MODULE_DIR))
    import golden_eval  # noqa: PLC0415

    return golden_eval


def test_golden_eval_gates_pass():
    if not (Path("models/repo-analyzer") / "analyzer_config.json").exists():
        pytest.skip("trained model not present; run training first")

    golden_eval = _load_golden_eval()
    gates, cases = golden_eval.load_cases()
    analyzer = golden_eval.TrainedModelAnalyzer(Settings(_env_file=None))
    assert analyzer.available(), "trained model files exist but cannot be loaded"

    scored, latencies = golden_eval.run_eval(analyzer, cases)
    failures = golden_eval.report_and_gate(scored, latencies, gates)
    assert not failures, f"golden eval gate failures: {failures}"
