"""Integration tests for TrainedModelAnalyzer using a tiny offline model.

Builds a random-weight 1-layer Bert encoder + minimal vocab file so the whole
save -> load -> analyze path is exercised without network or a real checkpoint.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertModel, BertTokenizer

from app.config import Settings
from app.llm.trained_analyzer import TrainedModelAnalyzer, compose_analyzer_text
from app.ml.repo_analyzer import (
    ANALYZER_CONFIG_FILE,
    RepoAnalyzerModel,
    save_repo_analyzer,
)

TINY_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "acme", "vision", "lib", "deep", "learning", "computer", "python",
    "web", "framework", "tool", "for", "building", "apis", "a",
]
TAGS = ["python", "deep-learning", "web", "cli", "database", "rust"]
USE_CASES = ["机器学习", "Web 开发", "命令行工具", "数据存储"]


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("repo-analyzer")
    vocab_file = model_dir / "vocab.txt"
    vocab_file.write_text("\n".join(TINY_VOCAB), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
    config = BertConfig(
        vocab_size=len(TINY_VOCAB),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    encoder = BertModel(config)
    model = RepoAnalyzerModel(
        base_model="tiny-test",
        num_tags=len(TAGS),
        num_use_cases=len(USE_CASES),
        encoder=encoder,
    )
    save_repo_analyzer(
        model_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
        tags=TAGS,
        use_cases=USE_CASES,
        tag_thresholds=[0.0] * len(TAGS),  # everything passes -> exercises top_k path
        use_case_thresholds=[0.0] * len(USE_CASES),
        max_len=32,
        metrics={"smoke": True},
    )
    return model_dir


def make_analyzer(model_dir, tag_threshold=None, uc_threshold=None) -> TrainedModelAnalyzer:
    settings = Settings(_env_file=None, trained_model_path=str(model_dir))
    if tag_threshold is not None:
        import json
        from pathlib import Path

        config_path = Path(model_dir) / ANALYZER_CONFIG_FILE
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["tag_thresholds"] = [tag_threshold] * len(config["tags"])
        config["use_case_thresholds"] = [uc_threshold] * len(config["use_cases"])
        config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return TrainedModelAnalyzer(settings)


def test_available_false_when_model_missing(tmp_path):
    settings = Settings(_env_file=None, trained_model_path=str(tmp_path / "nope"))
    assert not TrainedModelAnalyzer(settings).available()


def test_analyze_produces_structured_result(trained_model_dir):
    analyzer = make_analyzer(trained_model_dir)
    assert analyzer.available()
    result = analyzer.analyze(
        {
            "full_name": "acme/vision-lib",
            "description": "A deep learning library for computer vision",
            "language": "Python",
            "topics": ["pytorch"],
            "readme_excerpt": "",
        }
    )
    assert result.provider == "trained"
    assert result.summary.startswith("A deep learning library")
    assert result.tags and all(isinstance(tag, str) for tag in result.tags)
    assert result.use_cases and all(isinstance(label, str) for label in result.use_cases)
    assert "pip" in result.dependencies


def test_summary_uses_extractive_when_description_missing(trained_model_dir):
    analyzer = make_analyzer(trained_model_dir)
    result = analyzer.analyze(
        {
            "full_name": "acme/super-tool",
            "description": None,
            "language": "Rust",
            "topics": [],
            "readme_excerpt": "A blazing fast command line tool for batch conversion.\n" * 3,
        }
    )
    assert "blazing fast command line tool" in result.summary


def test_empty_predictions_fall_back_to_metadata(trained_model_dir, monkeypatch):
    # Simulate a model whose heads predict nothing -> metadata fallbacks must kick in
    import app.llm.trained_analyzer as trained_analyzer_module

    monkeypatch.setattr(
        trained_analyzer_module, "predict_labels", lambda *args, **kwargs: []
    )
    analyzer = make_analyzer(trained_model_dir)
    result = analyzer.analyze(
        {
            "full_name": "acme/ml-kit",
            "description": "Machine learning utilities",
            "language": "Python",
            "topics": ["pytorch"],
            "readme_excerpt": "",
        }
    )
    assert result.tags == ["pytorch"]
    assert "机器学习" in result.use_cases


def test_compose_text_is_stable():
    text = compose_analyzer_text(
        {
            "full_name": "a/b",
            "description": " desc ",
            "topics": ["X", "y"],
            "language": "Python",
            "readme_excerpt": "readme",
        }
    )
    assert text == "a/b. desc. X y. Python. readme"


def test_trained_embedder_uses_same_model(trained_model_dir):
    from app.embedders.trained_embedder import TrainedEmbedder

    embedder = TrainedEmbedder(str(trained_model_dir))
    assert embedder.name == "trained"
    assert embedder.dimension == 32  # tiny encoder hidden size
    vectors = embedder.embed(["deep learning python", "web framework"])
    assert len(vectors) == 2
    assert all(len(vector) == 32 for vector in vectors)
    norm = sum(value * value for value in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-5  # L2 normalized
