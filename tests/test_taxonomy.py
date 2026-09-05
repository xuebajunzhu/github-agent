from __future__ import annotations

from app.ml.taxonomy import match_use_cases


def test_topics_drive_use_cases():
    labels = match_use_cases(["deep-learning", "pytorch"], "")
    assert "机器学习" in labels
    assert "深度学习" not in labels  # taxonomy has one merged ML label


def test_keywords_in_description_drive_use_cases():
    labels = match_use_cases([], "A fast web framework for building APIs")
    assert "Web 开发" in labels
    assert "API 开发" in labels


def test_no_match_returns_empty():
    assert match_use_cases(["quantum-computing"], "quantum error correction") in ([], )


def test_deduplicates_labels():
    labels = match_use_cases(["llm", "langchain"], "an llm playground")
    assert labels.count("大模型应用") == 1
