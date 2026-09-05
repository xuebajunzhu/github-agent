from __future__ import annotations

from app.llm.rule_based import RuleBasedAnalyzer

analyzer = RuleBasedAnalyzer()


def test_extracts_tags_use_cases_and_dependencies():
    result = analyzer.analyze(
        {
            "full_name": "acme/vision",
            "description": "A deep learning library for computer vision and image processing",
            "language": "Python",
            "topics": ["PyTorch"],
            "readme_excerpt": "",
        }
    )
    assert result.provider == "rule"
    assert result.summary.startswith("A deep learning library")
    assert "深度学习" in result.use_cases
    assert "计算机视觉" in result.use_cases
    assert "pytorch" in result.tags
    assert "python" in result.tags
    assert "pip" in result.dependencies
    assert "torch" in result.dependencies


def test_summary_falls_back_to_readme_first_paragraph():
    result = analyzer.analyze(
        {
            "full_name": "acme/tool",
            "description": None,
            "language": "Go",
            "topics": [],
            "readme_excerpt": "# acme-tool\n\nA tiny command line utility for batch image conversion.\n",
        }
    )
    assert result.summary.startswith("A tiny command line utility")
    assert "命令行工具" in result.use_cases
    assert "go modules" in result.dependencies


def test_never_crashes_on_empty_input():
    result = analyzer.analyze({})
    assert result.summary
    assert result.tags == []
