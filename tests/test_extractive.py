from __future__ import annotations

from app.ml.extractive import extractive_summary, tokenize


def test_tokenize_lowercases_and_covers_chinese():
    tokens = tokenize("Hello World 深度学习")
    assert "hello" in tokens and "world" in tokens
    assert "深度学习" in tokens


def test_picks_informative_first_sentence():
    readme = (
        "# super-tool\n"
        "\n"
        "A blazing-fast command line tool for batch image conversion.\n"
        "\n"
        "## Installation\n"
        "\n"
        "pip install super-tool\n"
        "\n"
        "## License\n"
        "\n"
        "MIT\n"
    )
    summary = extractive_summary("acme/super-tool", readme)
    assert summary.startswith("A blazing-fast command line tool")
    assert "MIT" not in summary
    assert "pip install" not in summary


def test_skips_tables_images_and_html():
    readme = (
        "| col1 | col2 |\n"
        "|-------|------|\n"
        "\n"
        "![logo](logo.png)\n"
        "\n"
        "This project provides utilities for parsing NASA satellite data files.\n"
    )
    summary = extractive_summary("acme/satellite-utils", readme)
    assert "utilities for parsing" in summary


def test_empty_readme_returns_empty():
    assert extractive_summary("acme/x", "") == ""


def test_truncates_long_summary():
    readme = "A " * 300 + "project.\n"
    summary = extractive_summary("acme/x", readme, max_chars=100)
    assert len(summary) <= 105
