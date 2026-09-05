"""Dataset builder shared by manual training scripts and the self-evolution loop.

Turns raw repository rows (from the DB or collected jsonl files) into
train/val/test jsonl splits plus the label vocabulary.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.ml.taxonomy import match_use_cases


@dataclass
class DatasetStats:
    total: int
    tags: int
    use_cases: int
    train: int
    val: int
    test: int

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "tags": self.tags,
            "use_cases": self.use_cases,
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }


def compose_text(repo: dict) -> str:
    parts = [
        str(repo.get("full_name") or ""),
        str(repo.get("description") or ""),
        " ".join(str(topic) for topic in repo.get("topics") or []),
        str(repo.get("language") or ""),
    ]
    return ". ".join(part for part in parts if part.strip())


def normalize_repo(row: dict) -> dict | None:
    """Accept DB rows or collected jsonl records; None when unusable."""
    description = str(row.get("description") or "").strip()
    topics = [str(topic).lower() for topic in (row.get("topics") or [])]
    if len(description) < 15 or not topics:
        return None
    return {
        "github_id": row.get("github_id"),
        "full_name": row.get("full_name"),
        "description": description,
        "language": row.get("language"),
        "topics": topics,
    }


def build_dataset(
    repos: list[dict],
    output_dir: str | Path,
    *,
    min_topic_freq: int = 40,
    max_tags: int = 300,
    max_samples: int = 12000,
    seed: int = 42,
    val_frac: float = 0.04,
    test_frac: float = 0.04,
) -> DatasetStats:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned: list[dict] = []
    seen: set = set()
    for row in repos:
        repo = normalize_repo(row)
        if repo is None:
            continue
        key = repo["github_id"] if repo.get("github_id") is not None else repo["full_name"]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(repo)

    topic_counter: Counter = Counter()
    for repo in cleaned:
        topic_counter.update(repo["topics"])
    tags = [
        topic
        for topic, frequency in topic_counter.most_common(max_tags)
        if frequency >= min_topic_freq
    ]
    tag_set = set(tags)

    samples: list[dict] = []
    for repo in cleaned:
        tag_labels = sorted({topic for topic in repo["topics"] if topic in tag_set})
        if not tag_labels:
            continue
        samples.append(
            {
                "github_id": repo["github_id"],
                "full_name": repo["full_name"],
                "text": compose_text(repo),
                "tag_labels": tag_labels,
                "use_case_labels": match_use_cases(repo["topics"], repo["description"]),
            }
        )
    random.Random(seed).shuffle(samples)
    samples = samples[:max_samples]

    total = len(samples)
    val_size = max(1, int(total * val_frac))
    test_size = max(1, int(total * test_frac))
    splits = {
        "train": samples[: total - val_size - test_size],
        "val": samples[total - val_size - test_size : total - test_size],
        "test": samples[total - test_size :],
    }
    for split_name, split_samples in splits.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as output:
            for sample in split_samples:
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")

    vocab = {
        "tags": tags,
        "use_cases": sorted({label for sample in samples for label in sample["use_case_labels"]}),
        "stats": {"total": total, **{name: len(part) for name, part in splits.items()}},
    }
    (output_dir / "label_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return DatasetStats(
        total=total,
        tags=len(tags),
        use_cases=len(vocab["use_cases"]),
        train=len(splits["train"]),
        val=val_size,
        test=test_size,
    )


def read_jsonl_repos(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
