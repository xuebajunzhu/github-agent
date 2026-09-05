"""Build the training dataset: filter, build label vocab, split, write jsonl."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.taxonomy import match_use_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="training/data/raw_repos.jsonl")
    parser.add_argument("--output-dir", default="training/data/dataset")
    parser.add_argument("--min-topic-freq", type=int, default=40)
    parser.add_argument("--max-tags", type=int, default=300)
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--min-description-chars", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compose_text(repo: dict) -> str:
    parts = [
        str(repo.get("full_name") or ""),
        str(repo.get("description") or ""),
        " ".join(str(topic) for topic in repo.get("topics") or []),
        str(repo.get("language") or ""),
    ]
    return ". ".join(part for part in parts if part.strip())


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw)
    repos: list[dict] = []
    seen: set[int] = set()
    with raw_path.open("r", encoding="utf-8") as raw:
        for line in raw:
            repo = json.loads(line)
            if repo["github_id"] in seen:
                continue
            seen.add(repo["github_id"])
            if len((repo.get("description") or "").strip()) < args.min_description_chars:
                continue
            if not repo.get("topics"):
                continue
            repos.append(repo)

    print(f"{len(repos)} repos after basic filtering.")

    # Tag label space = most frequent GitHub topics.
    topic_counter: Counter = Counter()
    for repo in repos:
        topic_counter.update(str(topic).lower() for topic in repo["topics"])
    tags = [
        topic
        for topic, frequency in topic_counter.most_common(args.max_tags)
        if frequency >= args.min_topic_freq
    ]
    tag_set = set(tags)
    print(f"Tag vocabulary: {len(tags)} topics (freq >= {args.min_topic_freq}).")

    samples: list[dict] = []
    for repo in repos:
        lowered_topics = [str(topic).lower() for topic in repo["topics"]]
        tag_labels = sorted({topic for topic in lowered_topics if topic in tag_set})
        if not tag_labels:
            continue
        text = compose_text(repo)
        use_case_labels = match_use_cases(repo["topics"], repo.get("description") or "")
        samples.append(
            {
                "github_id": repo["github_id"],
                "full_name": repo["full_name"],
                "text": text,
                "tag_labels": tag_labels,
                "use_case_labels": use_case_labels,
            }
        )
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.max_samples]
    print(f"{len(samples)} samples with >=1 in-vocabulary tag label.")

    total = len(samples)
    val_size = max(1, int(total * 0.04))
    test_size = max(1, int(total * 0.04))
    splits = {
        "train": samples[: total - val_size - test_size],
        "val": samples[total - val_size - test_size : total - test_size],
        "test": samples[total - test_size :],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_samples in splits.items():
        with (output_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as output:
            for sample in split_samples:
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(split_samples)}")

    vocab = {
        "tags": tags,
        "use_cases": sorted({label for sample in samples for label in sample["use_case_labels"]}),
        "tag_topic_freq_top20": dict(topic_counter.most_common(20)),
        "stats": {"total": total, **{name: len(part) for name, part in splits.items()}},
    }
    (output_dir / "label_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote dataset to {output_dir}")


if __name__ == "__main__":
    main()
