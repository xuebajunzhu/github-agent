"""Re-tune thresholds for a saved model (no retraining needed).

Strategy: with pos_weight-calibrated heads, a single global threshold per head
is robust; per-label thresholds on a small validation split overfit (precision
collapse). Sweeps one threshold per head and rewrites analyzer_config.json.
    python training/retune_thresholds.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from app.ml.repo_analyzer import load_repo_analyzer
from train import RepoDataset, collate, collect_probs, load_split

GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training/data/dataset")
    parser.add_argument("--model-dir", default="models/repo-analyzer")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def best_global_threshold(probs: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    gold = targets.int().numpy()
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in GRID:
        predicted = (probs >= threshold).int().numpy()
        score = float(f1_score(gold, predicted, average="micro", zero_division=0))
        if score > best_f1:
            best_f1, best_threshold = score, threshold
    return best_threshold, best_f1


def main() -> None:
    args = parse_args()
    model, tokenizer, config, device = load_repo_analyzer(args.model_dir)
    vocab = json.loads((Path(args.data_dir) / "label_vocab.json").read_text(encoding="utf-8"))
    tags, use_cases = vocab["tags"], vocab["use_cases"]

    val_samples = load_split(Path(args.data_dir), "val")
    dataset = RepoDataset(val_samples, tags, use_cases)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer, config["max_len"]),
    )
    tag_probs, uc_probs, tag_targets, uc_targets = collect_probs(model, loader, device)

    tag_threshold, tag_f1 = best_global_threshold(tag_probs, tag_targets)
    uc_threshold, uc_f1 = best_global_threshold(uc_probs, uc_targets)

    config["tag_thresholds"] = [tag_threshold] * len(config["tags"])
    config["use_case_thresholds"] = [uc_threshold] * len(config["use_cases"])
    config["metrics"]["global_threshold_tags"] = tag_threshold
    config["metrics"]["global_threshold_use_cases"] = uc_threshold
    config["metrics"]["val_micro_f1_tags@global"] = round(tag_f1, 4)
    config["metrics"]["val_micro_f1_use_cases@global"] = round(uc_f1, 4)

    config_path = Path(args.model_dir) / "analyzer_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"global threshold tags={tag_threshold} (val micro-F1 {tag_f1:.4f})")
    print(f"global threshold use_cases={uc_threshold} (val micro-F1 {uc_f1:.4f})")
    print(f"Updated {config_path}")


if __name__ == "__main__":
    main()
