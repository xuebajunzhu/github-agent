"""Evaluate the trained analyzer on the held-out test split."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from app.ml.repo_analyzer import load_repo_analyzer
from train import RepoDataset, collate, load_split, micro_f1, tune_thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training/data/dataset")
    parser.add_argument("--model-dir", default="models/repo-analyzer")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    model, tokenizer, config, device = load_repo_analyzer(args.model_dir)
    vocab = json.loads((data_dir / "label_vocab.json").read_text(encoding="utf-8"))
    tags: list[str] = vocab["tags"]
    use_cases: list[str] = vocab["use_cases"]

    test_samples = load_split(data_dir, "test")
    dataset = RepoDataset(test_samples, tags, use_cases)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, tokenizer, config["max_len"]),
    )

    from train import collect_probs

    tag_probs, uc_probs, tag_targets, uc_targets = collect_probs(model, loader, device)

    def report(probs, targets, thresholds, label_set):
        predictions = torch.zeros_like(probs)
        for index, threshold in enumerate(thresholds):
            predictions[:, index] = (probs[:, index] >= threshold).float()
        gold, pred = targets.int().numpy(), predictions.int().numpy()
        return {
            "micro_f1": round(float(f1_score(gold, pred, average="micro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(gold, pred, average="macro", zero_division=0)), 4),
            "micro_precision": round(float(precision_score(gold, pred, average="micro", zero_division=0)), 4),
            "micro_recall": round(float(recall_score(gold, pred, average="micro", zero_division=0)), 4),
        }

    metrics = {
        "test_samples": len(test_samples),
        "tags": report(tag_probs, tag_targets, config["tag_thresholds"], tags),
        "use_cases": report(uc_probs, uc_targets, config["use_case_thresholds"], use_cases),
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Show a few end-to-end predictions for eyeballing.
    print("\n--- sample predictions ---")
    for sample in test_samples[:5]:
        from app.llm.trained_analyzer import compose_analyzer_text

        encoded = tokenizer(
            compose_analyzer_text(
                {
                    "full_name": sample["full_name"],
                    "description": sample["text"].split(". ")[1] if ". " in sample["text"] else "",
                    "topics": sample["tag_labels"][:3],
                }
            ),
            truncation=True,
            max_length=config["max_len"],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            tag_logits, uc_logits = model(encoded["input_ids"], encoded["attention_mask"])
        from app.ml.repo_analyzer import predict_labels

        predicted_tags = predict_labels(tag_logits[0], config["tags"], config["tag_thresholds"], top_k=6)
        predicted_uc = predict_labels(
            uc_logits[0], config["use_cases"], config["use_case_thresholds"], top_k=4
        )
        print(f"{sample['full_name']}")
        print(f"  gold tags: {sample['tag_labels'][:6]}")
        print(f"  pred tags: {predicted_tags}")
        print(f"  gold use_cases: {sample['use_case_labels']}")
        print(f"  pred use_cases: {predicted_uc}")

    (Path(args.model_dir) / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nMetrics written to {Path(args.model_dir) / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
