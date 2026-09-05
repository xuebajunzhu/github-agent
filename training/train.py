"""Train the RepoAnalyzerModel (two multi-label heads) on the prepared dataset.

CPU-friendly by design: multilingual MiniLM encoder, batch 16, short sequences.
Run: python training/train.py [--epochs 3 ...]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# HF hub may be unreachable in some networks; default to the mirror before imports.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import json
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from app.ml.repo_analyzer import DEFAULT_BASE_MODEL, RepoAnalyzerModel, save_repo_analyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training/data/dataset")
    parser.add_argument("--model-dir", default="models/repo-analyzer")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(data_dir: Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


class RepoDataset(Dataset):
    def __init__(self, samples: list[dict], tags: list[str], use_cases: list[str]):
        self.samples = samples
        self.tag_index = {tag: index for index, tag in enumerate(tags)}
        self.use_case_index = {label: index for index, label in enumerate(use_cases)}
        self.num_tags = len(tags)
        self.num_use_cases = len(use_cases)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        tag_target = torch.zeros(self.num_tags)
        for label in sample["tag_labels"]:
            tag_target[self.tag_index[label]] = 1.0
        use_case_target = torch.zeros(self.num_use_cases)
        for label in sample["use_case_labels"]:
            use_case_target[self.use_case_index[label]] = 1.0
        return {"text": sample["text"], "tag_target": tag_target, "use_case_target": use_case_target}


def collate(batch: list[dict], tokenizer, max_len: int) -> dict:
    encoded = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "tag_target": torch.stack([item["tag_target"] for item in batch]),
        "use_case_target": torch.stack([item["use_case_target"] for item in batch]),
    }


def compute_pos_weights(dataset: RepoDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-label pos_weight = negatives/positives, clamped, for imbalanced BCE."""
    tag_pos = torch.zeros(dataset.num_tags)
    uc_pos = torch.zeros(dataset.num_use_cases)
    for index in range(len(dataset)):
        sample = dataset[index]
        tag_pos += sample["tag_target"]
        uc_pos += sample["use_case_target"]
    total = len(dataset)

    def clamp(neg_over_pos: torch.Tensor) -> torch.Tensor:
        return neg_over_pos.clamp(1.0, 25.0)

    tag_weights = clamp((total - tag_pos) / tag_pos.clamp(min=1.0))
    uc_weights = clamp((total - uc_pos) / uc_pos.clamp(min=1.0))
    return tag_weights, uc_weights


@torch.no_grad()
def eval_loss(model, loader, tag_criterion, uc_criterion, device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        tag_logits, uc_logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        loss = tag_criterion(tag_logits, batch["tag_target"].to(device)) + uc_criterion(
            uc_logits, batch["use_case_target"].to(device)
        )
        total += float(loss)
        batches += 1
    model.train()
    return total / max(batches, 1)


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    tag_probs, uc_probs, tag_targets, uc_targets = [], [], [], []
    for batch in loader:
        tag_logits, uc_logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        tag_probs.append(torch.sigmoid(tag_logits.float().cpu()))
        uc_probs.append(torch.sigmoid(uc_logits.float().cpu()))
        tag_targets.append(batch["tag_target"])
        uc_targets.append(batch["use_case_target"])
    return (
        torch.cat(tag_probs),
        torch.cat(uc_probs),
        torch.cat(tag_targets),
        torch.cat(uc_targets),
    )


def micro_f1(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    predictions = (probs >= threshold).int().numpy()
    gold = targets.int().numpy()
    return float(f1_score(gold, predictions, average="micro", zero_division=0))


def tune_thresholds(probs: torch.Tensor, targets: torch.Tensor, grid=None) -> list[float]:
    """Per-label threshold maximizing F1 on validation (0.9 for never-seen labels)."""
    grid = grid if grid is not None else [round(0.1 + 0.05 * step, 2) for step in range(17)]
    thresholds: list[float] = []
    num_labels = targets.shape[1]
    for label_index in range(num_labels):
        gold = targets[:, label_index].int().numpy()
        if gold.sum() == 0:
            thresholds.append(0.9)
            continue
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in grid:
            predicted = (probs[:, label_index] >= threshold).int().numpy()
            tp = int(((predicted == 1) & (gold == 1)).sum())
            fp = int(((predicted == 1) & (gold == 0)).sum())
            fn = int(((predicted == 0) & (gold == 1)).sum())
            score = 2 * tp / max(2 * tp + fp + fn, 1)
            if score > best_f1:
                best_f1, best_threshold = score, threshold
        thresholds.append(best_threshold)
    return thresholds


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    vocab = json.loads((data_dir / "label_vocab.json").read_text(encoding="utf-8"))
    tags: list[str] = vocab["tags"]
    use_cases: list[str] = vocab["use_cases"]
    train_samples = load_split(data_dir, "train")
    val_samples = load_split(data_dir, "val")
    print(f"tags={len(tags)} use_cases={len(use_cases)} train={len(train_samples)} val={len(val_samples)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = RepoAnalyzerModel(
        base_model=args.base_model,
        num_tags=len(tags),
        num_use_cases=len(use_cases),
    ).to(device)

    train_dataset = RepoDataset(train_samples, tags, use_cases)
    val_dataset = RepoDataset(val_samples, tags, use_cases)
    collate_fn = lambda batch: collate(batch, tokenizer, args.max_len)  # noqa: E731
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    tag_weights, uc_weights = compute_pos_weights(train_dataset)
    tag_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=tag_weights.to(device))
    uc_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=uc_weights.to(device))
    print(
        f"pos_weight: tags median={float(tag_weights.median()):.1f} "
        f"use_cases median={float(uc_weights.median()):.1f} (clamped 1..25)"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.06), num_training_steps=total_steps
    )

    best_state, best_val_loss = None, float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            tag_logits, uc_logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            loss = tag_criterion(tag_logits, batch["tag_target"].to(device)) + uc_criterion(
                uc_logits, batch["use_case_target"].to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f} ({time.time() - started:.0f}s)",
                    flush=True,
                )
        tag_probs, uc_probs, tag_targets, uc_targets = collect_probs(model, val_loader, device)
        val_loss = eval_loss(model, val_loader, tag_criterion, uc_criterion, device)
        # With hundreds of sparse labels, micro-F1 @0.5 can legitimately be 0
        # early on; report @0.2 as a learning signal and select checkpoints by
        # validation loss (threshold-independent).
        info = (
            f"epoch {epoch} done in {time.time() - started:.0f}s | val_loss={val_loss:.4f} | "
            f"f1@0.5[tags]={micro_f1(tag_probs, tag_targets, 0.5):.4f} "
            f"f1@0.2[tags]={micro_f1(tag_probs, tag_targets, 0.2):.4f} | "
            f"f1@0.5[uc]={micro_f1(uc_probs, uc_targets, 0.5):.4f} "
            f"f1@0.2[uc]={micro_f1(uc_probs, uc_targets, 0.2):.4f}"
        )
        print(info, flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)

    # Threshold tuning on validation with the best checkpoint.
    tag_probs, uc_probs, tag_targets, uc_targets = collect_probs(model, val_loader, device)
    tag_thresholds = tune_thresholds(tag_probs, tag_targets)
    uc_thresholds = tune_thresholds(uc_probs, uc_targets)

    save_repo_analyzer(
        model_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
        tags=tags,
        use_cases=use_cases,
        tag_thresholds=tag_thresholds,
        use_case_thresholds=uc_thresholds,
        max_len=args.max_len,
        metrics={
            "val_micro_f1_tags@0.5": round(micro_f1(tag_probs, tag_targets, 0.5), 4),
            "val_micro_f1_tags@0.2": round(micro_f1(tag_probs, tag_targets, 0.2), 4),
            "val_micro_f1_use_cases@0.5": round(micro_f1(uc_probs, uc_targets, 0.5), 4),
            "val_micro_f1_use_cases@0.2": round(micro_f1(uc_probs, uc_targets, 0.2), 4),
            "best_val_loss": round(best_val_loss, 4),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "epochs": args.epochs,
            "base_model": args.base_model,
        },
    )
    print(f"Saved best model (val_loss {best_val_loss:.4f}) to {model_dir}")


if __name__ == "__main__":
    main()
