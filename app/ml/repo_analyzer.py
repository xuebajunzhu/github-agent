"""RepoAnalyzerModel: a multilingual MiniLM encoder with two multi-label heads.

One model serves the whole analysis step of the agent:
  - tag_head       -> technical tags (label space = GitHub topics vocabulary)
  - use_case_head  -> scenario labels (label space = app.ml.taxonomy)
Summary and dependencies are produced deterministically on top (see
app/ml/extractive.py and app/llm/trained_analyzer.py), which keeps the output
format guaranteed-valid -- no free-form generation, no JSON parsing.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ANALYZER_CONFIG_FILE = "analyzer_config.json"
WEIGHTS_FILE = "model.pt"
ENCODER_CONFIG_DIR = "encoder"
DEFAULT_BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class RepoAnalyzerModel(nn.Module):
    def __init__(
        self,
        base_model: str,
        num_tags: int,
        num_use_cases: int,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        # encoder injection is used by offline tests with a tiny random encoder
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden, num_tags)
        self.use_case_head = nn.Linear(hidden, num_use_cases)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return self.tag_head(pooled), self.use_case_head(pooled)


def save_repo_analyzer(
    model_dir: str | Path,
    model: RepoAnalyzerModel,
    tokenizer,
    tags: list[str],
    use_cases: list[str],
    tag_thresholds: list[float],
    use_case_thresholds: list[float],
    max_len: int,
    metrics: dict | None = None,
) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model": model.base_model,
        "tags": tags,
        "use_cases": use_cases,
        "tag_thresholds": tag_thresholds,
        "use_case_thresholds": use_case_thresholds,
        "max_len": max_len,
        "metrics": metrics or {},
    }
    torch.save(model.state_dict(), model_dir / WEIGHTS_FILE)
    tokenizer.save_pretrained(model_dir)
    # Encoder architecture config: lets load_repo_analyzer rebuild the skeleton
    # locally (weights come from model.pt) without touching the HF hub.
    model.encoder.config.save_pretrained(model_dir / ENCODER_CONFIG_DIR)
    (model_dir / ANALYZER_CONFIG_FILE).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_repo_analyzer(model_dir: str | Path, device: str | None = None):
    """Load a trained analyzer. Returns (model, tokenizer, config, device)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / ANALYZER_CONFIG_FILE).read_text(encoding="utf-8"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rebuild the architecture from the locally saved config (random init);
    # every weight is then overwritten from model.pt. local_files_only=True
    # keeps this path fully offline even when huggingface.co is unreachable.
    encoder_config = AutoConfig.from_pretrained(
        str(model_dir / ENCODER_CONFIG_DIR), local_files_only=True
    )
    encoder = AutoModel.from_config(encoder_config)
    model = RepoAnalyzerModel(
        base_model=config["base_model"],
        num_tags=len(config["tags"]),
        num_use_cases=len(config["use_cases"]),
        encoder=encoder,
    )
    state = torch.load(model_dir / WEIGHTS_FILE, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return model, tokenizer, config, device


def predict_labels(
    logits: torch.Tensor,
    labels: list[str],
    thresholds: list[float],
    top_k: int,
    min_prob: float = 0.05,
) -> list[str]:
    """Sigmoid + per-label thresholds; falls back to the best label above min_prob."""
    probs = torch.sigmoid(logits.detach().float())
    pairs = [
        (labels[index], float(probs[index]))
        for index in range(len(labels))
        if float(probs[index]) >= thresholds[index]
    ]
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    if not pairs:
        best_index = int(probs.argmax())
        if float(probs[best_index]) >= min_prob:
            return [labels[best_index]]
        return []
    return [label for label, _ in pairs[:top_k]]
