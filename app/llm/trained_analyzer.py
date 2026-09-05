"""TrainedModelAnalyzer: the locally trained RepoAnalyzerModel as an LLM provider.

Outputs are guaranteed well-formed (classification heads + extractive summary +
dependency mapping), so this provider never produces malformed JSON. It runs
fully offline on CPU and is the recommended primary/cost-free analyzer when a
trained model exists in TRAINED_MODEL_PATH.
"""
from __future__ import annotations

import threading
from pathlib import Path

import torch
from loguru import logger

from app.config import Settings
from app.llm.base import AnalysisResult, LLMAnalyzer
from app.ml.extractive import extractive_summary
from app.ml.repo_analyzer import ANALYZER_CONFIG_FILE, load_repo_analyzer, predict_labels


def compose_analyzer_text(project_data: dict, max_chars: int = 1500) -> str:
    """Compose model input text (must mirror training-time composition)."""
    parts = [
        str(project_data.get("full_name") or ""),
        str(project_data.get("description") or ""),
        " ".join(str(topic) for topic in (project_data.get("topics") or [])),
        str(project_data.get("language") or ""),
        str(project_data.get("readme_excerpt") or ""),
    ]
    return ". ".join(part.strip() for part in parts if part.strip())[:max_chars]


class TrainedModelAnalyzer(LLMAnalyzer):
    name = "trained"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._loaded = None
        self._stamp = None
        # patrol worker threads may race on first use -> guard lazy loading
        self._load_lock = threading.Lock()

    def available(self) -> bool:
        model_dir = Path(self._settings.trained_model_path)
        if not (model_dir / ANALYZER_CONFIG_FILE).exists():
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_loaded(self):
        from app.ml.model_registry import stamp

        current_stamp = stamp(self._settings.trained_model_path)
        if self._loaded is None or current_stamp != self._stamp:
            with self._load_lock:
                current_stamp = stamp(self._settings.trained_model_path)
                if self._loaded is None or current_stamp != self._stamp:
                    logger.info(
                        f"Loading trained repo analyzer from '{self._settings.trained_model_path}' ..."
                    )
                    self._loaded = load_repo_analyzer(self._settings.trained_model_path)
                    self._stamp = current_stamp
                    model, _, config, device = self._loaded
                    logger.info(
                        f"Trained analyzer ready on {device}: {len(config['tags'])} tag labels, "
                        f"{len(config['use_cases'])} use-case labels "
                        f"(metrics: {config.get('metrics', {})})."
                    )
        return self._loaded

    def analyze(self, project_data: dict) -> AnalysisResult:
        model, tokenizer, config, device = self._ensure_loaded()
        text = compose_analyzer_text(project_data)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=config["max_len"],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            tag_logits, use_case_logits = model(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )
        tags = predict_labels(
            tag_logits[0],
            config["tags"],
            config["tag_thresholds"],
            top_k=8,
        )
        use_cases = predict_labels(
            use_case_logits[0],
            config["use_cases"],
            config["use_case_thresholds"],
            top_k=5,
        )

        # Never return empty labels: derive a minimum from repo metadata.
        if not tags:
            tags = [str(topic).lower() for topic in (project_data.get("topics") or [])][:5]
        if not tags and project_data.get("language"):
            tags = [str(project_data["language"]).lower()]
        # Model tags capture implicit themes; repo topics are owner-provided
        # ground truth -- merge both (model tags first, dedup, cap).
        repo_topics = [str(topic).lower() for topic in (project_data.get("topics") or [])]
        for topic in repo_topics:
            if topic not in tags:
                tags.append(topic)
        tags = tags[:8]

        # Ensemble for use_cases: classifier + high-precision rule taxonomy.
        # The union lifts recall -- in particular for Chinese input, which the
        # classifier heads (trained on English text) under-serve.
        from app.ml.taxonomy import match_use_cases

        rule_labels = match_use_cases(
            project_data.get("topics") or [],
            " ".join(
                [
                    " ".join(str(topic) for topic in (project_data.get("topics") or [])),
                    str(project_data.get("description") or ""),
                    str(project_data.get("readme_excerpt") or ""),
                ]
            ),
        )
        for label in rule_labels:
            if label not in use_cases:
                use_cases.append(label)
        use_cases = use_cases[:6]

        description = str(project_data.get("description") or "").strip()
        readme = str(project_data.get("readme_excerpt") or "").strip()
        if description:
            summary = description
        elif readme:
            summary = extractive_summary(
                str(project_data.get("full_name") or ""), readme
            ) or f"{project_data.get('full_name')}: an open-source project."
        else:
            summary = f"{project_data.get('full_name')}: an open-source project."

        dependencies = self._infer_dependencies(
            str(project_data.get("language") or ""), tags
        )
        return AnalysisResult(
            summary=summary[:1000],
            use_cases=use_cases,
            tags=tags,
            dependencies=dependencies,
            provider=self.name,
        )

    @staticmethod
    def _infer_dependencies(language: str, tags: list[str]) -> list[str]:
        from app.llm.rule_based import DEPENDENCIES_BY_LANGUAGE, DEPENDENCIES_BY_TAG

        dependencies: list[str] = []
        dependencies.extend(DEPENDENCIES_BY_LANGUAGE.get(language.lower(), []))
        for tag in tags:
            for dep in DEPENDENCIES_BY_TAG.get(tag, []):
                if dep not in dependencies:
                    dependencies.append(dep)
        return dependencies[:6]
