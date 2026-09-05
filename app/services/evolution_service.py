"""Self-evolution service: the training loop that improves itself unattended.

Cycle (run_cycle):
  1. HARVEST  training rows from the agent's own DB (the 24/7 crawl keeps
              growing them) + any collected jsonl files.
  2. BUILD    dataset via app.ml.dataset_builder into a fresh staging dir.
  3. TRAIN    challenger via training/train.py in an isolated subprocess.
  4. TUNE     per-head global thresholds via training/retune_thresholds.py.
  5. EVALUATE challenger against the hand-labeled golden set + gates.
  6. PROMOTE  only if challenger passes every gate AND beats the current
              champion on golden micro-F1; otherwise the challenger is
              discarded (no regression can ever ship).
  7. SELF-REPAIR
       - if the deployed model fails its canary at cycle start, roll back to
         the newest archived version that still works;
       - after promotion, swap is lock-protected, analyzer/embedder hot-reload
         and stored vectors are rebuilt inline (remainder via patrol).

State/lineage lives in models/repo-analyzer/version.json and
models/evolution_history.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.ml.dataset_builder import build_dataset
from app.ml.model_registry import (
    MODEL_SWAP_LOCK,
    current_version,
    read_version_info,
)
from app.utils.timeutil import utcnow

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "training"
HISTORY_FILE = ROOT / "models" / "evolution_history.json"
CANARY_INPUT = {
    "full_name": "acme/canary",
    "description": "A deep learning library for computer vision and NLP research",
    "language": "Python",
    "topics": ["deep-learning"],
    "readme_excerpt": "",
}


@dataclass
class EvolutionReport:
    started_at: str
    finished_at: str = ""
    outcome: str = "unknown"  # promoted | discarded | rolled_back | skipped | repaired
    reason: str = ""
    data_size: int = 0
    tags: int = 0
    champion_f1: float | None = None
    challenger_f1: float | None = None
    gate_failures: list[str] = field(default_factory=list)
    reembedded: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class EvolutionService:
    def __init__(self, settings: Settings, session_factory, embedding, vectors, analysis):
        self._settings = settings
        self._session_factory = session_factory
        self._embedding = embedding
        self._vectors = vectors
        self._analysis = analysis
        self._swap_lock = MODEL_SWAP_LOCK
        self.last_report: dict | None = None
        self.run_count = 0
        self.history: list[dict] = self._load_history()

    # ------------------------------------------------------------------ utils

    def _load_history(self) -> list[dict]:
        if HISTORY_FILE.exists():
            try:
                return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return []
        return []

    def _save_history(self, report: dict) -> None:
        self.history.append(report)
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(self.history[-30:], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def current_version(self) -> str | None:
        return current_version(self._settings.trained_model_path)

    def _harvest_db(self) -> list[dict]:
        from app.models.project import Project

        with self._session_factory() as session:
            rows = (
                session.query(
                    Project.github_id,
                    Project.full_name,
                    Project.description,
                    Project.language,
                    Project.topics,
                )
                .filter(Project.topics.isnot(None))
                .all()
            )
        repos = []
        for github_id, full_name, description, language, topics in rows:
            repos.append(
                {
                    "github_id": github_id,
                    "full_name": full_name,
                    "description": description,
                    "language": language,
                    "topics": topics or [],
                }
            )
        return repos

    def _collect_jsonl_rows(self) -> list[dict]:
        from app.ml.dataset_builder import read_jsonl_repos

        rows: list[dict] = []
        for path in sorted((TRAINING_DIR / "data").glob("raw_repos*.jsonl")):
            try:
                rows.extend(read_jsonl_repos(path))
            except (OSError, ValueError) as exc:
                logger.warning(f"Skipping unreadable dataset file {path.name}: {exc}")
        return rows

    def _canary(self, model_dir: str | Path) -> bool:
        """A model is deployable only if it loads fresh and answers sensibly."""
        try:
            from app.ml.repo_analyzer import load_repo_analyzer

            model, tokenizer, config, device = load_repo_analyzer(model_dir)
            encoded = tokenizer(CANARY_INPUT["full_name"], return_tensors="pt")
            with __import__("torch").no_grad():
                tag_logits, uc_logits = model(
                    input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
                )
            del model, tokenizer
            return tag_logits.numel() > 0 and uc_logits.numel() > 0
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Canary failed for {model_dir}: {exc}")
            return False

    def _evaluate_model(self, model_dir: str | Path) -> dict:
        """Golden-set micro-F1/recall + gate failures for an arbitrary model dir."""
        sys.path.insert(0, str(TRAINING_DIR))
        import golden_eval  # noqa: PLC0415

        settings = Settings(_env_file=None, trained_model_path=str(model_dir))
        analyzer = golden_eval.TrainedModelAnalyzer(settings)
        gates, cases = golden_eval.load_cases()
        scored, latencies = golden_eval.run_eval(analyzer, cases)
        gated = [s for s in scored if not s["known_hard"]]
        agg = golden_eval.aggregate(gated)
        failures = golden_eval.report_and_gate(scored, latencies, gates)
        return {
            "micro_f1": agg["micro_f1"],
            "recall": agg["micro_recall"],
            "gate_failures": failures,
        }

    def _train_staging(self, data_dir: Path, model_dir: Path, epochs: int) -> None:
        env_python = sys.executable
        commands = [
            [env_python, str(TRAINING_DIR / "train.py"),
             "--data-dir", str(data_dir), "--model-dir", str(model_dir), "--epochs", str(epochs)],
            [env_python, str(TRAINING_DIR / "retune_thresholds.py"),
             "--data-dir", str(data_dir), "--model-dir", str(model_dir)],
        ]
        for command in commands:
            result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "")[-1500:]
                raise RuntimeError(f"training step failed: {command[-1]}: {tail}")

    # --------------------------------------------------------------- repairs

    def _repair_deployed_model(self) -> EvolutionReport | None:
        live = self._settings.trained_model_path
        if self._canary(live):
            return None
        logger.warning(f"Deployed model at '{live}' failed canary; attempting rollback.")
        archives = sorted(
            (ROOT / "models" / "archive").glob("*"),
            key=lambda p: p.name,
            reverse=True,
        )
        for candidate in archives:
            if not candidate.is_dir() or not self._canary(candidate):
                continue
            with self._swap_lock:
                backup = ROOT / "models" / f"broken-{utcnow().strftime('%Y%m%d%H%M%S')}"
                if Path(live).exists():
                    shutil.move(str(live), str(backup))
                shutil.move(str(candidate), str(live))
            logger.warning(f"Rolled back to archived model {candidate.name}.")
            return EvolutionReport(
                started_at=utcnow().isoformat(),
                finished_at=utcnow().isoformat(),
                outcome="repaired",
                reason=f"canary failed; rolled back to {candidate.name}",
            )
        logger.error("No archive passed canary; keeping rule-based fallback and forcing retrain.")
        return None

    # ----------------------------------------------------------- the cycle

    def run_cycle(self, force: bool = False, max_samples: int | None = None, epochs: int | None = None) -> dict:
        report = EvolutionReport(started_at=utcnow().isoformat())
        self.run_count += 1
        try:
            repair = self._repair_deployed_model()
            if repair is not None:
                self.last_report = repair.as_dict()
                self._save_history(self.last_report)
                return self.last_report

            repos = self._harvest_db() + self._collect_jsonl_rows()
            stamp_before = current_version(self._settings.trained_model_path)

            run_id = utcnow().strftime("%Y%m%d%H%M%S")
            work_dir = ROOT / "training" / "data" / "evolution" / run_id
            staging_dir = ROOT / "models" / f"staging-{run_id}"

            stats = build_dataset(
                repos,
                work_dir,
                min_topic_freq=self._settings.evolution_min_topic_freq,
                max_samples=max_samples or self._settings.evolution_max_train_samples,
            )
            report.data_size, report.tags = stats.total, stats.tags

            last_data_size = 0
            if stamp_before:
                last_data_size = int(
                    read_version_info(self._settings.trained_model_path).get("data_size", 0)
                )
            if not force and stats.total - last_data_size < self._settings.evolution_min_new_samples:
                report.outcome = "skipped"
                report.reason = (
                    f"only {stats.total - last_data_size} new samples "
                    f"(< {self._settings.evolution_min_new_samples}); no retrain"
                )
                return self._finish(report)

            epochs = epochs or self._settings.evolution_train_epochs
            logger.info(f"Evolution: training challenger ({stats.total} samples, {epochs} epochs) ...")
            self._train_staging(work_dir, staging_dir, epochs)

            challenger = self._evaluate_model(staging_dir)
            report.challenger_f1 = challenger["micro_f1"]
            report.gate_failures = challenger["gate_failures"]
            if challenger["gate_failures"]:
                report.outcome = "discarded"
                report.reason = f"challenger failed gates: {challenger['gate_failures']}"
                return self._finish(report, discard_staging=staging_dir)

            champion = self._evaluate_model(self._settings.trained_model_path)
            report.champion_f1 = champion["micro_f1"]
            if champion["micro_f1"] >= challenger["micro_f1"]:
                report.outcome = "discarded"
                report.reason = (
                    f"challenger F1 {challenger['micro_f1']} did not beat "
                    f"champion {champion['micro_f1']}"
                )
                return self._finish(report, discard_staging=staging_dir)

            self._promote(staging_dir, challenger, stats.total, report)
        except Exception as exc:  # noqa: BLE001
            report.outcome = report.outcome if report.outcome != "unknown" else "failed"
            report.reason = f"{type(exc).__name__}: {exc}"
            report.errors.append(report.reason)
            logger.exception("Evolution cycle failed")
        return self._finish(report)

    def _finish(self, report: EvolutionReport, discard_staging: Path | None = None) -> dict:
        report.finished_at = utcnow().isoformat()
        if discard_staging is not None:
            shutil.rmtree(discard_staging, ignore_errors=True)
        self.last_report = report.as_dict()
        self._save_history(self.last_report)
        logger.info(f"Evolution {report.outcome}: {report.reason or report.as_dict()}")
        return self.last_report

    def _promote(self, staging_dir: Path, challenger_metrics: dict, data_size: int, report: EvolutionReport) -> None:
        live = Path(self._settings.trained_model_path)
        version = utcnow().strftime("%Y%m%d%H%M%S")
        archive_dir = ROOT / "models" / "archive" / version
        (ROOT / "models" / "archive").mkdir(parents=True, exist_ok=True)

        with self._swap_lock:
            if live.exists():
                shutil.move(str(live), str(archive_dir))
            shutil.move(str(staging_dir), str(live))
            (live / "version.json").write_text(
                json.dumps(
                    {
                        "version": version,
                        "trained_at": utcnow().isoformat(),
                        "data_size": data_size,
                        "epochs": self._settings.evolution_train_epochs,
                        "golden_micro_f1": challenger_metrics["micro_f1"],
                        "golden_recall": challenger_metrics["recall"],
                        "lineage": str(archive_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            # hot reload both consumers under the swap lock (guard for fakes)
            if self._analysis is not None:
                for analyzer in getattr(self._analysis, "analyzers", []) or []:
                    if getattr(analyzer, "name", "") == "trained":
                        analyzer._loaded = None
                        analyzer._stamp = None
                        analyzer._ensure_loaded()
            embedder = getattr(self._embedding, "embedder", None)
            if embedder is not None and hasattr(embedder, "_reload_if_swapped"):
                embedder._reload_if_swapped()
            report.reembedded = self._reembed_all()

        report.outcome = "promoted"
        report.reason = f"challenger promoted as version {version}; archived {archive_dir.name}"

    def _reembed_all(self) -> int:
        """Rebuild vectors with the new model space (bounded; patrol finishes)."""
        from app.models.project import Project
        from app.services.embedding_service import build_project_text, project_embedding_id

        cap = self._settings.evolution_max_reembed
        reembedded = 0
        with self._session_factory() as session:
            projects = (
                session.query(Project)
                .filter(Project.analysis_status == "done")
                .order_by(Project.id)
                .limit(max(cap, 1))
                .all()
            )
            for project in projects:
                try:
                    text = build_project_text(project)
                    vector = self._embedding.embed(text)
                    embedding_id = project_embedding_id(project.id)
                    self._vectors.upsert(
                        [embedding_id],
                        [vector],
                        [text],
                        [{"github_id": project.github_id, "full_name": project.full_name}],
                    )
                    project.embedding_id = embedding_id
                    reembedded += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Re-embed failed for {project.full_name}: {exc}")
                    project.embedding_id = None  # patrol will finish it
            session.commit()
        logger.info(f"Re-embedded {reembedded} vectors in the new model space.")
        return reembedded
