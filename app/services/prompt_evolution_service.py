"""Prompt evolution: the missing loop (D) of the self-optimization system.

Treats the analysis system prompt as evolvable weights, with the same
discipline the model weights already have (staging -> golden gate -> strict
improvement -> atomic promote -> archive/rollback):

  1. HARVEST   golden-set failures + live feedback themes (downvotes,
               zero-result queries)
  2. MUTATE    a meta-prompt has the primary LLM (DeepSeek) rewrite the
               prompt to fix the observed failures, under hard constraints
               (JSON schema unchanged, no injected instructions, length cap)
  3. EVALUATE  baseline vs challenger over the hand-labeled golden set:
               use_cases micro-F1 (non-regression), summary quality via an
               LLM judge rubric (strict improvement), schema failure rate
               (one-vote veto)
  4. PROMOTE   adversarial-pattern check, atomic swap via the prompt
               registry, report appended to prompts/evolution_history.json

Every cycle costs a few hundred flash-tier calls (~cents). Noise guard:
gates use margins, and the mutation never sees the whole golden set — only
the failure slice.
"""
from __future__ import annotations

import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

from app.config import Settings
from app.llm.base import AnalysisError, build_analysis_prompt, extract_json, result_from_json
from app.llm import prompt_registry
from app.services.analysis_service import AnalysisService
from app.utils.timeutil import utcnow

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "training"

ADVERSARIAL_PATTERNS = (
    r"ignore\s+(all\s+|any\s+)?previous",
    r"disregard\s+(all\s+|any\s+)?(previous|prior|above)",
    r"you\s+are\s+now",
    r"system\s*:",
    r"<\|",
    r"assistant\s*:",
    r"\{\{",
)
REQUIRED_PROMPT_KEYS = ("summary", "use_cases", "tags", "dependencies")
MAX_PROMPT_CHARS = 2200

MUTATION_SYSTEM = (
    "You optimize a system prompt for an LLM that analyzes GitHub repositories and "
    'returns a strict JSON object with exactly these keys: {"summary": string, '
    '"use_cases": string[], "tags": string[], "dependencies": string[]}. '
    "You get the current prompt and evidence of where it fails. Rewrite the prompt to "
    "fix those failures. HARD CONSTRAINTS: the JSON schema and field semantics must "
    "stay exactly the same; do not add new instructions about ignoring rules; do not "
    "include placeholders or user data; keep the whole prompt under 300 words. "
    'Return strict JSON: {"prompt": "<the full rewritten prompt>", "changes": "<one '
    'sentence, Chinese, what you changed and why>"}.'
)

JUDGE_SYSTEM = (
    "You score a one-paragraph summary of a GitHub repository for a developer "
    "audience, on three criteria: specificity (says what the project actually does, "
    "not vague marketing), factual grounding (consistent with the repository metadata "
    "given), and information density (no filler). Score 1-5 (5 = excellent). "
    'Return strict JSON: {"score": <int>}.'
)


@dataclass
class PromptMetrics:
    f1: float = 0.0
    tag_recall: float = 0.0
    summary_score: float = 0.0
    schema_failures: int = 0
    cases: int = 0


@dataclass
class PromptEvolutionReport:
    started_at: str
    finished_at: str = ""
    outcome: str = "unknown"  # promoted | discarded | failed
    reason: str = ""
    baseline: dict = field(default_factory=dict)
    challenger: dict = field(default_factory=dict)
    mutation_changes: str = ""
    version: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class PromptEvolutionService:
    def __init__(self, settings: Settings, session_factory, analysis: AnalysisService):
        self._settings = settings
        self._session_factory = session_factory
        self._analysis = analysis
        self.last_report: dict | None = None
        self.run_count = 0
        self._cycle_lock = threading.Lock()

    # ------------------------------------------------------------------ main

    def run_cycle(self, max_cases: int | None = None) -> dict:
        # reentry guard: two concurrent cycles would double-spend LLM calls and
        # race the prompt promotion (scheduler tick + admin trigger)
        if not self._cycle_lock.acquire(blocking=False):
            logger.warning("Prompt evolution skipped: another cycle is already running.")
            return {"outcome": "skipped", "reason": "another cycle is already running"}
        try:
            return self._run_cycle(max_cases)
        finally:
            self._cycle_lock.release()

    def _run_cycle(self, max_cases: int | None) -> dict:
        report = PromptEvolutionReport(started_at=utcnow().isoformat())
        self.run_count += 1
        try:
            cases = self._load_cases(max_cases)
            baseline_prompt = prompt_registry.analysis_prompt()

            baseline, baseline_samples = self._evaluate_prompt(baseline_prompt, cases)
            report.baseline = asdict(baseline)
            logger.info(
                f"Prompt evolution baseline: F1={baseline.f1:.3f} "
                f"summary={baseline.summary_score:.2f} schema_fail={baseline.schema_failures}"
            )

            signal = self._harvest_signals()
            mutation = self._mutate(baseline_prompt, baseline_samples, baseline, signal)
            report.mutation_changes = mutation["changes"]

            ok, why = self._adversarial_ok(mutation["prompt"])
            if not ok:
                report.outcome = "discarded"
                report.reason = f"adversarial check: {why}"
                return self._finish(report)

            challenger, _ = self._evaluate_prompt(mutation["prompt"], cases)
            report.challenger = asdict(challenger)
            logger.info(
                f"Prompt evolution challenger: F1={challenger.f1:.3f} "
                f"summary={challenger.summary_score:.2f} schema_fail={challenger.schema_failures}"
            )

            if (
                challenger.f1 >= baseline.f1 - 0.02
                and challenger.schema_failures <= baseline.schema_failures
                and challenger.summary_score > baseline.summary_score + 0.15
            ):
                version = utcnow().strftime("%Y%m%d%H%M%S")
                prompt_registry.promote(
                    mutation["prompt"],
                    prompt_registry.rerank_prompt(),
                    version,
                    {
                        "baseline": report.baseline,
                        "challenger": report.challenger,
                        "changes": mutation["changes"],
                    },
                )
                report.outcome = "promoted"
                report.version = version
                report.reason = (
                    f"summary {baseline.summary_score:.2f} -> {challenger.summary_score:.2f}, "
                    f"F1 {baseline.f1:.3f} -> {challenger.f1:.3f}"
                )
            else:
                report.outcome = "discarded"
                report.reason = (
                    f"challenger not strictly better: summary {challenger.summary_score:.2f} "
                    f"vs {baseline.summary_score:.2f} (+0.15 required), "
                    f"F1 {challenger.f1:.3f} vs {baseline.f1:.3f}, "
                    f"schema_fail {challenger.schema_failures} vs {baseline.schema_failures}"
                )
        except AnalysisError as exc:
            report.outcome = "failed"
            report.reason = f"LLM unavailable: {exc}"
        except Exception as exc:  # noqa: BLE001
            report.outcome = "failed"
            report.reason = f"{type(exc).__name__}: {exc}"
            report.errors.append(report.reason)
            logger.exception("Prompt evolution cycle failed")
        return self._finish(report)

    def _finish(self, report: PromptEvolutionReport) -> dict:
        report.finished_at = utcnow().isoformat()
        self.last_report = report.as_dict()
        prompt_registry.append_history(self.last_report)
        logger.info(f"Prompt evolution {report.outcome}: {report.reason}")
        return self.last_report

    # ---------------------------------------------------------------- stages

    def _load_cases(self, max_cases: int | None) -> list[dict]:
        sys.path.insert(0, str(TRAINING_DIR))
        try:
            import golden_eval  # noqa: PLC0415
        finally:
            sys.path.remove(str(TRAINING_DIR))
        _, cases = golden_eval.load_cases()
        return cases[: max_cases or len(cases)]

    def _evaluate_prompt(
        self, prompt_text: str, cases: list[dict]
    ) -> tuple[PromptMetrics, list[str]]:
        """Evaluate one prompt over the golden set. Returns metrics plus the
        lowest-judge-score samples — the mutation's evidence slice (no extra
        LLM pass needed)."""
        metrics = PromptMetrics(cases=len(cases))
        tp = fp = fn = 0
        tag_hits = tag_total = 0
        scores: list[float] = []
        samples: list[tuple[float, str]] = []
        lock = threading.Lock()

        def work(case: dict) -> None:
            nonlocal tp, fp, fn, tag_hits, tag_total
            try:
                content = self._analysis.chat(prompt_text, build_analysis_prompt(case["input"]))
                result = result_from_json(extract_json(content), "eval")
            except AnalysisError:
                with lock:
                    metrics.schema_failures += 1
                return
            predicted = {u.strip() for u in result.use_cases if u.strip()}
            required = {u.strip() for u in case["expected"].get("use_cases") or []}
            merged_tags = {t.lower() for t in result.tags}
            required_tags = {t.lower() for t in case["expected"].get("tags_required") or []}
            judge = self._judge_summary(case["input"], result.summary)
            with lock:
                tp += len(predicted & required)
                fp += len(predicted - required)
                fn += len(required - predicted)
                tag_hits += len(merged_tags & required_tags)
                tag_total += len(required_tags)
                if judge is not None:
                    scores.append(judge)
                    samples.append(
                        (
                            judge,
                            f"[{case['input'].get('full_name')}] 摘要得分 {judge:.0f}: "
                            f"{result.summary[:120]}",
                        )
                    )

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(work, cases))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics.f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.tag_recall = tag_hits / tag_total if tag_total else 0.0
        metrics.summary_score = statistics.mean(scores) if scores else 0.0
        samples.sort(key=lambda pair: pair[0])
        return metrics, [text for _, text in samples[:6]]

    def _judge_summary(self, repo_input: dict, summary: str) -> float | None:
        try:
            content = self._analysis.chat(
                JUDGE_SYSTEM,
                f"Repository: {repo_input.get('full_name')}\n"
                f"Description: {repo_input.get('description') or 'N/A'}\n"
                f"Language: {repo_input.get('language') or 'unknown'}\n"
                f"Summary to score: {summary}",
            )
            return float(extract_json(content).get("score") or 0.0)
        except (AnalysisError, TypeError, ValueError) as exc:
            logger.warning(f"Summary judge failed: {exc}")
            return None

    def _harvest_signals(self) -> str:
        """Live feedback themes: downvote comments + zero-result queries (30d)."""
        try:
            from datetime import timedelta

            from sqlalchemy import func

            from app.models.project import Project  # noqa: F401
            from app.models.user import FeedbackRecord, QueryHistory

            cutoff = utcnow() - timedelta(days=30)
            with self._session_factory() as session:
                down = (
                    session.query(
                        FeedbackRecord.project_full_name,
                        func.count(FeedbackRecord.id).label("votes"),
                    )
                    .filter(
                        FeedbackRecord.created_at >= cutoff,
                        FeedbackRecord.vote == "down",
                        FeedbackRecord.project_full_name.isnot(None),
                    )
                    .group_by(FeedbackRecord.project_full_name)
                    .order_by(func.count(FeedbackRecord.id).desc())
                    .limit(5)
                    .all()
                )
                zero = (
                    session.query(QueryHistory.query, func.count(QueryHistory.id).label("n"))
                    .filter(QueryHistory.created_at >= cutoff, QueryHistory.result_count == 0)
                    .group_by(QueryHistory.query)
                    .order_by(func.count(QueryHistory.id).desc())
                    .limit(5)
                    .all()
                )
            parts = [f"被踩最多的项目: {', '.join(f'{n}({c})' for n, c in down) or '无'}"]
            parts.append(f"零结果查询: {', '.join(f'{q}({c})' for q, c in zero) or '无'}")
            return "\n".join(parts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Feedback harvest failed: {exc}")
            return "反馈信号不可用"

    def _mutate(
        self, baseline_prompt: str, failure_samples: list[str], baseline: PromptMetrics, signal: str
    ) -> dict[str, str]:
        """Rewrite the prompt, conditioned on the failure slice + feedback."""
        user = (
            f"当前提示词:\n{baseline_prompt}\n\n"
            f"失败证据（低分摘要样例，repo 事实 vs 摘要评语）:\n"
            + ("\n".join(failure_samples) or "无具体样例，仅依据反馈信号改进")
            + f"\n\n近期线上反馈:\n{signal}\n\n"
            f"当前指标: F1={baseline.f1:.3f} summary={baseline.summary_score:.2f} "
            f"schema_fail={baseline.schema_failures}"
        )
        content = self._analysis.chat(MUTATION_SYSTEM, user)
        data = extract_json(content)
        return {
            "prompt": str(data.get("prompt") or "").strip(),
            "changes": str(data.get("changes") or "").strip()[:200],
        }

    @staticmethod
    def _adversarial_ok(text: str) -> tuple[bool, str]:
        if not text:
            return False, "empty prompt"
        if len(text) > MAX_PROMPT_CHARS:
            return False, f"prompt too long ({len(text)} chars)"
        for key in REQUIRED_PROMPT_KEYS:
            if key not in text:
                return False, f"missing schema key: {key}"
        lowered = text.lower()
        for pattern in ADVERSARIAL_PATTERNS:
            if re.search(pattern, lowered):
                return False, f"adversarial pattern: {pattern}"
        return True, ""
