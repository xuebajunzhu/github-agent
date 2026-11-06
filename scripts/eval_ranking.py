"""Offline ranking evaluation: measure what rerank/hybrid actually buy.

Runs the live index (read-only) over a small set of queries with known
relevant projects, in several configurations, and reports Precision@3,
MRR and latency. This is the data source for the rerank-vs-fusion tuning
decision -- no more anecdotes.

  .venv/Scripts/python.exe scripts/eval_ranking.py                 # all modes
  .venv/Scripts/python.exe scripts/eval_ranking.py --no-rerank     # single mode
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.database import create_db_engine, init_db  # noqa: E402
from app.services.container import build_services  # noqa: E402
from app.services.recommend_service import RecommendService  # noqa: E402

# (query, [relevant full_names]) — relevance verified by hand against the index
CASES: list[tuple[str, set[str]]] = [
    ({"query": "我需要一个终端文件管理器"}, {"gokcehan/lf", "jarun/nnn", "yorukot/superfile", "dylanaraps/fff", "sxyazi/yazi"}),
    ({"query": "我需要一个监控服务器性能的面板"}, {"netdata/netdata", "martinrusev/amon", "BSoDium/Slashboard-desktop"}),
    ({"query": "kubernetes 集群可观测性监控"}, {"kubernetes/kubernetes", "derailed/k9s", "netdata/netdata"}),
    ({"query": "网页抓取爬虫 API"}, {"firecrawl/firecrawl"}),
    ({"query": "深度学习图像分类"}, {"pytorch/pytorch", "tensorflow/tensorflow"}),
]

# Cross-lingual slice: Chinese scenario -> projects whose descriptions/summaries
# are English-only. This is where the index was measurably weak (zh queries get
# outcompeted by any project that literally contains Chinese text).
XL_CASES: list[tuple[str, set[str]]] = [
    ({"query": "自托管的应用部署平台"}, {"dokku/dokku", "portainer/portainer", "kubernetes/kubernetes"}),
    ({"query": "密码管理工具"}, {"dani-garcia/vaultwarden"}),
    ({"query": "静态站点生成器"}, {"withastro/astro"}),
    ({"query": "端到端测试工具"}, {"microsoft/playwright"}),
]


def make_service(settings: Settings, engine):
    services = build_services(settings, engine)
    return RecommendService(
        settings,
        services.session_factory,
        services.embedding,
        services.vectors,
        services.analysis,
    )


def evaluate(service: RecommendService, label: str, cases=None) -> dict:
    cases = cases if cases is not None else CASES
    p3s: list[float] = []
    rr: list[float] = []
    latencies: list[float] = []
    per_query = []
    service.recommend("warmup query", top_k=3)  # model warm
    for query_dict, relevant in cases:
        query = query_dict["query"]
        start = time.perf_counter()
        hits = service.recommend(query, top_k=3)
        ms = (time.perf_counter() - start) * 1000
        names = [hit.project.full_name for hit in hits]
        hits_at_3 = len(set(names) & relevant) / 3
        first = next((i + 1 for i, n in enumerate(names) if n in relevant), None)
        p3s.append(hits_at_3)
        rr.append(1.0 / first if first else 0.0)
        latencies.append(ms)
        per_query.append((query, names, ms))
    return {
        "label": label,
        "p3": statistics.mean(p3s) if p3s else 0.0,
        "mrr": statistics.mean(rr) if rr else 0.0,
        "latency_avg": statistics.mean(latencies) if latencies else 0.0,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rerank", action="store_true", help="only the no-rerank baseline")
    args = parser.parse_args()

    base = dict(
        _env_file=None,
        hybrid_retrieval=True,
        llm_provider="rule",  # never spend tokens in an eval sweep
        llm_fallback_providers=[],
        asr_provider="none",
        tts_provider="none",
    )
    settings_engine = create_db_engine(Settings().database_url)
    init_db(settings_engine)

    modes: list[tuple[str, Settings]] = []
    modes.append(
        (
            "hybrid only (no rerank)",
            Settings(**base, embedder_provider=Settings().embedder_provider,
                     enable_llm_rerank=False, query_translation=False),
        )
    )
    if not args.no_rerank:
        modes.append(
            (
                "hybrid + bge rerank",
                Settings(**base, embedder_provider=Settings().embedder_provider,
                         enable_llm_rerank=True, rerank_provider="bge"),
            )
        )
        modes.append(
            (
                "hybrid + laya rerank",
                Settings(**base, embedder_provider=Settings().embedder_provider,
                         enable_llm_rerank=True, rerank_provider="laya"),
            )
        )
    # cross-lingual mode: CJK queries translated by the OFFLINE opus-mt model
    # (rule chain has no chat -> the translation chain falls to the offline
    # leg; no API cost, no dead-key retries)
    prod = Settings()
    modes.append(
        (
            "hybrid + zh->en translation",
            Settings(
                _env_file=None,
                hybrid_retrieval=True,
                enable_llm_rerank=False,
                query_translation=True,
                llm_provider="rule",
                llm_fallback_providers=[],
                offline_translator_model=prod.offline_translator_model,
                asr_provider="none",
                tts_provider="none",
                database_url=prod.database_url,
                embedder_provider=prod.embedder_provider,
                embedder_fallback_providers=prod.embedder_fallback_providers,
            ),
        )
    )

    results = []
    for label, settings in modes:
        settings.database_url = Settings().database_url
        service = make_service(settings, settings_engine)
        outcome = evaluate(service, label)
        xl_outcome = evaluate(service, label + " [xl]", XL_CASES)
        results.append(outcome)
        print(
            f"{outcome['label']:<28} P@3={outcome['p3']:.2f}  MRR={outcome['mrr']:.2f}  "
            f"latency={outcome['latency_avg']:.0f}ms"
        )
        print(
            f"{xl_outcome['label']:<28} P@3={xl_outcome['p3']:.2f}  MRR={xl_outcome['mrr']:.2f}"
        )
        xl_per_query = xl_outcome["per_query"]
        print()
    print("per-query (baseline):")
    for query, names, ms in results[0]["per_query"]:
        print(f"  [{ms:5.0f}ms] {query} -> {', '.join(names)}")
    print("per-query (cross-lingual, baseline config):")
    for query, names, ms in xl_per_query:
        print(f"  {query} -> {', '.join(names)}")
    settings_engine.dispose()


if __name__ == "__main__":
    main()
