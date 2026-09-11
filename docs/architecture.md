# 整体架构

## 1. 分层视图

系统分为五层，上层依赖下层，跨层通过 `ServiceContainer` 注入解耦：

```text
┌────────────────────────────────────────────────────────────────────┐
│                        入口层 (Entry Points)                        │
│   app/main.py (FastAPI, HTTP API + 调度器)                         │
│   app/worker.py (纯后台 24/7 worker)                               │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                        API 层 (HTTP Interface)                     │
│   app/api/routes/*      路由：health/auth/recommend/projects/admin │
│                          /voice/feedback                           │
│   app/api/dependencies.py  依赖注入 + 管理员校验                   │
│   app/api/middleware.py    每 IP 限流中间件                        │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                        服务层 (Business Logic)                     │
│   IngestService   RecommendService   AnalysisService               │
│   EmbeddingService   GitHubService   PatrolService                 │
│   EvolutionService   SchedulerService                              │
│   container.py  —— 组装所有服务（依赖注入容器）                    │
└────────────────────────────────────────────────────────────────────┘
                          │                │                │
            ┌─────────────┘                │                └─────────────┐
            ▼                              ▼                              ▼
┌───────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│      可插拔后端         │   │       数据层 (Storage)     │   │      外部依赖            │
│  llm/       (分析器)    │   │  SQLAlchemy 引擎 →        │   │  GitHub REST API         │
│  embedders/ (向量化)    │   │  PostgreSQL / SQLite      │   │  OpenAI / Anthropic API   │
│  vectorstore/(向量库)   │   │  Chroma / 内存向量库      │   │  HuggingFace (训练)       │
│  speech/    (语音)      │   └──────────────────────────┘   └──────────────────────────┘
└───────────────────────┘
```

## 2. 核心抽象（面向接口）

代码通过三个 **抽象基类（ABC）** 实现可插拔，替换实现不触碰业务逻辑：

| 抽象 | 基类位置 | 关键方法 | 实现 |
| --- | --- | --- | --- |
| LLM 分析器 | [llm/base.py](../app/llm/base.py) | `analyze()` / `available()` / `rank_candidates()` | `trained` / `openai` / `anthropic` / `local` / `rule` |
| 向量化器 | [embedders/base.py](../app/embedders/base.py) | `embed()` / `embed_text()` | `trained` / `sentence_transformers` / `openai` / `tfidf` |
| 向量库 | [vectorstore/base.py](../app/vectorstore/base.py) | `upsert()` / `query()` / `delete()` / `count()` / `existing_ids()` | `chroma` / `memory` |

三个工厂负责按配置实例化：
- `build_analyzer_chain()` — [llm/__init__.py](../app/llm/__init__.py)
- `create_embedder()` — [embedders/__init__.py](../app/embedders/__init__.py)
- `create_vector_store()` — [vectorstore/__init__.py](../app/vectorstore/__init__.py)

## 3. 启动流程（lifespan）

`create_app()` 返回 `FastAPI` 实例，`lifespan` 在进程启动/关闭时执行：

```text
Settings() ──▶ configure_logging()
                 │
                 ├── create_db_engine(database_url)
                 ├── init_db(engine)                # Base.metadata.create_all
                 ├── build_services(settings, engine)   # 组装全部服务
                 ├── (可选) SchedulerService(services).start()  # APScheduler 定时任务
                 └── 注入 app.state.services / app.state.scheduler
                                                       │
                                                     yield ──▶ 响应请求
                                                       │
                                        shutdown: 停调度器 → 关 GitHub 客户端 → 释放引擎
```

两条启动路径共享同一套初始化逻辑：

- **形态 A**：`uvicorn app.main:app` —— API + 调度器同进程（部署主形态）；
- **形态 B**：`python -m app.worker` —— 只跑调度器，不开 HTTP，与 A 可并存并共享数据库。

## 4. 数据流

### 4.1 检索入库流水线（Ingest）

```text
SchedulerService._run_fetch / POST /api/admin/trigger-fetch
        │
        ▼
IngestService.run_scheduled_fetch(keywords)
        │  关键词轮换切片 _next_keyword_slice()
        ▼
GitHubService.search_repositories(keyword)      # httpx 直连 REST，429/403 指数退避
        │
        ▼
ingest_repositories(repos)                       # 按 github_id upsert
        │  ├─ 新项目 → Project(analysis_status="pending")
        │  └─ 已存在 → 更新 stars/forks/pushed_at；描述变化则重置为 pending 并清 embedding_id
        ▼
_analyze_pending()                               # README 顺序抓取 → 多线程并行分析
        │  └─ AnalysisService.analyze() → AnalysisResult
        ▼
_embed_pending()                                 # build_project_text → embed → vectors.upsert
        │  └─ project.embedding_id = "project-<id>"
        ▼
analysis_status: pending → done / failed
```

### 4.2 场景推荐流水线（Recommend）

```text
POST /api/recommend  {"query": "...", "top_k": 5}
        │
        ▼
RecommendService.recommend(query, top_k)
        │  嵌入查询向量
        ▼
vectors.query(query_vector)                       # 返回 [(embedding_id, similarity)]
        │  解析 embedding_id → project_id → 回查 PostgreSQL 补全详情
        ▼
(可选) _rerank()                                 # ENABLE_LLM_RERANK=true 时用 LLM 重排 + 理由
        ▼
_record_history()                                # 记录 query_history（可选，永不失败）
        ▼
返回 RecommendHit[]（project + score + dependencies + reason）
```

### 4.3 巡检自愈（Patrol，短周期）

```text
SchedulerService._run_patrol（每 10 分钟，启动即执行一次）
        ▼
PatrolService.run_patrol()
        ├─ requeue_failed()          # failed → pending 重试
        ├─ process_backlog()         # 处理待分析积压 + 待嵌入
        └─ reconcile_vectors()       # 修复向量漂移（如内存库重启丢失）
```

### 4.4 自我进化（Evolution，长周期）

```text
SchedulerService._run_evolution / POST /api/admin/run-evolution
        ▼
EvolutionService.run_cycle()
        1. repair:  金丝雀探测线上模型 → 损坏则回滚到最近可用存档
        2. harvest: 数据库抓取项目 + 采集 jsonl
        3. build:   app.ml.dataset_builder.build_dataset()
        4. train:   独立子进程 training/train.py + retune_thresholds.py
        5. evaluate: 金标准集 micro-F1 + 门禁
        6. promote: 门禁全过 且 F1 严格优于冠军 → 原子换目录 + 热重载 + 向量重建
                   否则丢弃挑战者（退化模型永不发布）
        7. finish:  写 version.json / evolution_history.json
```

## 5. 降级策略矩阵

任何外部依赖不可用时，服务都能照常启动并给出可用结果：

| 环节 | 首选 | 降级 1 | 降级 2（兜底） |
| --- | --- | --- | --- |
| LLM 分析 | 自训练模型（离线、零成本） | OpenAI / Anthropic / 本地小模型 | 规则分析器（关键词映射，永不失败） |
| 向量化 | 自训练 encoder / sentence-transformers | OpenAI Embedding | 纯 Python 哈希 TF-IDF（零依赖） |
| 向量库 | Chroma（持久化） | — | 内存向量库（重启丢失，靠巡检重建） |
| 语音识别/合成 | faster-whisper / Melo-TTS | — | 接口返回 503 并提示（文本接口不受影响） |
| GitHub API | 带 token（5000 次/h） | 匿名（60 次/h） | 限流指数退避重试，超限跳过本次任务 |

**关键设计约束**：向量化模型在**启动时一次性选定**（而非每次调用降级），因为不同模型维度/语义空间不兼容，混用会破坏相似度检索；切换模型后需重建索引。降级事件通过 Loguru 记录，`GET /health` 可查看当前实际生效的 provider。