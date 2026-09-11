# 关键类与函数说明

本文按模块列出关键类、函数及其职责与签名，便于精确定位实现。

## 1. 配置与入口

### `Settings`（[app/config.py](../app/config.py)）

pydantic-settings 基类，从环境变量/`.env` 读取全部配置。关键字段分组：

```python
class Settings(BaseSettings):
    # 应用/安全
    app_name, debug, secret_key, admin_token
    rate_limit_per_minute, max_upload_mb, log_file
    # 数据库
    database_url
    # GitHub
    github_client_id, github_client_secret, github_redirect_uri, github_token, github_api_base
    github_search_keywords, github_language, github_min_stars, github_per_page
    github_max_repos_per_keyword, github_max_analyze_per_run, github_fetch_readme, ...
    # LLM
    llm_provider, llm_fallback_providers, openai_api_key, anthropic_api_key, ...
    # Embedder / Vector store
    embedder_provider, embedder_fallback_providers, embedding_dimension, vector_store_backend, ...
    # 调度与自进化
    scheduler_enabled, fetch_interval_minutes, patrol_interval_minutes, evolution_*, analysis_workers
    # 语音
    asr_provider, asr_model_path, tts_provider, tts_model_dir, ...
```

- `_split_list()`：兼容 JSON 数组 / 逗号分隔字符串 / 列表的列表字段解析。
- `is_github_oauth_configured`（属性）：`client_id` 与 `client_secret` 是否都已配置。
- `get_settings()`：`@lru_cache` 单例。

### `create_app()` / `lifespan`（[app/main.py](../app/main.py)）

- `create_app(settings=None) -> FastAPI`：构建应用，挂载 `SessionMiddleware`、`CORSMiddleware`、`RateLimitMiddleware`，注册 `all_routers`，添加 `/` 与 `/info`。
- `lifespan(app)`：启动时建引擎、建表、`build_services`、启动调度器；关闭时停调度器、关客户端、释放引擎。

## 2. 数据库（[app/database.py](../app/database.py)）

- `create_db_engine(database_url) -> Engine`：创建引擎；SQLite 启用 `check_same_thread=False`、`StaticPool`（内存库）、WAL + busy_timeout。
- `create_sessionmaker(engine) -> sessionmaker`：`autoflush=False`、`expire_on_commit=False`。
- `init_db(engine)`：`Base.metadata.create_all` 建表。

## 3. ORM 模型（[app/models/](../app/models/)）

### `Project`（[models/project.py](../app/models/project.py)）

核心实体，字段分组：

```python
# 元数据
id, github_id(unique), full_name, url, description, stars, forks,
language, topics(JSON), readme_excerpt
# 时间戳
created_at, updated_at, pushed_at
# AI 分析结果
ai_summary, ai_use_cases(JSON), ai_tags(JSON), ai_dependencies(JSON)
# 向量与状态
embedding_id, analysis_status(pending/done/failed), last_analyzed_at
```

- `to_analysis_dict() -> dict`：抽取分析器需要的输入字段。

### `User` / `QueryHistory` / `FeedbackRecord`（[models/user.py](../app/models/user.py)）

- `User`：`github_user_id`、`username`、`access_token_encrypted`。
- `QueryHistory`：`user_id`、`query`、`top_k`、`result_count`。
- `FeedbackRecord`：`query`、`project_id`、`vote(up/down)`、`comment`、`source(web/voice/api)`。

## 4. 服务层（[app/services/](../app/services/)）

### `ServiceContainer` / `build_services`（[container.py](../app/services/container.py)）

```python
@dataclass
class ServiceContainer:
    settings, engine, session_factory
    github, analysis, embedding, vectors, ingest, recommend
    token_cipher, patrol, evolution, stt, tts
```

`build_services(settings, engine)` 按依赖顺序构建上述对象并返回容器。

### `SchedulerService`（[scheduler.py](../app/services/scheduler.py)）

- `start()`：注册 4 类任务（interval 触发）：
  - `fetch_github_trending`（默认 1440min）
  - `update_existing_projects`（默认 360min）
  - `patrol_analysis`（默认 10min，`next_run_time=now` 启动即跑）
  - `model_evolution`（默认 1440min，需 `evolution_enabled`）
- `_run_fetch/_run_refresh/_run_patrol/_run_evolution`：各自 `asyncio.to_thread` 调用对应服务，异常只记录不中断。
- `shutdown()`：停调度器。

### `GitHubService`（[github_service.py](../app/services/github_service.py)）

- `search_repositories(keyword, *, language, min_stars, per_page, max_repos, sort) -> list[dict]`：分页搜索并归一化。
- `get_repository(full_name) -> dict`：单仓库详情。
- `get_readme_excerpt(full_name) -> str | None`：抓 README 前 N 字符。
- `_request(method, path, ...) -> httpx.Response`：带 403/429 限流退避与指数重试的核心请求。
- `_is_rate_limited()` / `_retry_after_seconds()`：从 `x-ratelimit-*` / `retry-after` 头判断限流与等待时长。
- `_normalize_repo(item) -> dict`：把 GitHub API 原始条目归一为统一结构。
- 异常层级：`GitHubServiceError → GitHubAPIError / GitHubRateLimitError`。

### `AnalysisService`（[analysis_service.py](../app/services/analysis_service.py)）

- `provider_name`（属性）：第一个 `available()` 的分析器名，否则 `"none"`。
- `analyze(project_data) -> AnalysisResult`：按优先级逐一尝试，任一失败捕获并降级，全部失败抛 `AnalysisError`。

### `EmbeddingService`（[embedding_service.py](../app/services/embedding_service.py)）

- `__init__` 启动时按 `embedder_provider + fallbacks` 解析出唯一 embedder（候选去重）。
- `embed(text)` / `embed_batch(texts)`。
- 模块级函数：
  - `project_embedding_id(project_id) -> "project-<id>"`
  - `parse_embedding_id(embedding_id) -> int | None`
  - `build_project_text(project) -> str`：拼接 full_name + description + summary + tags + topics + language。

### `IngestService`（[ingest_service.py](../app/services/ingest_service.py)）

- `run_scheduled_fetch(keywords=None) -> dict`：按关键词检索并入库，返回 `IngestStats.as_dict()`。
- `_next_keyword_slice() -> list[str]`：`keywords_per_fetch>0` 时轮换切片。
- `ingest_repositories(repos) -> dict`：按 `github_id` upsert；描述变化重置为 `pending` + 清 `embedding_id`。
- `refresh_existing(limit=None) -> dict`：刷新已入库项目元数据。
- `process_backlog() -> dict`：分析 pending + 嵌入未嵌入项目（供巡检调用）。
- `requeue_failed() -> int`：failed → pending。
- `_analyze_pending(session, stats)`：README 顺序抓取 → `ThreadPoolExecutor` 并行分析 → 回写结果。
- `_embed_pending(session, stats)`：`build_project_text` → embed → `vectors.upsert`。

### `RecommendService`（[recommend_service.py](../app/services/recommend_service.py)）

- `recommend(query, top_k=5) -> list[RecommendHit]`：主入口。
- `_rerank(query, results)`：`ENABLE_LLM_RERANK` 时用可用分析器重排并生成理由。
- `_record_history(query, top_k, result_count)`：写入 `QueryHistory`（失败不影响推荐）。
- `RecommendHit` 数据类：`project / score / dependencies / reason`。

### `PatrolService`（[patrol_service.py](../app/services/patrol_service.py)）

- `run_patrol() -> dict`：requeue failed → process backlog → reconcile vectors，产出 `PatrolReport`。
- `reconcile_vectors(limit=None) -> int`：用 `vectors.existing_ids()` 找“丢了”的向量并重建。

### `EvolutionService`（[evolution_service.py](../app/services/evolution_service.py)）

- `run_cycle(force=False, max_samples=None, epochs=None) -> dict`：七步进化主循环，产出 `EvolutionReport`。
- `_harvest_db()` / `_collect_jsonl_rows()`：从 DB 与 `training/data/raw_repos*.jsonl` 收割训练数据。
- `_canary(model_dir) -> bool`：现场加载并推理，验证模型可用。
- `_evaluate_model(model_dir) -> dict`：调用 `training/golden_eval` 得 micro-F1 / recall / 门禁失败项。
- `_train_staging(data_dir, model_dir, epochs)`：独立子进程跑 `train.py` + `retune_thresholds.py`。
- `_repair_deployed_model()`：线上模型金丝雀失败则回滚最近可用存档。
- `_promote(...)`：原子换目录 + 写 `version.json` + 分析器/嵌入器热重载 + 向量重建。
- `_reembed_all()`：晋升后用新向量空间重建（上限 `EVOLUTION_MAX_REEMBED`，余量由巡检补）。

## 5. LLM 分析器（[app/llm/](../app/llm/)）

### 抽象与数据（[base.py](../app/llm/base.py)）

```python
@dataclass
class AnalysisResult:
    summary: str
    use_cases: list[str]
    tags: list[str]
    dependencies: list[str]
    provider: str

class LLMAnalyzer(ABC):
    name: str
    def analyze(self, project_data) -> AnalysisResult  # abstract
    def available(self) -> bool   # 默认 True
    def rank_candidates(self, query, candidates) -> tuple[list[int], str] | None  # 默认 None
```

模块级函数：`build_analysis_prompt()`、`build_rerank_prompt()`、`extract_json()`（容错 JSON 解析）、`result_from_json()`。

### 各实现

| 类 | `name` | 分析路径 | `rank_candidates` |
| --- | --- | --- | --- |
| `TrainedModelAnalyzer` | `trained` | 分类头推理 + 抽取式摘要 + 依赖映射（结构保证合法） | 不支持（返回 None） |
| `OpenAIAnalyzer` | `openai` | Chat Completions → JSON | 支持 |
| `AnthropicAnalyzer` | `anthropic` | Messages API → JSON | 支持 |
| `LocalModelAnalyzer` | `local` | HF pipeline 文本生成 → JSON | 不支持 |
| `RuleBasedAnalyzer` | `rule` | 关键词映射表（永不失败） | 不支持 |

> 关键细节：`TrainedModelAnalyzer.analyze()` 中 `use_cases` 采用“**分类头 ∪ 规则 taxonomy 集成**”，标签采用“**模型 tags ∪ 仓库 topics**”合并，摘要优先用 description、其次 README 抽取式摘要。

### `build_analyzer_chain`（[llm/__init__.py](../app/llm/__init__.py)）

按 `[llm_provider, *llm_fallback_providers]` 构建去重降级链；空链兜底 `RuleBasedAnalyzer`。

## 6. 向量化器（[app/embedders/](../app/embedders/)）

`Embedder` 抽象（[base.py](../app/embedders/base.py)）：`embed(texts) -> list[list[float]]`（抽象）、`embed_text(text)`。

| 类 | `name` | 说明 |
| --- | --- | --- |
| `TrainedEmbedder` | `trained` | 复用自训练模型 encoder，`_reload_if_swapped()` 支持热重载，`@torch.no_grad` + `MODEL_SWAP_LOCK` 串行化 |
| `SentenceTransformerEmbedder` | `sentence_transformers` | 本地 sentence-transformers，`normalize_embeddings=True` |
| `OpenAIEmbedder` | `openai` | OpenAI Embedding API，自动读取返回维度 |
| `HashingTfidfEmbedder` | `tfidf` | SHA-256 分桶 + 次线性词频 + L2 归一化（确定性、零依赖） |

`create_embedder(name, settings)`（[embedders/__init__.py](../app/embedders/__init__.py)）按名称实例化。

## 7. 向量库（[app/vectorstore/](../app/vectorstore/)）

`VectorStore` 抽象（[base.py](../app/vectorstore/base.py)）。

```python
def upsert(ids, embeddings, documents=None, metadatas=None)
def query(embedding, top_k) -> list[tuple[str, float]]   # [(id, similarity)]
def delete(ids)
def count() -> int
def existing_ids(ids) -> set[str]   # 供巡检修漂移
```

- `ChromaVectorStore`：`PersistentClient` + `hnsw:space=cosine`；`query` 返回 `1 - distance`。
- `MemoryVectorStore`：字典存储 + 余弦相似度；`_check_dimension()` 维数校验，防止模型切换后混用。
- `create_vector_store(settings)`：`chroma` 初始化失败或未安装自动回退 `memory`。

## 8. 自训练模型（[app/ml/](../app/ml/)）

### `RepoAnalyzerModel`（[repo_analyzer.py](../app/ml/repo_analyzer.py)）

```python
class RepoAnalyzerModel(nn.Module):
    def __init__(self, base_model, num_tags, num_use_cases, dropout=0.1, encoder=None)
    def forward(self, input_ids, attention_mask) -> (tag_logits, use_case_logits)
```

结构：多语言 MiniLM encoder → mean pooling → Dropout → 两个 `nn.Linear` 分类头。

模块级函数：
- `save_repo_analyzer(model_dir, model, tokenizer, tags, use_cases, tag_thresholds, use_case_thresholds, max_len, metrics)`：保存权重 + 词表 + encoder 架构 + `analyzer_config.json`。
- `load_repo_analyzer(model_dir, device=None)`：离线重建（`local_files_only=True`）并加载权重，返回 `(model, tokenizer, config, device)`。
- `predict_labels(logits, labels, thresholds, top_k, min_prob)`：sigmoid + 逐标签阈值，兜底取最高概率标签。

### `USE_CASE_TAXONOMY` / `match_use_cases`（[taxonomy.py](../app/ml/taxonomy.py)）

30 类中文使用场景标签；`match_use_cases(topics, text) -> list[str]` 先按话题集合、再按关键词子串匹配。

### `extractive_summary`（[extractive.py](../app/ml/extractive.py)）

`extractive_summary(full_name, readme, max_chars=220, max_sentences=2) -> str`：句子打分（关键词重叠 + 位置先验 + 长度），过滤 boilerplate/表格/图片/HTML。

### `build_dataset`（[dataset_builder.py](../app/ml/dataset_builder.py)）

`build_dataset(repos, output_dir, *, min_topic_freq, max_tags, max_samples, seed, val_frac, test_frac) -> DatasetStats`：
清洗去重 → 话题计数定词表 → 弱监督标注（topics 标签 + `match_use_cases` 使用场景）→ 打散切分 → 写出 `train/val/test.jsonl` + `label_vocab.json`。

### `model_registry`（[model_registry.py](../app/ml/model_registry.py)）

- `MODEL_SWAP_LOCK`：全局线程锁，串行化模型交换与推理。
- `current_version(models_dir)` / `stamp(models_dir)` / `read_version_info(models_dir)`。

## 9. 工具函数（[app/utils/](../app/utils/)）

| 函数/类 | 说明 |
| --- | --- |
| `TokenCipher.encrypt/decrypt` | Fernet 加解密；`derive_key(secret)` 从 SECRET_KEY 派生密钥 |
| `jwt.encode/decode` | 纯标准库 HS256 JWT（`iat`/`exp`，hmac 校验） |
| `configure_logging(debug, log_file)` | Loguru 控制台 + 文件（10MB 轮转、保留 5 份） |
| `utcnow()` | naive UTC 当前时间 |
| `parse_github_datetime(value)` | 解析 GitHub ISO 时间字符串为 naive UTC |

## 10. 语音（[app/speech/](../app/speech/)）

- `FasterWhisperSTT.transcribe(audio_bytes, filename) -> str`：`faster_whisper` CPU int8 转写（VAD 过滤，返回纯文本）。
- `SherpaOnnxTTS.synthesize(text) -> bytes`：sherpa-onnx + Melo-TTS 合成 WAV（16-bit PCM）。
- `samples_to_wav(samples, sample_rate)`：float32 采样 → WAV 字节。
- `build_stt/build_tts(settings)`：按配置构建，禁用或不可用返回 `None`。