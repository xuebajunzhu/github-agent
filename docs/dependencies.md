# 依赖关系

## 1. 内部依赖（模块间调用关系）

依赖方向总体为：**入口/API → 服务层 → 可插拔后端 / 数据层**，服务之间通过 `build_services()` 显式注入。

```text
main.py / worker.py
   │  create_app → build_services(settings, engine)
   ▼
ServiceContainer (container.py)
   ├─ GitHubService(settings)                          ── httpx → GitHub API
   ├─ AnalysisService( build_analyzer_chain(settings) )
   │        └─ llm/: TrainedModelAnalyzer / OpenAIAnalyzer / Anthropic / Local / Rule
   ├─ EmbeddingService(settings)
   │        └─ embedders/: TrainedEmbedder / SentenceTransformer / OpenAI / HashingTfidf
   ├─ vectors = create_vector_store(settings)
   │        └─ vectorstore/: ChromaVectorStore / MemoryVectorStore
   ├─ ingest   = IngestService(settings, session_factory, github, analysis, embedding, vectors)
   ├─ recommend= RecommendService(settings, session_factory, embedding, vectors, analysis)
   ├─ patrol   = PatrolService(settings, session_factory, ingest, embedding, vectors)
   ├─ evolution= EvolutionService(settings, session_factory, embedding, vectors, analysis)
   ├─ token_cipher = TokenCipher(fernet_key, secret_key)
   ├─ stt/tts = build_stt/build_tts(settings)
   └─ scheduler = SchedulerService(services)      ← 由 main/worker 在 start 时创建
```

### 服务间关键依赖细节

| 服务 | 依赖 |
| --- | --- |
| `IngestService` | `GitHubService`（检索）、`AnalysisService`（分析）、`EmbeddingService`（嵌入）、`VectorStore`（写向量） |
| `RecommendService` | `EmbeddingService`（查询向量）、`VectorStore`（检索）、`AnalysisService`（可选重排） |
| `PatrolService` | `IngestService`（积压/重试）、`EmbeddingService` + `VectorStore`（修向量漂移） |
| `EvolutionService` | `EmbeddingService` + `VectorStore`（晋升后重建向量）、`AnalysisService`（热重载分析器）、`app.ml.*`、子进程 `training/train.py` 等 |
| `SchedulerService` | `IngestService` / `PatrolService` / `EvolutionService` |

### LLM 分析器与 ML 的依赖

- `TrainedModelAnalyzer` 依赖 `app.ml.repo_analyzer`（`load_repo_analyzer`/`predict_labels`）、`app.ml.extractive`、`app.ml.taxonomy`、`app.ml.model_registry`、以及 `app.llm.rule_based` 中的依赖映射表。
- `TrainedEmbedder` 依赖 `app.ml.repo_analyzer`、`app.ml.model_registry`。
- `dataset_builder` 依赖 `app.ml.taxonomy.match_use_cases`。

## 2. 外部运行时依赖

依赖分四层，按需安装。

### 2.1 核心依赖（[requirements.txt](../requirements.txt)）

| 包 | 用途 |
| --- | --- |
| `fastapi` / `uvicorn[standard]` | Web 框架与 ASGI 服务器 |
| `sqlalchemy` | ORM（PostgreSQL/SQLite） |
| `pydantic` / `pydantic-settings` | 数据校验与配置 |
| `httpx` | GitHub REST 客户端（同步/异步） |
| `loguru` | 日志 |
| `APScheduler` | 定时任务（24/7 循环） |
| `Authlib` | GitHub OAuth |
| `itsdangerous` | Starlette Session 中间件的签名依赖 |
| `cryptography` | Fernet 加密（存储 token） |
| `openai` / `anthropic` | 云端 LLM 分析器 |
| `psycopg2-binary` | PostgreSQL 驱动 |
| `python-multipart` | 文件上传（语音） |
| `numpy` | 数值计算（向量/语音） |

### 2.2 ML 依赖（[requirements-ml.txt](../requirements-ml.txt)）

| 包 | 用途 |
| --- | --- |
| `chromadb` | 持久化向量库 |
| `sentence-transformers` | 本地向量化模型 |
| `transformers` | 本地 LLM / 自训练模型 |
| `torch` | 模型推理/训练 |

### 2.3 语音依赖（[requirements-speech.txt](../requirements-speech.txt)）

| 包 | 用途 |
| --- | --- |
| `faster-whisper` | 离线语音识别（ASR） |
| `sherpa-onnx` | 离线语音合成（TTS） |
| `python-multipart` | 音频上传 |

### 2.4 开发/训练依赖

| 文件 | 用途 |
| --- | --- |
| [requirements-dev.txt](../requirements-dev.txt) | `pytest`（测试） |
| [training/requirements-training.txt](../training/requirements-training.txt) | 训练流水线依赖（torch/transformers 等） |

## 3. 外部服务依赖

| 服务 | 用途 | 是否必需 |
| --- | --- | --- |
| GitHub REST API | 仓库检索 / 详情 / README / OAuth | 检索必需（有降级退避，无 token 限额低） |
| OpenAI API | 分析/向量化/重排（可选 provider） | 否 |
| Anthropic API | 分析/重排（可选 provider） | 否 |
| HuggingFace Hub | 下载预训练模型（训练时） | 启动推理可完全离线（`local_files_only`） |
| PostgreSQL | 生产数据库（Docker Compose） | 否（默认 SQLite） |

## 4. 降级依赖关系汇总

任何可选依赖缺失都会自动走降级路径，**不影响服务启动**：

```text
无 torch / 无 trained 模型      → LLM 落到 rule；Embedder 落到 tfidf
无 chromadb 或初始化失败        → 向量库落到 memory
无 OpenAI/Anthropic key         → 相关分析器 available()=False，被链跳过
无语音模型 / 无语音依赖         → build_stt/tts 返回 None，语音接口 503
无 GitHub token                 → 匿名限额 60 次/h（刷新/README 建议关闭）
```