# 主要模块职责

本文按包/目录逐项说明每个模块的职责与关键文件。所有路径均相对仓库根目录。

## 1. `app/` — 应用主包

| 文件 | 职责 |
| --- | --- |
| [main.py](../app/main.py) | FastAPI 入口。`create_app()` 工厂 + `lifespan` 装配；挂载 Session/CORS/限流中间件；注册全部路由；`/` 返回落地页。 |
| [worker.py](../app/worker.py) | 独立的 24/7 后台 worker（不开 HTTP），仅启动调度器。 |
| [config.py](../app/config.py) | `Settings` 配置中心（pydantic-settings），从环境变量/`.env` 加载；集中定义全部可调项。 |
| [database.py](../app/database.py) | SQLAlchemy 引擎/会话创建；SQLite 启用 WAL + busy_timeout；`init_db()` 建表。 |

## 2. `app/api/` — HTTP 接口层

| 文件 | 职责 |
| --- | --- |
| [routes/__init__.py](../app/api/routes/__init__.py) | 路由注册表 `all_routers`，汇总 7 个子路由。 |
| [routes/health.py](../app/api/routes/health.py) | `GET /health` 健康检查，含各组件实际 provider 与降级状态。 |
| [routes/auth.py](../app/api/routes/auth.py) | `GET /auth/github/login` / `/callback`，GitHub OAuth 登录、换取并加密 token、签发 JWT。 |
| [routes/recommend.py](../app/api/routes/recommend.py) | `POST /api/recommend` 场景推荐。 |
| [routes/projects.py](../app/api/routes/projects.py) | `GET /api/projects` 列表、`GET /api/projects/{id}` 详情。 |
| [routes/admin.py](../app/api/routes/admin.py) | 管理员接口：手动触发检索/巡检/进化；状态统计；反馈聚合报表。 |
| [routes/voice.py](../app/api/routes/voice.py) | 语音接口：`/transcribe`、`/speak`、`/ask`、`/status`、`/page`。 |
| [routes/feedback.py](../app/api/routes/feedback.py) | `POST /api/feedback` 收集 👍/👎，作为优化循环数据源。 |
| [dependencies.py](../app/api/dependencies.py) | `get_services()` 依赖注入；`require_admin()` 管理员 Header 校验（常量时间比较）。 |
| [middleware.py](../app/api/middleware.py) | `RateLimitMiddleware` 每 IP 滑动窗口限流（默认 120 次/分）。 |

## 3. `app/services/` — 业务逻辑层（核心）

| 文件 | 职责 |
| --- | --- |
| [container.py](../app/services/container.py) | `ServiceContainer` 数据类 + `build_services()` 组装所有服务（依赖注入容器）。 |
| [github_service.py](../app/services/github_service.py) | GitHub REST 客户端：搜索/详情/README 抓取，403/429 限流退避与重试。 |
| [analysis_service.py](../app/services/analysis_service.py) | 分析降级链：按优先级尝试多个分析器，任一失败自动降级。 |
| [embedding_service.py](../app/services/embedding_service.py) | 启动时解析 embedder（含兜底）；`build_project_text()` 拼接待嵌入文本；embedding_id 编解码。 |
| [ingest_service.py](../app/services/ingest_service.py) | 检索→入库→分析→嵌入流水线；关键词轮换；多线程分析与积压处理。 |
| [recommend_service.py](../app/services/recommend_service.py) | 场景推荐：语义检索 + 回查详情 +（可选）LLM 重排 + 查询历史记录。 |
| [patrol_service.py](../app/services/patrol_service.py) | 24/7 自愈巡检：重试失败 + 处理积压 + 修复向量漂移。 |
| [evolution_service.py](../app/services/evolution_service.py) | 自我进化循环：收割/训练/门禁/晋升/回滚/向量重建。 |
| [scheduler.py](../app/services/scheduler.py) | APScheduler 封装：注册 fetch/refresh/patrol/evolution 四类定时任务。 |

## 4. `app/models/` 与 `app/schemas/` — 数据模型

### ORM 模型（SQLAlchemy）

| 模型 | 表 | 说明 |
| --- | --- | --- |
| `Base` | — | `DeclarativeBase` 基类。 |
| `Project` | `projects` | GitHub 仓库及其 AI 分析结果；`analysis_status` 状态机 + `embedding_id`。 |
| `User` | `users` | OAuth 登录用户，`access_token_encrypted` 为 Fernet 密文。 |
| `QueryHistory` | `query_history` | 每次推荐查询的日志（驱动调优）。 |
| `FeedbackRecord` | `feedback` | 外部用户点赞/点踩记录（驱动优化闭环）。 |

### Pydantic Schema（请求/响应）

| 文件 | 用途 |
| --- | --- |
| [project.py](../app/schemas/project.py) | `ProjectOut`（项目序列化输出）。 |
| [recommend.py](../app/schemas/recommend.py) | `RecommendRequest` / `RecommendationItem` / `RecommendResponse`。 |
| [admin.py](../app/schemas/admin.py) | 触发请求/状态响应体。 |
| [user.py](../app/schemas/user.py) | `UserOut` / `TokenResponse`。 |

## 5. `app/llm/` — 可插拔 LLM 分析后端

| 文件 | 职责 |
| --- | --- |
| [base.py](../app/llm/base.py) | `LLMAnalyzer` 抽象基类、`AnalysisResult` 数据类、Prompt 构建、宽松 JSON 解析。 |
| [__init__.py](../app/llm/__init__.py) | `build_analyzer_chain()` 工厂，按优先级构建降级链并去重。 |
| [trained_analyzer.py](../app/llm/trained_analyzer.py) | 本地自训练模型分析器（分类头 + 抽取式摘要 + 依赖映射，保证结构合法）。 |
| [openai_llm.py](../app/llm/openai_llm.py) | OpenAI（及兼容端点）分析器 + 重排。 |
| [anthropic_llm.py](../app/llm/anthropic_llm.py) | Anthropic (Claude) 分析器 + 重排。 |
| [local_llm.py](../app/llm/local_llm.py) | 本地小生成模型（HF transformers pipeline），解析失败即降级。 |
| [rule_based.py](../app/llm/rule_based.py) | 规则分析器（关键词映射表，永不失败）+ 语言/标签→依赖映射表。 |

## 6. `app/embedders/` — 可插拔向量化后端

| 文件 | 职责 |
| --- | --- |
| [base.py](../app/embedders/base.py) | `Embedder` 抽象基类。 |
| [__init__.py](../app/embedders/__init__.py) | `create_embedder()` 工厂。 |
| [trained_embedder.py](../app/embedders/trained_embedder.py) | 复用自训练模型的 encoder 做语义检索（含热重载）。 |
| [sentence_transformer_embedder.py](../app/embedders/sentence_transformer_embedder.py) | 本地 sentence-transformers 模型。 |
| [openai_embedder.py](../app/embedders/openai_embedder.py) | OpenAI Embedding API。 |
| [tfidf_embedder.py](../app/embedders/tfidf_embedder.py) | 纯 Python 哈希 TF-IDF 兜底（SHA-256 分桶，跨运行稳定）。 |

## 7. `app/vectorstore/` — 向量库

| 文件 | 职责 |
| --- | --- |
| [base.py](../app/vectorstore/base.py) | `VectorStore` 抽象基类 + `VectorStoreError`/`VectorDimensionError`。 |
| [__init__.py](../app/vectorstore/__init__.py) | `create_vector_store()` 工厂（chroma 失败自动回退 memory）。 |
| [chroma_store.py](../app/vectorstore/chroma_store.py) | Chroma 持久化实现（cosine 空间，`1 - distance` 转相似度）。 |
| [memory_store.py](../app/vectorstore/memory_store.py) | 内存向量库（余弦相似度，维数校验）。 |

## 8. `app/ml/` — 自训练模型

| 文件 | 职责 |
| --- | --- |
| [repo_analyzer.py](../app/ml/repo_analyzer.py) | `RepoAnalyzerModel`（多语言 MiniLM + 双头多标签分类）；保存/加载/标签预测。 |
| [taxonomy.py](../app/ml/taxonomy.py) | 30 类“使用场景”中文标签体系；`match_use_cases()` 话题/关键词匹配。 |
| [extractive.py](../app/ml/extractive.py) | 抽取式摘要：README 句子打分（克制 boilerplate）。 |
| [dataset_builder.py](../app/ml/dataset_builder.py) | 原始仓库行 → train/val/test jsonl + 标签词表（弱监督）。 |
| [model_registry.py](../app/ml/model_registry.py) | 版本/时间戳管理、`MODEL_SWAP_LOCK` 热重载锁。 |

## 9. `app/speech/` — 离线语音

| 文件 | 职责 |
| --- | --- |
| [__init__.py](../app/speech/__init__.py) | `build_stt()` / `build_tts()` 工厂。 |
| [asr.py](../app/speech/asr.py) | `FasterWhisperSTT` 离线语音识别（CPU int8，中英）。 |
| [tts.py](../app/speech/tts.py) | `SherpaOnnxTTS` 离线语音合成（Melo-TTS 中英混读）+ `samples_to_wav()`。 |

## 10. `app/utils/` — 工具

| 文件 | 职责 |
| --- | --- |
| [crypto.py](../app/utils/crypto.py) | `TokenCipher`：Fernet 对称加密（存储 GitHub token）；`derive_key()`。 |
| [jwt.py](../app/utils/jwt.py) | 纯标准库 HS256 JWT 实现（`encode`/`decode`）。 |
| [logging.py](../app/utils/logging.py) | Loguru 日志配置（控制台 + 文件轮转）。 |
| [timeutil.py](../app/utils/timeutil.py) | `utcnow()` / `parse_github_datetime()`（统一 naive UTC）。 |

## 11. `training/` — 自训练流水线

| 文件/目录 | 职责 |
| --- | --- |
| [README.md](../training/README.md) | 训练原理、步骤、性能参考、金标准评测。 |
| [collect_data.py](../training/collect_data.py) | 采集原始仓库数据（jsonl）。 |
| [prepare_data.py](../training/prepare_data.py) | 构建数据集（调用 `app/ml/dataset_builder`）。 |
| [train.py](../training/train.py) | 训练挑战者模型（子进程可调）。 |
| [retune_thresholds.py](../training/retune_thresholds.py) | 逐头全局阈值搜索。 |
| [evaluate.py](../training/evaluate.py) | 测试集评估。 |
| [golden_eval.py](../training/golden_eval.py) | 金标准门禁（独立手标 107 例 + 结构检查）。 |
| [download_speech_models.py](../training/download_speech_models.py) | 下载语音模型（whisper-small / Melo-TTS）。 |
| [evals/golden_set.json](../training/evals/golden_set.json) | 手工评测集（90 名仓库 + 20 对抗用例）。 |

## 12. `scripts/`、`.github/workflows/` 与 `tests/`

| 文件/目录 | 职责 |
| --- | --- |
| [scripts/integration_test.py](../scripts/integration_test.py) | 远程集成测试（针对公网 URL）。 |
| [scripts/pull_model_release.py](../scripts/pull_model_release.py) | 拉取最新模型 Release 并本地复检晋升。 |
| [scripts/start_api.bat](../scripts/start_api.bat) / [start_worker.bat](../scripts/start_worker.bat) | Windows 启动脚本。 |
| [scripts/register_autostart.bat](../scripts/register_autostart.bat) | 开机自启注册。 |
| [scripts/run_evolution_once.py](../scripts/run_evolution_once.py) | 手动跑一轮进化。 |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | 每次 push 跑全量测试。 |
| [.github/workflows/evolution.yml](../.github/workflows/evolution.yml) | 夜间云端训练 + 金标准门禁 + 发布 Release 资产。 |
| [.github/workflows/deploy.yml](../.github/workflows/deploy.yml) | 公网临时部署 + 远程集成测试。 |
| [tests/](../tests/) | 单元 + API 集成测试（`conftest.py` 提供全降级 settings 与 TestClient）。 |