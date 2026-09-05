# GitHub 开源项目智能检索与分析 Agent

定时从 GitHub 检索开源项目，用可插拔的大模型自动生成摘要、场景与标签，向量化入库；用户输入具体场景时，通过语义搜索推荐相关项目及其依赖。

全部核心组件采用成熟开源方案：FastAPI + SQLAlchemy + APScheduler + httpx + Chroma + sentence-transformers，LLM 支持 OpenAI / Anthropic / 本地模型 / 规则兜底四级降级。

## 核心特性

- **定时检索**：APScheduler 周期性按关键词 + 星标数搜索 GitHub 仓库，并刷新已入库项目。
- **AI 分析**：自动生成 `summary` / `use_cases` / `tags` / `dependencies`，支持五级 provider：**本地训练模型（默认，零成本）** → OpenAI/Anthropic → 本地小模型 → 规则兜底（永不失败）。
- **自训练模型**：`training/` 内置完整训练流水线（采集 → 弱监督标注 → CPU 训练 → 评估），在 Windows 上约 1 小时训出专属的仓库分析模型，详见 [training/README.md](training/README.md)。模型质量由**独立手工评测集**把关（90 个名仓库手标 + 20 个对抗用例 + 结构检查，已接入 pytest 回归门禁）。
- **语音交互（全离线）**：说话描述场景 → 离线语音识别（faster-whisper）→ 语义推荐 → 离线语音播报（sherpa-onnx + Melo-TTS 中英混读）。内置网页 `GET /api/voice/page`，麦克风一键交互。
- **向量化存储**：自训练模型的 encoder 直接做语义检索（中英文查询）；也可切换 sentence-transformers / OpenAI / 哈希 TF-IDF 兜底。
- **场景推荐**：`POST /api/recommend` 输入自然语言场景，返回语义最相似的项目、匹配分数与可能涉及的依赖，可选 LLM 重排并生成推荐理由。
- **GitHub OAuth**：authlib 实现登录，用户 access token 加密（Fernet）入库；登录用户的 token 可提升 API 限额（60 → 5000 次/小时）。
- **全链路降级**：任何外部依赖（LLM API、embedding 模型、语音模型、Chroma、GitHub API）不可用时都有兜底路径，服务照常启动。

## 架构

```text
┌─────────────┐      ┌──────────────┐      ┌──────────────────┐
│   Web API   │─────▶│   服务层      │─────▶│ 数据库层          │
│  (FastAPI)  │      │ (业务逻辑)    │      │ PostgreSQL/SQLite │
└─────────────┘      └──────────────┘      │ + Chroma 向量库    │
       │                      │              └──────────────────┘
       │                      ▼
       │             ┌──────────────────┐
       │             │ 定时任务调度器     │
       │             │ (APScheduler)    │
       │             └──────────────────┘
       │                      │
       │      ┌───────────────┼───────────────┐
       │      ▼               ▼               ▼
       │ ┌──────────┐  ┌────────────┐  ┌────────────┐
       │ │ GitHub   │  │ AI 分析     │  │ 向量化      │
       │ │ 检索模块  │  │ (可插拔LLM) │  │ (可插拔)    │
       │ └──────────┘  └────────────┘  └────────────┘
       ▼
┌──────────────────────┐
│ GitHub OAuth (authlib)│
└──────────────────────┘
```

## 快速开始（本地）

```bash
# 1. 创建虚拟环境（Windows）
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖（核心依赖即可跑通全部降级路径）
pip install -r requirements.txt -r requirements-dev.txt

# 可选：启用 Chroma 持久化、本地向量化模型、本地 LLM
pip install -r requirements-ml.txt

# 3. 配置
copy .env.example .env    # Linux/macOS: cp .env.example .env
# 至少设置 SECRET_KEY / ADMIN_TOKEN；需要 OAuth 时填 GITHUB_CLIENT_ID/SECRET

# 4. 启动
uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000/docs 查看 Swagger
```

完全不配置任何 API Key 也能启动：此时自动降级为「本地训练模型 + 内存向量库」，适合本地开发与体验。

## 7x24 自主循环（待命 + 巡检）

agent 内置全天候自主循环（已随部署启用），三种运行形态（可并存，共享同一数据库）：

```bash
# 形态 A：API + 自主循环（uvicorn 进程内含调度器，部署采用此形态，端口 8788）
uvicorn app.main:app --host 0.0.0.0 --port 8788

# 形态 B：纯后台 worker（不开 HTTP）
.venv\Scripts\python.exe -m app.worker

# 形态 C：开机自启（无需管理员，脚本已装入当前用户「启动」文件夹）
copy scripts\start_api.bat "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\GitHubAgentAPI.bat"
```

循环内容：

| 任务 | 默认周期 | 说明 |
| --- | --- | --- |
| `fetch_github_trending` | 24h | 按**关键词轮换**扫描 GitHub（`KEYWORDS_PER_FETCH` 控制每轮片长，长期覆盖全量关键词） |
| `update_existing_projects` | 6h | 刷新已入库项目的星标/描述，描述变化自动触发重新分析+重新向量化 |
| `patrol_analysis` | 10min（启动即跑一次） | **巡检自愈**：处理待分析积压 → 重试失败项目 → 检测并修复向量索引漂移（如内存向量库重启丢失） |

资源利用：巡检只处理增量积压（通常几秒跑完），分析阶段支持 `ANALYSIS_WORKERS` 多线程并行吃满空闲 CPU；所有任务 `max_instances=1 + coalesce` 防止堆积。手动触发：`POST /api/admin/run-patrol`（需管理员 Header），巡检报告见 `GET /api/admin/status` 的 `patrol_last_report`。

## 语音交互（可选，全离线）

```bash
pip install -r requirements.txt -r requirements-speech.txt
python training/download_speech_models.py     # 下载 whisper-small(~480MB) + Melo-TTS 中英(~160MB)
uvicorn app.main:app --reload
# 浏览器打开 http://127.0.0.1:8000/api/voice/page
# 点击麦克风说：我需要一个终端文件管理器 -> 返回推荐卡片并语音播报
```

模型未下载时语音接口返回 503 并提示，文本接口与整体服务不受影响。

## 部署与对外开放（当前状态）

项目已在本机完整部署并对公网开放：

- **本机服务**：`http://0.0.0.0:8788`（落地页 `/`、API `/docs`、语音页 `/api/voice/page`）；
  另一进程占用过 `127.0.0.1:8000`，故选用 8788。
- **公网入口**：Cloudflare 快速隧道（免注册），URL 记录在 `PUBLIC_URL.txt`，
  启动方式 `tools\cloudflared.exe tunnel --url http://localhost:8788`。
  快速隧道 URL 随隧道重启变化；需要固定域名可在 Cloudflare 注册账号创建命名隧道。
- **生产加固**：随机生成的 SECRET_KEY / ADMIN_TOKEN（`.env`）、每 IP 限流 120 次/分、
  上传上限 10MB、日志落盘 `logs/agent.log`（10MB 轮转）、SQLite WAL 多进程共享、
  Chroma 向量持久化（`chroma_data/`，重启不丢向量）。
- **开机自启**：`GitHubAgentAPI.bat` 已放入用户启动文件夹（无需管理员）。

## 反馈优化闭环

外部使用者每次推荐都可以 👍/👎（落地页按钮或 `POST /api/feedback`），连同查询词、命中的
项目与评语写入 `feedback` 表；每次推荐同时记录 `query_history`。优化输入：

```bash
# 管理员视角：点赞/点踩统计、被踩最多的项目、高频查询、零结果查询、最近评语
curl -H "X-Admin-Token: $ADMIN_TOKEN" http://127.0.0.1:8788/api/admin/feedback
```

优化策略：零结果查询 → 补充对应领域关键词进入轮换扫描；被踩项目多 → 检查其分析质量或
阈值；高频查询 → 优先保证该领域语料覆盖。反馈数据积累后按此循环迭代（训练 → 金标准
评测门禁 → 上线 → 收集反馈 → 再训练）。

## 自我进化（训练的自我进化 / 自我优化 / 自我修复）

`EvolutionService`（`app/services/evolution_service.py`）把训练管线接进 7×24 循环，默认每天
自跑一轮（`POST /api/admin/run-evolution` 可手动触发），每轮七步：

1. **收割**：训练数据直接来自 agent 自己——数据库里持续增长的抓取项目 + 采集的 jsonl；
2. **构建**：`app/ml/dataset_builder` 去重、定标签词表、切分 train/val/test；
3. **训练**：在**独立子进程**里把挑战者模型训到 `models/staging-<ts>/`（绝不碰线上模型）；
4. **调优**：自动做逐头全局阈值搜索；
5. **评测**：挑战者跑 107 例金标准集 + 全部门禁；
6. **晋升**：门禁全过 **且** 金标准 F1 严格优于现役冠军模型才替换上线（否则挑战者直接
   删除——退化模型永远不可能发布）；晋升 = 原子换目录 + 写 `version.json` + 分析器/嵌入器
   热重载 + 新向量空间下的向量重建（上限内当轮完成，余量由巡检补齐）；
7. **自修复**：每轮先做金丝雀探测（现场加载模型并推理），线上模型损坏则自动回滚到最近
   一个能通过金丝雀的存档版本；没有可用存档则留在线性降级链兜底并强制重训。

版本谱系：`models/repo-analyzer/version.json`（当前版本+指标+数据量）、
`models/evolution_history.json`（最近 30 轮报告）、`models/archive/<ts>/`（历史版本）。
查看状态：`GET /api/admin/evolution`。相关配置见 `.env.example` 的 `EVOLUTION_*` 段。

## GitHub 版本管理与云端资源

仓库：https://github.com/xuebajunzhu/github-agent （main 分支）

- **版本管理**：git 管理，`.gitignore` 锚定根目录排除 `.env`/`models/`/数据/日志/临时工具；
- **CI**（`.github/workflows/ci.yml`）：每次 push 在 GitHub 托管环境跑全量测试
  （重量级用例在无模型/无 torch 的环境自动跳过）；
- **夜间云端训练**（`.github/workflows/evolution.yml`）：每天 22:00 UTC 用 GitHub Actions
  的算力采集新数据 → 训练挑战者 → 金标准门禁 → 通过则把模型发布为 Release 资产
  （`model-<run_id>` tag）；也可在 Actions 页面手动 dispatch；
- **本地拉取晋升**（`scripts/pull_model_release.py`）：下载最新模型 Release，本地金丝雀 +
  金标准门禁仍会复检后才晋升上线——云端训练、本地把关。

推送走本机代理（仓库级 `http.proxy=127.0.0.1:7897`）；认证用 gh CLI（keyring 中已有 token）。

## Docker 部署

```bash
copy .env.example .env     # 填好 SECRET_KEY、ADMIN_TOKEN、GITHUB_TOKEN 等
docker compose up --build  # PostgreSQL + 应用 + Chroma 持久化卷
```

镜像默认安装 `requirements-ml.txt`（体积较大）；只装核心依赖可构建时传入 `--build-arg INSTALL_ML=false`。

## API 摘要

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（含各组件实际生效的 provider） |
| GET | `/auth/github/login` | 跳转 GitHub OAuth 授权页 |
| GET | `/auth/github/callback` | OAuth 回调，换取并加密存储 token，签发 JWT |
| POST | `/api/recommend` | 场景推荐，body: `{"query": "...", "top_k": 5}` |
| GET | `/api/projects/{id}` | 项目详情 |
| GET | `/api/projects?limit=&offset=&language=` | 项目列表 |
| POST | `/api/admin/trigger-fetch` | 手动触发一次检索（Header: `X-Admin-Token`） |
| GET | `/api/admin/status` | 入库/分析/向量统计（需管理员 Header） |
| GET | `/api/voice/status` | 语音组件可用性 |
| POST | `/api/voice/ask` | 语音提问（multipart 音频）→ 文本 + 推荐 + 语音回答（base64 wav） |
| POST | `/api/voice/transcribe` | 音频 → 文本 |
| POST | `/api/voice/speak` | 文本 → WAV 语音 |
| GET | `/api/voice/page` | 语音交互网页（浏览器麦克风） |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "我需要一个用于图像分类的深度学习库", "top_k": 5}'

curl -X POST http://127.0.0.1:8000/api/admin/trigger-fetch \
  -H "X-Admin-Token: change-me-admin-token" \
  -d '{"keywords": ["rust cli"]}'
```

## 配置说明（主要项）

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./github_agent.db` | 生产建议 PostgreSQL |
| `GITHUB_TOKEN` | 空 | 服务端检索用 token，显著提高限额 |
| `GITHUB_SEARCH_KEYWORDS` | JSON 数组 | 定时检索的关键词列表 |
| `GITHUB_MIN_STARS` | 100 | 最低星标过滤 |
| `LLM_PROVIDER` | `openai` | `trained`（本地自训练模型，推荐）/ `openai` / `anthropic` / `local` / `rule` |
| `TRAINED_MODEL_PATH` | `./models/repo-analyzer` | 自训练模型目录（存在即自动启用） |
| `LLM_FALLBACK_PROVIDERS` | `["local","rule"]` | 主模型失败后的降级顺序 |
| `EMBEDDER_PROVIDER` | `sentence_transformers` | `trained`（自训练 encoder，推荐）/ `sentence_transformers` / `openai` / `tfidf` |
| `ASR_PROVIDER` / `TTS_PROVIDER` | `faster_whisper` / `sherpa_onnx` | 语音识别/合成，`none` 禁用；模型见 `training/download_speech_models.py` |
| `VECTOR_STORE_BACKEND` | `chroma` | `chroma` / `memory` |
| `SCHEDULER_ENABLED` | `true` | 设为 `false` 可禁用定时任务（如仅做 API 服务） |
| `FETCH_INTERVAL_MINUTES` | 1440 | 检索周期（分钟） |
| `ENABLE_LLM_RERANK` | `false` | 推荐结果用 LLM 重排 + 推荐理由 |

完整配置见 `.env.example`，全部支持环境变量或 `.env` 文件注入。

## 降级策略矩阵

| 环节 | 首选 | 降级 1 | 降级 2（兜底） |
| --- | --- | --- | --- |
| LLM 分析 | 自训练模型（离线、零成本） | OpenAI / Anthropic / 本地小模型 | 规则分析器（关键词映射，永不失败） |
| 向量化 | 自训练 encoder / sentence-transformers（本地） | OpenAI Embedding | 纯 Python 哈希 TF-IDF（零依赖） |
| 语音识别/合成 | faster-whisper small / Melo-TTS（离线，CPU） | — | 接口返回 503 并提示（文本接口不受影响） |
| 向量库 | Chroma（持久化） | — | 内存向量库（重启丢失，需重建索引） |
| GitHub API | 携带 token（5000 次/h） | 匿名（60 次/h） | 限流时指数退避重试，超限则本次任务跳过 |

注意：向量化模型是**启动时一次性选定**的（而非每次调用时降级），因为不同模型维度/语义空间不兼容，混用会破坏相似度检索。切换模型后需要对已有项目重建索引。降级事件均通过 Loguru 记录，`GET /health` 可查看当前实际生效的 provider。

## 数据流

1. **定时检索**：调度器触发 → 按关键词搜索（星标排序）→ 按 `github_id` 去重 upsert（新项目 `analysis_status=pending`）→ AI 分析生成摘要/标签 → 向量化写入向量库 → `done`。
2. **场景推荐**：查询向量化 → 向量库 top-K 相似检索 → 回查 PostgreSQL 补全详情 →（可选）LLM 重排并生成推荐理由 → 返回 JSON。
3. **OAuth 登录**：`/auth/github/login` → GitHub 授权 → 回调换取 access_token → Fernet 加密存库 → 签发 JWT。

## 项目结构

```text
github-agent/
├── app/
│   ├── main.py                 # FastAPI 入口（app 工厂 + lifespan）
│   ├── config.py               # pydantic-settings 配置
│   ├── database.py             # SQLAlchemy 引擎/建表
│   ├── models/                 # ORM：Project / User / QueryHistory
│   ├── ml/                     # 自训练模型：双头分类器 / 标签体系 / 抽取式摘要
│   ├── schemas/                # Pydantic 请求/响应模型
│   ├── api/
│   │   ├── dependencies.py     # DI、管理员校验
│   │   └── routes/             # auth / recommend / projects / admin / health
│   ├── services/
│   │   ├── github_service.py   # GitHub REST 检索（限流退避）
│   │   ├── analysis_service.py # 分析降级链
│   │   ├── embedding_service.py# 向量化 + 文本构建
│   │   ├── ingest_service.py   # 检索→入库→分析→嵌入流水线
│   │   ├── recommend_service.py# 场景推荐（可选 LLM 重排）
│   │   ├── scheduler.py        # APScheduler 任务
│   │   └── container.py        # 服务容器组装
│   ├── llm/                    # 可插拔分析后端（trained/openai/anthropic/local/rule）
│   ├── embedders/              # 可插拔向量化（trained/sentence-transformers/openai/tfidf）
│   ├── speech/                 # 离线语音：faster-whisper ASR + sherpa-onnx Melo TTS
│   ├── static/                 # 语音交互网页（voice.html）
│   ├── vectorstore/            # Chroma / 内存向量库
│   └── utils/                  # 日志、加解密、JWT、时间
├── training/                   # 自训练流水线：采集/准备/训练/评估（见 training/README.md）
├── models/                     # 训练产物（models/repo-analyzer）
├── tests/                      # 单元 + API 集成测试
├── docker-compose.yml / Dockerfile
├── requirements*.txt / .env.example
└── pytest.ini
```

## 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

测试覆盖：规则分析器、分析降级链、哈希向量化确定性、内存向量库、GitHub 客户端限流重试（httpx MockTransport，不访问真实 API）、JWT 签名校验，以及一条端到端集成（模拟检索 → 规则分析 → 向量化 → 场景推荐命中）。全部测试不依赖网络与任何 API Key。

## 设计取舍说明

- **GitHub 客户端用 httpx 直连 REST API**（设计稿允许 PyGithub 或直连）：便于精确实现 403/429 限流退避、`Retry-After` 处理与 MockTransport 测试。
- **TF-IDF 兜底用哈希方案而非 scikit-learn**：维度固定（与主模型对齐）、零依赖、跨运行稳定（SHA-256 分桶）。
- **JWT 用标准库 HS256 实现**：避免额外依赖；生产可无缝替换为 pyjwt/jose。
- **OAuth 需要会话中间件**：authlib 的 state 存储依赖 session，应用已内置 `SessionMiddleware`。
- **新增 `ingest_service` 与 `vectorstore` 模块**：把「检索→分析→嵌入」流水线独立成服务，向量库抽象成独立包，便于按设计稿第 7 节扩展 pgvector/Qdrant/FAISS。

## 后续演进

- 向量库接入 Qdrant/pgvector（生产规模），数据库迁移引入 Alembic。
- 分布式任务：APScheduler → Celery + Redis。
- 增量爬取 GitHub Trending 页面、按用户 OAuth token 提升个人限额。
- 推荐结果缓存与 query_history 驱动的推荐调优。
