# 项目运行方式

## 1. 环境要求

- Python 3.12+
- 可选：能够访问 GitHub API / 互联网（训练与云端模型远端下载需要）
- 推荐：SQLite（默认）可零依赖启动；生产可用 PostgreSQL

## 2. 本地快速开始

```bash
# 1. 创建虚拟环境（Windows）
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate

# 2. 安装核心依赖（跑通全部降级路径已足够）
pip install -r requirements.txt -r requirements-dev.txt

# 3. 配置
cp .env.example .env              # Windows: copy .env.example .env
# 至少设置 SECRET_KEY / ADMIN_TOKEN；需要 OAuth 填 GITHUB_CLIENT_ID/SECRET

# 4. 启动 API
uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000/docs 查看 Swagger
```

**零配置也能启动**：不设置任何 API Key 时自动降级为“本地训练模型 + 内存向量库”。若希望启用 Chroma 持久化、本地向量化模型、本地 LLM，追加安装：

```bash
pip install -r requirements-ml.txt
```

## 3. 7×24 自主循环（三种运行形态，可并存）

```bash
# 形态 A：API + 自主循环（部署主形态，端口 8788）
uvicorn app.main:app --host 0.0.0.0 --port 8788

# 形态 B：纯后台 worker（不开 HTTP）
.venv\Scripts\python.exe -m app.worker

# 形态 C：开机自启（Windows，无需管理员）
copy scripts\start_api.bat "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\GitHubAgentAPI.bat"
```

循环默认周期（可经 `.env` 调整）：

| 任务 | 默认周期 | 说明 |
| --- | --- | --- |
| `fetch_github_trending` | 24h | 关键词轮换扫描 GitHub |
| `update_existing_projects` | 6h | 刷新项目星标/描述，变化触发重新分析+向量化 |
| `patrol_analysis` | 10min（启动即跑） | 巡检自愈：积压 + 重试失败 + 修复向量漂移 |
| `model_evolution` | 24h | 自我进化（需 `EVOLUTION_ENABLED=true`） |

手动触发（需管理员 Header）：

```bash
curl -X POST http://127.0.0.1:8788/api/admin/trigger-fetch  -H "X-Admin-Token: $ADMIN_TOKEN" -d '{"keywords": ["rust cli"]}'
curl -X POST http://127.0.0.1:8788/api/admin/run-patrol     -H "X-Admin-Token: $ADMIN_TOKEN"
curl -X POST http://127.0.0.1:8788/api/admin/run-evolution  -H "X-Admin-Token: $ADMIN_TOKEN"
```

## 4. 语音交互（可选，全离线）

```bash
pip install -r requirements.txt -r requirements-speech.txt
python training/download_speech_models.py     # 下载 whisper-small(~480MB) + Melo-TTS(~160MB)
uvicorn app.main:app --reload
# 浏览器打开 http://127.0.0.1:8000/api/voice/page，点击麦克风即可交互
```

模型未下载时语音接口返回 503 并提示，文本接口与整体服务不受影响。

## 5. Docker 部署

```bash
cp .env.example .env           # 填好 SECRET_KEY、ADMIN_TOKEN、GITHUB_TOKEN 等
docker compose up --build      # PostgreSQL + 应用 + Chroma 持久化卷
```

- 镜像默认安装 `requirements-ml.txt`（体积较大）；
- 只装核心依赖可构建时传入 `--build-arg INSTALL_ML=false`。

## 6. 自训练模型（全离线，零 API 成本）

```bash
pip install -r requirements.txt -r training/requirements-training.txt

# 1. 采集训练数据（约 120 关键词 x 100 仓库）
python training/collect_data.py --output training/data/raw_repos.jsonl

# 2. 构建数据集（过滤 + 词表 + train/val/test）
python training/prepare_data.py

# 3. 训练（CPU 约 30-60 分钟）
python training/train.py

# 4. 评估 + 阈值调优
python training/evaluate.py
```

产物在 `models/repo-analyzer/`。训练完成后设置 `LLM_PROVIDER=trained`、`EMBEDDER_PROVIDER=trained` 即自动生效。

### 金标准门禁（独立手工评测）

```bash
python training/golden_eval.py                          # 全量评测 + 门禁 + 报告
python training/golden_eval.py --fresh "quantum computing" "astronomy"   # 新领域漂移检查
```

门禁不过退出码非 0，已接入 pytest 回归（`tests/test_golden_eval.py`）。

## 7. 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

测试覆盖：规则分析器、分析降级链、哈希向量化确定性、内存向量库、GitHub 客户端限流重试（httpx MockTransport）、JWT、语音，以及一条端到端集成链路。**全部测试不依赖网络与任何 API Key**（`tests/conftest.py` 提供全降级 `settings` 与 `TestClient`）。

## 8. 部署与对外开放（生产形态参考）

- **本机服务**：`http://0.0.0.0:8788`（落地页 `/`、API `/docs`、语音页 `/api/voice/page`）。
- **公网入口**：Cloudflare 快速隧道 `cloudflared tunnel --url http://localhost:8788`。
- **生产加固**：随机 `SECRET_KEY`/`ADMIN_TOKEN`、每 IP 限流 120 次/分、上传上限 10MB、日志落盘轮转、SQLite WAL 多进程共享、Chroma 向量持久化。

## 9. CI/CD（GitHub Actions）

| 工作流 | 触发 | 作用 |
| --- | --- | --- |
| [ci.yml](../.github/workflows/ci.yml) | push / PR / 手动 | 跑全量测试（无模型/无 torch 的用例自动跳过） |
| [evolution.yml](../.github/workflows/evolution.yml) | 每天 22:00 UTC / 手动 | 云端采集→训练→金标准门禁→通过则发布为 Release 资产（`model-<run_id>` tag） |
| [deploy.yml](../.github/workflows/deploy.yml) | 每天 21:00 UTC / 手动 | 在 runner 上拉起公网实例 + 远程集成测试，临时暴露给人类体验 |

本地拉取云端晋升的模型：

```bash
python scripts/pull_model_release.py     # 下载最新 Release，本地金丝雀 + 金标准门禁复检后晋升
```

## 10. 常用接口速查

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查（含各组件实际 provider） |
| GET | `/auth/github/login` | 跳转 GitHub OAuth |
| GET | `/auth/github/callback` | OAuth 回调，签发 JWT |
| POST | `/api/recommend` | 场景推荐 `{"query": "...", "top_k": 5}` |
| GET | `/api/projects/{id}` | 项目详情 |
| GET | `/api/projects?limit=&offset=&language=` | 项目列表 |
| POST | `/api/admin/trigger-fetch` | 手动检索（`X-Admin-Token`） |
| POST | `/api/admin/run-patrol` | 手动巡检（`X-Admin-Token`） |
| POST | `/api/admin/run-evolution` | 手动进化（`X-Admin-Token`） |
| GET | `/api/admin/status` | 入库/分析/向量统计 |
| GET | `/api/admin/feedback` | 反馈聚合报表 |
| POST | `/api/feedback` | 提交 👍/👎 |
| POST | `/api/voice/ask` | 语音提问 → 文本 + 推荐 + 语音回答 |
| GET | `/api/voice/page` | 语音交互网页 |

> 完整配置项见 [.env.example](../.env.example)，全部支持环境变量或 `.env` 注入。