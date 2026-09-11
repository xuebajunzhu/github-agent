# Code Wiki — GitHub 开源项目智能检索与分析 Agent

> 本文档是对本仓库的**结构化代码说明（Code Wiki）**，面向需要快速理解、维护或扩展此项目的开发者。
> 它从五个维度展开：整体架构、模块职责、关键类与函数、依赖关系、运行方式。

## 项目是什么

一个 **7×24 小时自主运行的 GitHub 开源项目检索与分析 Agent**：

1. **定时检索**（APScheduler）按关键词从 GitHub 搜索仓库并去重入库；
2. **AI 分析**（可插拔 LLM）自动生成 `summary` / `use_cases` / `tags` / `dependencies`；
3. **向量化入库**（可插拔 Embedder + Chroma/内存向量库）；
4. **场景推荐**（语义搜索）根据用户的自然语言场景返回最相似的项目与依赖；
5. **自我进化**（自训练模型 + 金标准门禁 + 自动晋升/回滚），全程离线、零 API 成本；
6. **可选语音交互**（faster-whisper ASR + sherpa-onnx/Melo-TTS TTS，全离线）。

技术栈：`FastAPI + SQLAlchemy + APScheduler + httpx + Chroma + sentence-transformers`，
LLM 支持 **5 级降级**：本地训练模型 → OpenAI / Anthropic → 本地小模型 → 规则兜底（永不失败）。

## 目录结构索引

| 文档 | 内容 |
| --- | --- |
| [README.md](architecture.md) | （当前）项目概述与文档导航 |
| [architecture.md](architecture.md) | 整体架构、分层、数据流、降级策略矩阵 |
| [modules.md](modules.md) | 每个模块/包的职责与文件清单 |
| [api-reference.md](api-reference.md) | 关键类、函数、数据结构说明 |
| [dependencies.md](dependencies.md) | 内部依赖关系图 + 外部依赖清单 |
| [runbook.md](runbook.md) | 本地开发、7×24 循环、Docker、训练、语音、CI/CD 运行方式 |

## 仓库顶层结构速览

```text
github-agent/
├── app/                    # 应用主包（FastAPI + 服务层 + 可插拔后端）
│   ├── main.py             # FastAPI 入口（app 工厂 + lifespan）
│   ├── worker.py           # 纯后台 24/7 worker（不开 HTTP）
│   ├── config.py           # pydantic-settings 配置中心
│   ├── database.py         # SQLAlchemy 引擎 / 会话 / 建表
│   ├── api/                # 路由、依赖注入、限流中间件
│   ├── models/             # ORM 模型（Project / User / QueryHistory / Feedback）
│   ├── schemas/            # Pydantic 请求/响应模型
│   ├── services/           # 业务逻辑与编排（核心）
│   ├── llm/                # 可插拔 LLM 分析后端（trained/openai/anthropic/local/rule）
│   ├── embedders/          # 可插拔向量化后端（trained/sentence_transformers/openai/tfidf）
│   ├── vectorstore/        # 向量库抽象（chroma / memory）
│   ├── ml/                 # 自训练模型：双头分类器 / 标签体系 / 抽取式摘要 / 数据构建
│   ├── speech/             # 离线语音（ASR + TTS）
│   ├── static/             # 落地页 & 语音交互网页
│   └── utils/              # 日志 / 加密 / JWT / 时间工具
├── training/               # 自训练流水线（采集/准备/训练/评估/金标准门禁）
├── models/                 # 训练产物（models/repo-analyzer 等）
├── scripts/                # 辅助脚本（集成测试/拉取模型/开机自启/启动脚本）
├── tests/                  # 单元 + API 集成测试（无网络、无 Key）
├── .github/workflows/      # CI / 夜间训练 / 公网部署
├── Dockerfile / docker-compose.yml
├── requirements*.txt / .env.example
└── pytest.ini
```

## 快速入口

- 想了解**为什么这样设计、数据怎么流动** → [architecture.md](architecture.md)
- 想找**某个功能在哪个文件实现** → [modules.md](modules.md)
- 想了解**具体类/函数的签名与职责** → [api-reference.md](api-reference.md)
- 想知道**模块之间怎么依赖、外部依赖有哪些** → [dependencies.md](dependencies.md)
- 想**把项目跑起来** → [runbook.md](runbook.md)