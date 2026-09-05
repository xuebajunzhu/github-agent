# 训练专属的「仓库分析」模型

本项目在 Windows 上用纯开源栈（PyTorch + Transformers）训练一个**专用于本 agent** 的本地模型，零 API 成本、完全离线推理。

## 为什么不是训练一个生成式 LLM

agent 需要的输出里，`tags` 和 `use_cases` 本质是**多标签分类**问题，`summary`/`dependencies` 可用抽取与映射解决。分类路线相比微调小型生成式 LLM：

- **成功率有保证**：输出是分类头 + sigmoid 阈值，天然结构化，不存在 JSON 解析失败、幻觉字段等问题；
- **CPU 可训练**：多语言 MiniLM（118M 参数）在本机 8 核 CPU 上约 30-60 分钟完成，16GB 内存即可；
- **数据零标注成本**：GitHub topics 就是现成标签（弱监督），用项目自带的 GitHub 客户端采集即可。

## 模型结构

| 输出 | 实现方式 |
| --- | --- |
| tags | 多标签分类头（标签空间 = 高频 GitHub topics 词表，默认 400 类） |
| use_cases | 多标签分类头（标签空间 = `app/ml/taxonomy.py` 的 30 类场景体系，中文标签） |
| summary | 原始 description 优先；缺失时用抽取式摘要（README 句子打分，见 `app/ml/extractive.py`） |
| dependencies | language/tag → 依赖映射（`app/llm/rule_based.py` 中的表） |

backbone：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（中英文均可，mean pooling）。

## 训练步骤（Windows，CPU 即可）

```bash
# 0. 安装训练依赖（torch 的 PyPI Windows 轮子即 CPU 版）
pip install -r requirements.txt -r training/requirements-training.txt

# 1. 采集训练数据（约 120 个关键词 x 100 仓库，匿名限速下约 15-20 分钟）
#    有 GitHub token 时会自动使用（更快）；HuggingFace 默认走 hf-mirror.com
python training/collect_data.py --output training/data/raw_repos.jsonl

# 2. 构建数据集（过滤 + 词表 + 划分 train/val/test）
python training/prepare_data.py

# 3. 训练（默认 3 epochs / batch 16 / max_len 96；8 核 CPU 约 30-60 分钟）
python training/train.py
# 可调：--epochs --batch-size --lr --max-len --base-model

# 4. 评估测试集 + 逐标签阈值调优 + 打样例
python training/evaluate.py
```

产物在 `models/repo-analyzer/`：

```text
analyzer_config.json   # 标签表、逐标签阈值、max_len、训练指标
model.pt               # 双头模型权重
encoder/config.json    # encoder 架构配置（离线重建用，不联网）
tokenizer 相关文件
test_metrics.json      # 测试集指标
```

训练完成后应用自动生效：`LLM_PROVIDER=trained`（或放进 `LLM_FALLBACK_PROVIDERS`），
`app/llm/trained_analyzer.py` 的 `TrainedModelAnalyzer` 会从该目录加载。模型目录缺失或
torch 未安装时自动降级到下一个 provider，不影响服务启动。

## 已实现的性能参考

两期训练（CPU）：第一期 8040 仓库 / 268 标签 / 7607 样本；第二期合并新领域关键词后达
**15502 仓库 / 300 标签 / 12000 样本**（3 epochs，约 100 分钟）。

| 头 | 标签数 | 测试集 micro-F1（第二期） | macro-F1 | 说明 |
| --- | --- | --- | --- | --- |
| use_cases | 30 | **0.713** | 0.724 | 场景识别质量高，是相对规则版的最大提升 |
| tags | 300 | 0.226 | 0.064 | 细粒度 topic 预测；运行时与仓库自带 topics 合并输出 |

第二期收益：use_cases 在更大更难的测试集上提升（val f1@0.5 从 0.490 → 0.554），
且训练覆盖翻倍的领域（telegram/安全/IoT/办公等 130 个新领域关键词）。

- 全局阈值：tags=0.55、use_cases=0.70（验证集上按 micro-F1 搜索，见 `retune_thresholds.py`）；
- 推理速度：CPU 约 0.4 秒/仓库（含 tokenizer），批处理更快；
- 语音（faster-whisper small + Melo-TTS，全离线 CPU）：短句合成约 4 秒、识别约 5 秒；
  端到端实测「我需要一个终端文件管理器」语音提问 → 命中 lf/walk/nnn.nvim 并语音回答；
- 端到端分析实测：真实检索 "terminal file manager" 前 20 个仓库，use_cases（命令行工具/文件处理）
  与依赖（yazi→cargo、lf→go modules、nnn→cmake）全部正确。

经验教训（已固化在脚本里）：标签极度稀疏时必须用 `pos_weight` 加权 BCE（否则模型只学标签先验）；
小验证集上**不要**做逐标签阈值搜索（精度会崩），每头一个全局阈值即可。

## 评测（独立 golden set）

`training/evals/golden_set.json` 是**独立于训练弱监督标签**的手工评测集，用于回答
「模型实际效果好不好」，而非「与标注管线是否一致」：

- **90 个名仓库案例**：覆盖全部 30 个场景类别的权威手标（fastapi、pytorch、kubernetes、
  vnpy、ardupilot ...），含必须命中的 use_cases、必须包含的 tags、依赖提示、禁止出现的标签；
- **20 个对抗/鲁棒用例**：中文描述、名称与描述冲突、否定句（`machine-learning-free`，标记为
  known_hard 不进门禁）、空描述靠 README 摘要、超长描述、大写 topics、纯语言回退等；
- **结构检查**：标签空间合法性、tags 小写/去重/上限、summary 非空且长度受控、延迟 p95。

运行（门禁不过退出码非 0，已接入 pytest 回归 `tests/test_golden_eval.py`）：

```bash
python training/golden_eval.py            # 全量评测 + 门禁 + 报告
python training/golden_eval.py --fresh "quantum computing" "astronomy"   # 新领域漂移检查
```

**两轮结果对比**（首跑暴露问题 → 修复 → 复跑）：

| 指标 | 首跑 | 修复后 | 门禁 |
| --- | --- | --- | --- |
| use_cases micro-F1 | 0.710 | **0.714** | ≥ 0.60 |
| use_cases recall | 0.859 | **0.977** | ≥ 0.60 |
| 禁止标签违规 | 0 | 0 | = 0 |
| merged tags recall | 0.963 | 0.963 | ≥ 0.80 |
| 依赖提示命中率 | 52/52 | 52/52 | 漏 0 |
| 延迟 p95 | 0.24s | 0.13s | ≤ 2.5s |

首跑发现的两个真实问题与修复：(1) 中文描述用例大面积失败（分类头仅用英文语料训练）→
use_cases 输出改为「分类头 ∪ 高精度规则 taxonomy」集成，并给规则表补齐中文领域词，
recall 0.859 → 0.977；(2) 模型 tag 头在名仓库上零增益（model-only recall 0.000，生产质量
由 topics 合并承载）——如实记录为已知特性，不通过调阈值粉饰。诚实保留的已知限制：
否定句（"machine-learning-free"）会因关键词子串匹配误报 机器学习（known_hard，不进门禁）。

## 重新训练 / 换 backbone

- 数据更多更好：调大 `--max-samples`、降低 `--min-topic-freq` 提高标签覆盖；
- 换 backbone：`--base-model` 任意 HF encoder（如 `microsoft/Multilingual-MiniLM-L12-H384`）；
- 国内网络：脚本默认 `HF_ENDPOINT=https://hf-mirror.com`，可覆盖为官方源。
