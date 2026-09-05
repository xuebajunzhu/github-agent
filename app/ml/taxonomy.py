"""Use-case taxonomy: maps GitHub topics / text keywords to use-case labels.

Shared by the training data builder (weak supervision) and documented as the
label space of the trained model. Use-case labels are Chinese per the design doc.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UseCaseSpec:
    label: str
    topics: set[str] = field(default_factory=set)
    keywords: list[str] = field(default_factory=list)


# keyword lists include both English and Chinese markers; matching is plain
# substring over lowercased description+readme+topics text.
USE_CASE_TAXONOMY: list[UseCaseSpec] = [
    UseCaseSpec("机器学习", {"machine-learning", "ml", "deep-learning", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "mxnet", "jax"}, ["machine learning", "deep learning", "neural network", "机器学习", "深度学习", "神经网络"]),
    UseCaseSpec("自然语言处理", {"nlp", "natural-language-processing", "transformers", "bert", "gpt", "llm", "text-processing", "tokenization", "spacy", "chatbot", "language-model"}, ["natural language", "text classification", "named entity", "language model", "chatbot", "自然语言处理", "分词", "词性标注", "情感分析", "命名实体"]),
    UseCaseSpec("大模型应用", {"llm", "large-language-models", "gpt", "openai", "langchain", "rag", "agent", "ai-agents", "chatgpt", "vector-database", "embeddings"}, ["large language model", "llm", "rag", "retrieval augmented", "prompt", "大语言模型", "大模型", "文档问答"]),
    UseCaseSpec("计算机视觉", {"computer-vision", "cv", "image-processing", "opencv", "object-detection", "image-classification", "segmentation", "ocr", "yolo"}, ["computer vision", "image classification", "object detection", "image segmentation", "ocr", "计算机视觉", "目标检测", "图像分割"]),
    UseCaseSpec("Web 开发", {"web", "web-development", "webapp", "website", "frontend", "backend", "fullstack", "django", "flask", "fastapi", "express", "vue", "react", "angular", "nextjs", "svelte", "laravel", "spring-boot", "rails"}, ["web application", "web framework", "web development", "full-stack", "website", "网站", "网页"]),
    UseCaseSpec("API 开发", {"api", "apis", "rest-api", "graphql", "grpc", "api-gateway", "openapi", "swagger", "microservices"}, ["rest api", "graphql", "grpc", "api development", "microservice", " apis", "接口开发", "网关"]),
    UseCaseSpec("命令行工具", {"cli", "command-line", "commandline", "terminal", "shell", "bash", "zsh"}, ["command line", "cli tool", "terminal", "shell", "命令行", "终端"]),
    UseCaseSpec("数据存储", {"database", "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "orm", "nosql", "elasticsearch", "key-value"}, ["database", "orm", "sql", "storage engine", "key-value store", "数据库", "键值", "存储引擎"]),
    UseCaseSpec("数据可视化", {"visualization", "data-visualization", "charts", "plotting", "dataviz", "d3", "dashboard"}, ["visualization", "charting", "dashboard", "plotting", "可视化", "图表", "走势图"]),
    UseCaseSpec("数据分析", {"data-analysis", "data-science", "pandas", "numpy", "dataframe", "analytics", "jupyter", "statistics"}, ["data analysis", "data science", "analytics", "statistical", "数据分析", "数据处理"]),
    UseCaseSpec("数据采集", {"crawler", "scraper", "scraping", "spider", "web-crawler"}, ["crawler", "scraping", "spider", "web scraping", "爬虫", "抓取", "定时抓取"]),
    UseCaseSpec("自动化", {"automation", "workflow", "robot", "rpa", "task-runner", "ci-cd", "github-actions", "airflow"}, ["automation", "workflow orchestration", "task automation", "ci/cd", "自动化", "工作流"]),
    UseCaseSpec("开发框架", {"framework", "library", "boilerplate", "starter", "template", "scaffold"}, ["framework", "boilerplate", "template", "框架"]),
    UseCaseSpec("测试工具", {"testing", "test", "unit-testing", "e2e-testing", "mock", "pytest", "selenium", "fuzzing"}, ["testing", "unit test", "test framework", "e2e", "测试"]),
    UseCaseSpec("安全", {"security", "pentesting", "cryptography", "encryption", "vulnerability", "malware", "forensics", "authentication", "authorization", "oauth", "jwt"}, ["security", "encryption", "penetration test", "vulnerability", "authentication", "加密", "密码库", "两步验证", "身份认证", "漏洞"]),
    UseCaseSpec("区块链", {"blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart-contracts", "solidity", "web3"}, ["blockchain", "cryptocurrency", "smart contract", "web3", "区块链", "智能合约"]),
    UseCaseSpec("游戏开发", {"game", "gamedev", "game-engine", "game-development", "unity", "unreal", "godot", "opengl", "vulkan"}, ["game", "game engine", "game development", "graphics rendering", "游戏引擎", "游戏"]),
    UseCaseSpec("多媒体处理", {"audio", "video", "ffmpeg", "media", "image-editor", "audio-processing", "video-editing", "speech"}, ["audio processing", "video editing", "media player", "speech recognition", "text-to-speech", "视频", "音频", "音乐播放", "媒体播放"]),
    UseCaseSpec("移动开发", {"android", "ios", "mobile", "flutter", "react-native", "swift", "kotlin", "mobile-app"}, ["android", "ios", "mobile app", "flutter", "移动应用", "移动端"]),
    UseCaseSpec("嵌入式与物联网", {"iot", "embedded", "embedded-systems", "arduino", "raspberry-pi", "mqtt", "firmware", "rtos", "esp32"}, ["iot", "embedded", "microcontroller", "firmware", "mqtt", "物联网", "树莓派", "固件", "传感器", "智能家居", "嵌入式"]),
    UseCaseSpec("DevOps 与部署", {"devops", "docker", "kubernetes", "k8s", "terraform", "ansible", "deployment", "container", "helm", "monitoring", "observability", "prometheus", "grafana"}, ["docker", "kubernetes", "deployment", "infrastructure as code", "monitoring", "observability", "容器", "部署", "编排", "监控"]),
    UseCaseSpec("网络编程", {"networking", "network", "proxy", "vpn", "http", "tcp", "websocket", "p2p", "dns"}, ["network", "proxy", "http server", "websocket", "p2p", "代理", "网络"]),
    UseCaseSpec("教育与学习", {"tutorial", "learning", "education", "course", "interview", "cheatsheet", "awesome", "roadmap", "book", "algorithm"}, ["tutorial", "learning", "interview preparation", "algorithm study", "教育", "学习", "初学者", "算法可视化", "课程"]),
    UseCaseSpec("开发工具", {"developer-tools", "toolkit", "productivity", "editor", "ide", "vim", "vscode", "plugin", "extension", "git"}, ["developer tools", "productivity", "editor", "ide extension", "dotfiles", "version control", "开发工具", "编辑器", "版本控制"]),
    UseCaseSpec("爬虫与搜索引擎", {"search-engine", "elasticsearch", "full-text-search", "search", "indexing"}, ["search engine", "full-text search", "indexing", "搜索引擎", "全文检索"]),
    UseCaseSpec("推荐系统", {"recommendation-system", "recommender", "collaborative-filtering", "recsys"}, ["recommendation", "recommender system", "推荐系统", "推荐引擎", "协同过滤", "个性化推荐"]),
    UseCaseSpec("机器人", {"robotics", "ros", "robot", "slam", "drone"}, ["robotics", "ros", "slam", "robot", "机器人", "自动驾驶", "无人机"]),
    UseCaseSpec("科学计算", {"scientific-computing", "simulation", "physics", "chemistry", "bioinformatics", "computational-biology", "mathematics"}, ["simulation", "scientific computing", "bioinformatics", "physics", "科学计算", "计算机代数", "数值计算"]),
    UseCaseSpec("金融量化", {"finance", "quant", "trading", "fintech", "stock", "cryptocurrency-trading"}, ["finance", "trading", "quantitative", "stock", "量化交易", "交易所", "交易机器人"]),
    UseCaseSpec("文件处理", {"pdf", "file-manager", "compression", "markdown", "office", "excel", "csv"}, ["pdf", "file management", "file format", "markdown", "spreadsheet", "markup converter", "文档转换", "批量重命名"]),
]


def match_use_cases(topics: list[str], text: str) -> list[str]:
    """Return use-case labels matched by topics first, then text keywords."""
    lowered_topics = {str(topic).lower() for topic in topics}
    lowered_text = (text or "").lower()
    matched: list[str] = []
    for spec in USE_CASE_TAXONOMY:
        if lowered_topics & spec.topics:
            matched.append(spec.label)
            continue
        if any(keyword in lowered_text for keyword in spec.keywords):
            matched.append(spec.label)
    return matched
