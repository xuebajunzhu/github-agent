"""Rule-based analyzer: the ultimate fallback that needs no network or model.

Generates summary/use_cases/tags/dependencies from repo metadata using
keyword mapping tables. Quality is intentionally modest but it never fails.
"""
from __future__ import annotations

from app.llm.base import AnalysisResult, LLMAnalyzer

# keyword -> tag (scanned against description + README + topics, lowercased)
TAG_KEYWORDS: dict[str, str] = {
    "machine learning": "machine-learning",
    "deep learning": "deep-learning",
    "neural network": "neural-network",
    "natural language": "nlp",
    "nlp": "nlp",
    "llm": "llm",
    "computer vision": "computer-vision",
    "image": "computer-vision",
    "web framework": "web",
    "web application": "web",
    "rest api": "api",
    "api": "api",
    "database": "database",
    "orm": "orm",
    "cli": "cli",
    "command line": "cli",
    "visualization": "visualization",
    "crawler": "crawler",
    "scraping": "crawler",
    "async": "async",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "testing": "testing",
    "compiler": "compiler",
    "game": "game",
    "security": "security",
    "encryption": "security",
    "blockchain": "blockchain",
}

# keyword -> use case label (Chinese labels per the design doc examples)
KEYWORD_USE_CASES: list[tuple[str, str]] = [
    ("machine learning", "机器学习"),
    ("deep learning", "深度学习"),
    ("natural language", "自然语言处理"),
    ("nlp", "自然语言处理"),
    ("llm", "大模型应用"),
    ("computer vision", "计算机视觉"),
    ("image", "计算机视觉"),
    ("web framework", "Web 开发"),
    ("web application", "Web 开发"),
    ("api", "API 开发"),
    ("cli", "命令行工具"),
    ("command line", "命令行工具"),
    ("database", "数据存储"),
    ("orm", "数据存储"),
    ("visualization", "数据可视化"),
    ("crawler", "数据采集"),
    ("scraping", "数据采集"),
    ("automation", "自动化"),
    ("testing", "测试工具"),
    ("security", "安全"),
    ("game", "游戏开发"),
]

DEPENDENCIES_BY_LANGUAGE: dict[str, list[str]] = {
    "python": ["pip"],
    "javascript": ["npm"],
    "typescript": ["npm"],
    "rust": ["cargo"],
    "go": ["go modules"],
    "java": ["maven / gradle"],
    "kotlin": ["maven / gradle"],
    "c++": ["cmake / vcpkg"],
    "c": ["cmake"],
    "c#": ["nuget"],
    "ruby": ["gem / bundler"],
    "php": ["composer"],
    "swift": ["swift package manager"],
    "dart": ["pub"],
}

DEPENDENCIES_BY_TAG: dict[str, list[str]] = {
    "pytorch": ["torch"],
    "tensorflow": ["tensorflow"],
    "react": ["react", "react-dom"],
    "vue": ["vue"],
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "llm": ["transformers"],
}


class RuleBasedAnalyzer(LLMAnalyzer):
    name = "rule"

    def analyze(self, project_data: dict) -> AnalysisResult:
        full_name = str(project_data.get("full_name") or "unknown repository")
        description = str(project_data.get("description") or "").strip()
        language = str(project_data.get("language") or "").strip()
        topics = [str(topic).lower() for topic in (project_data.get("topics") or [])]
        readme = str(project_data.get("readme_excerpt") or "").strip()
        haystack = " ".join([description, readme, " ".join(topics)]).lower()

        tags: list[str] = []
        for source in (topics, [language.lower()] if language else []):
            for tag in source:
                if tag and tag not in tags:
                    tags.append(tag)
        for keyword, tag in TAG_KEYWORDS.items():
            if keyword in haystack and tag not in tags:
                tags.append(tag)
        tags = tags[:8]

        use_cases: list[str] = []
        for keyword, label in KEYWORD_USE_CASES:
            if keyword in haystack and label not in use_cases:
                use_cases.append(label)
        use_cases = use_cases[:5]

        summary = description or self._summary_from_readme(full_name, language, readme)

        dependencies: list[str] = []
        lang_key = language.lower()
        dependencies.extend(DEPENDENCIES_BY_LANGUAGE.get(lang_key, []))
        for tag in tags:
            for dep in DEPENDENCIES_BY_TAG.get(tag, []):
                if dep not in dependencies:
                    dependencies.append(dep)
        dependencies = dependencies[:6]

        return AnalysisResult(
            summary=summary,
            use_cases=use_cases,
            tags=tags,
            dependencies=dependencies,
            provider=self.name,
        )

    @staticmethod
    def _summary_from_readme(full_name: str, language: str, readme: str) -> str:
        for line in readme.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if len(stripped) >= 20:
                return stripped[:200]
        suffix = f" written in {language}" if language else ""
        return f"{full_name}: an open-source project{suffix}."
