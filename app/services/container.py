"""Service container: wires settings, database and all services together."""
from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.backup_service import BackupService
from app.services.digest_service import DigestService
from app.services.embedding_service import EmbeddingService
from app.services.evolution_service import EvolutionService
from app.services.gap_scout import GapScoutService
from app.services.github_service import GitHubService
from app.services.ingest_service import IngestService
from app.services.patrol_service import PatrolService
from app.services.prompt_evolution_service import PromptEvolutionService
from app.services.recommend_service import RecommendService
from app.services.research_service import ResearchService
from app.speech import SpeechToText, TextToSpeech, build_stt, build_tts
from app.utils.crypto import TokenCipher
from app.utils.notifier import Notifier
from app.vectorstore.base import VectorStore


@dataclass
class ServiceContainer:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker
    github: GitHubService
    analysis: AnalysisService
    embedding: EmbeddingService
    vectors: VectorStore
    ingest: IngestService
    recommend: RecommendService
    token_cipher: TokenCipher
    patrol: PatrolService
    evolution: EvolutionService
    digest: DigestService
    research: ResearchService
    prompt_evolution: PromptEvolutionService
    gap_scout: GapScoutService
    backup: BackupService
    notifier: Notifier
    stt: SpeechToText | None = None
    tts: TextToSpeech | None = None


def build_services(settings: Settings, engine: Engine) -> ServiceContainer:
    from app.database import create_sessionmaker
    from app.llm import build_analyzer_chain
    from app.vectorstore import create_vector_store

    session_factory = create_sessionmaker(engine)
    analysis = AnalysisService(build_analyzer_chain(settings), settings)
    embedding = EmbeddingService(settings)
    vectors = create_vector_store(settings)
    # Space-consistency guard: a persistent store whose vectors were built by
    # a different embedder (provider switch, dimension change) silently
    # corrupts similarity search. Wipe + mark everything for re-embed instead.
    if vectors.ensure_space(embedding.space_id):
        _park_all_projects_for_reembed(session_factory)
    github = GitHubService(settings)
    notifier = Notifier(settings)
    ingest = IngestService(settings, session_factory, github, analysis, embedding, vectors)
    recommend = RecommendService(settings, session_factory, embedding, vectors, analysis)
    recommend.preload_reranker()  # warm local rerankers off the request path
    patrol = PatrolService(settings, session_factory, ingest, embedding, vectors, notifier)
    evolution = EvolutionService(settings, session_factory, embedding, vectors, analysis, notifier)
    return ServiceContainer(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        github=github,
        analysis=analysis,
        embedding=embedding,
        vectors=vectors,
        ingest=ingest,
        recommend=recommend,
        token_cipher=TokenCipher(settings.fernet_key, settings.secret_key),
        patrol=patrol,
        evolution=evolution,
        digest=DigestService(settings, session_factory),
        research=ResearchService(
            settings, session_factory, github, analysis, embedding, vectors
        ),
        prompt_evolution=PromptEvolutionService(settings, session_factory, analysis),
        gap_scout=GapScoutService(settings, session_factory, analysis, notifier),
        backup=BackupService(settings),
        notifier=notifier,
        stt=build_stt(settings),
        tts=build_tts(settings),
    )


def _park_all_projects_for_reembed(session_factory: sessionmaker) -> None:
    from app.models.project import Project

    with session_factory() as session:
        count = (
            session.query(Project)
            .filter(Project.analysis_status == "done")
            .update({"embedding_id": None}, synchronize_session=False)
        )
        session.commit()
    if count:
        logger.warning(
            f"Embedding space changed: parked {count} projects for re-embedding "
            "(patrol/process_backlog rebuilds them incrementally)."
        )
