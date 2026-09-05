"""Service container: wires settings, database and all services together."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.embedding_service import EmbeddingService
from app.services.evolution_service import EvolutionService
from app.services.github_service import GitHubService
from app.services.ingest_service import IngestService
from app.services.patrol_service import PatrolService
from app.services.recommend_service import RecommendService
from app.speech import SpeechToText, TextToSpeech, build_stt, build_tts
from app.utils.crypto import TokenCipher
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
    stt: SpeechToText | None = None
    tts: TextToSpeech | None = None


def build_services(settings: Settings, engine: Engine) -> ServiceContainer:
    from app.database import create_sessionmaker
    from app.llm import build_analyzer_chain
    from app.vectorstore import create_vector_store

    session_factory = create_sessionmaker(engine)
    analysis = AnalysisService(build_analyzer_chain(settings))
    embedding = EmbeddingService(settings)
    vectors = create_vector_store(settings)
    github = GitHubService(settings)
    ingest = IngestService(settings, session_factory, github, analysis, embedding, vectors)
    recommend = RecommendService(settings, session_factory, embedding, vectors, analysis)
    patrol = PatrolService(settings, session_factory, ingest, embedding, vectors)
    evolution = EvolutionService(settings, session_factory, embedding, vectors, analysis)
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
        stt=build_stt(settings),
        tts=build_tts(settings),
    )
