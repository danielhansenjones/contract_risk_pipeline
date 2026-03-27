import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey,
    Integer, JSON, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from shared.settings import settings


class JobStatus(str, enum.Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    RETRYING  = "retrying"


class JobStage(str, enum.Enum):
    INGESTION      = "ingestion"
    CLASSIFICATION = "classification"
    SCORING        = "scoring"
    ASSEMBLY       = "assembly"
    DONE           = "done"


class Base(DeclarativeBase):
    pass


class Job(Base):

    __tablename__ = "jobs"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status      = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    stage       = Column(Enum(JobStage), nullable=False, default=JobStage.INGESTION)
    object_key  = Column(String, nullable=False)
    filename    = Column(String, nullable=False)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    error       = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                         onupdate=lambda: datetime.now(timezone.utc))

    chunks = relationship("Chunk", back_populates="job", order_by="Chunk.index")
    result = relationship("RiskResult", back_populates="job", uselist=False)


class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id      = Column(String, ForeignKey("jobs.id"), nullable=False)
    index       = Column(Integer, nullable=False)
    text        = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    clause_type = Column(String, nullable=True)
    confidence  = Column(Float, nullable=True)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    job = relationship("Job", back_populates="chunks")


class RiskResult(Base):
    """
    Structured output from the ML pipeline. Cached here for fast API reads.
    Full JSON report is also written to MinIO for downstream consumers.
    """
    __tablename__ = "risk_results"

    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id         = Column(String, ForeignKey("jobs.id"), nullable=False, unique=True)
    overall_score  = Column(Integer, nullable=False)
    risk_level     = Column(String, nullable=False)
    clause_summary = Column(JSON, nullable=False)
    flags          = Column(JSON, nullable=False)
    report_key     = Column(String, nullable=True)
    created_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    job = relationship("Job", back_populates="result")


def get_engine():
    url = (
        f"postgresql+psycopg2://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}/{settings.postgres_db}"
    )
    return create_engine(url, pool_pre_ping=True)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
