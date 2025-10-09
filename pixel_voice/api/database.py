"""
Database models and persistence layer for Pixel Voice API.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
import structlog

from .models import JobStatus, PipelineStage
from .auth import UserRole

logger = structlog.get_logger(__name__)

Base = declarative_base()


class User(Base):
    """User database model."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.STANDARD.value)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    api_key = Column(String(255), unique=True, nullable=True, index=True)
    quota_limit = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    jobs = relationship("Job", back_populates="user")
    usage_records = relationship("UsageRecord", back_populates="user")

    __table_args__ = (
        Index("idx_user_email_active", "email", "is_active"),
        Index("idx_user_api_key_active", "api_key", "is_active"),
    )


class Job(Base):
    """Pipeline job database model."""

    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default=JobStatus.PENDING.value)
    stages = Column(JSON, nullable=False)  # List of pipeline stages
    current_stage = Column(String(100), nullable=True)
    progress = Column(Float, default=0.0)
    input_data = Column(JSON, nullable=True)
    output_paths = Column(JSON, nullable=True)
    config_overrides = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="jobs")
    stage_results = relationship("StageResult", back_populates="job")

    __table_args__ = (
        Index("idx_job_user_status", "user_id", "status"),
        Index("idx_job_created_at", "created_at"),
        Index("idx_job_status_created", "status", "created_at"),
    )


class StageResult(Base):
    """Pipeline stage result database model."""

    __tablename__ = "stage_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    stage = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    input_path = Column(String(500), nullable=True)
    output_path = Column(String(500), nullable=True)
    execution_time = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("Job", back_populates="stage_results")

    __table_args__ = (
        Index("idx_stage_job_stage", "job_id", "stage"),
        Index("idx_stage_status_created", "status", "created_at"),
    )


class UsageRecord(Base):
    """User usage tracking database model."""

    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    record_type = Column(String(50), nullable=False)  # api_call, youtube_download, etc.
    count = Column(Integer, default=1)
    metadata = Column(JSON, nullable=True)
    date = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="usage_records")

    __table_args__ = (
        Index("idx_usage_user_type_date", "user_id", "record_type", "date"),
        Index("idx_usage_date", "date"),
    )


class SystemConfig(Base):
    """System configuration database model."""

    __tablename__ = "system_config"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(255), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("idx_config_key", "key"),)


class AuditLog(Base):
    """Audit log database model."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_resource", "resource_type", "resource_id"),
        Index("idx_audit_created_at", "created_at"),
    )


class DatabaseManager:
    """Database connection and session manager."""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def close(self):
        """Close database connections."""
        self.engine.dispose()
        logger.info("Database connections closed")


class UserRepository:
    """User data access layer."""

    def __init__(self, db: Session):
        self.db = db

    def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create new user."""
        user = User(**user_data)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        return self.db.query(User).filter(User.api_key == api_key, User.is_active == True).first()

    def update_user(self, user_id: uuid.UUID, update_data: Dict[str, Any]) -> Optional[User]:
        """Update user."""
        user = self.get_user_by_id(user_id)
        if user:
            for key, value in update_data.items():
                setattr(user, key, value)
            user.updated_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(user)
        return user

    def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users with pagination."""
        return self.db.query(User).offset(skip).limit(limit).all()


class JobRepository:
    """Job data access layer."""

    def __init__(self, db: Session):
        self.db = db

    def create_job(self, job_data: Dict[str, Any]) -> Job:
        """Create new job."""
        job = Job(**job_data)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job_by_id(self, job_id: uuid.UUID) -> Optional[Job]:
        """Get job by ID."""
        return self.db.query(Job).filter(Job.id == job_id).first()

    def update_job(self, job_id: uuid.UUID, update_data: Dict[str, Any]) -> Optional[Job]:
        """Update job."""
        job = self.get_job_by_id(job_id)
        if job:
            for key, value in update_data.items():
                setattr(job, key, value)
            self.db.commit()
            self.db.refresh(job)
        return job

    def list_user_jobs(
        self, user_id: uuid.UUID, status: Optional[str] = None, skip: int = 0, limit: int = 100
    ) -> List[Job]:
        """List user jobs with optional status filter."""
        query = self.db.query(Job).filter(Job.user_id == user_id)
        if status:
            query = query.filter(Job.status == status)
        return query.order_by(Job.created_at.desc()).offset(skip).limit(limit).all()

    def get_active_jobs_count(self, user_id: uuid.UUID) -> int:
        """Get count of active jobs for user."""
        return (
            self.db.query(Job)
            .filter(
                Job.user_id == user_id,
                Job.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value]),
            )
            .count()
        )


class UsageRepository:
    """Usage tracking data access layer."""

    def __init__(self, db: Session):
        self.db = db

    def record_usage(
        self, user_id: uuid.UUID, record_type: str, count: int = 1, metadata: Optional[Dict] = None
    ):
        """Record usage."""
        usage = UsageRecord(
            user_id=user_id, record_type=record_type, count=count, metadata=metadata
        )
        self.db.add(usage)
        self.db.commit()

    def get_daily_usage(self, user_id: uuid.UUID, record_type: str, date: datetime) -> int:
        """Get daily usage count."""
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        result = (
            self.db.query(self.db.func.sum(UsageRecord.count))
            .filter(
                UsageRecord.user_id == user_id,
                UsageRecord.record_type == record_type,
                UsageRecord.date >= start_of_day,
                UsageRecord.date < end_of_day,
            )
            .scalar()
        )

        return result or 0


# Global database manager
db_manager: Optional[DatabaseManager] = None


def init_database(database_url: str):
    """Initialize database."""
    global db_manager
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    logger.info("Database initialized", database_url=database_url)


def get_db() -> Session:
    """Dependency to get database session."""
    if not db_manager:
        raise RuntimeError("Database not initialized")

    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
