from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, default="queued", nullable=False)
    files: Mapped[Any] = mapped_column(JSONB, default=list, nullable=False)
    collection: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[Any] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    documents = relationship("DocumentModel", back_populates="job", cascade="all, delete-orphan")


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    job_id: Mapped[str] = mapped_column(Text, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str | None] = mapped_column(Text, nullable=True)

    job = relationship("JobModel", back_populates="documents")
    parties = relationship("PartyModel", back_populates="document", cascade="all, delete-orphan")
    relations = relationship("RelationModel", back_populates="document", cascade="all, delete-orphan")


class PartyModel(Base):
    __tablename__ = "parties"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str | None] = mapped_column(Text, nullable=True)

    document = relationship("DocumentModel", back_populates="parties")
    obligations = relationship("ObligationModel", back_populates="party", cascade="all, delete-orphan")
    risks = relationship("RiskModel", back_populates="party", cascade="all, delete-orphan")


class RelationModel(Base):
    __tablename__ = "relations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    source_party_id: Mapped[str] = mapped_column(Text, ForeignKey("parties.id", ondelete="CASCADE"), nullable=False)
    target_party_id: Mapped[str] = mapped_column(Text, ForeignKey("parties.id", ondelete="CASCADE"), nullable=False)
    relation_type: Mapped[str] = mapped_column(Text, nullable=False)

    document = relationship("DocumentModel", back_populates="relations")
    source_party = relationship("PartyModel", foreign_keys=[source_party_id])
    target_party = relationship("PartyModel", foreign_keys=[target_party_id])


class ObligationModel(Base):
    __tablename__ = "obligations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    party_id: Mapped[str] = mapped_column(Text, ForeignKey("parties.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    party = relationship("PartyModel", back_populates="obligations")


class RiskModel(Base):
    __tablename__ = "risks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    party_id: Mapped[str] = mapped_column(Text, ForeignKey("parties.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    risk_level: Mapped[str | None] = mapped_column(Text, nullable=True)

    party = relationship("PartyModel", back_populates="risks")
