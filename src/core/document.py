"""Document Intermediate Representation (Document IR).

The canonical typed representation of every uploaded document.
All pipeline stages consume and produce Document IR instead of raw strings.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Structural primitives
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """A structural block within a page."""

    kind: str  # "heading", "paragraph", "table", "list", "code", "image"
    content: str
    level: int | None = None  # heading level (1-6)
    page: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Heading:
    text: str
    level: int  # 1-6
    page: int = 0


@dataclass
class Paragraph:
    text: str
    page: int = 0
    kind: str = "body"  # "body", "definition", "bullet", "numbered", "clause"


@dataclass
class Table:
    content: str  # markdown representation
    page: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Image:
    bytes: bytes = b""
    ext: str = ""
    page: int = 0
    index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Annotation:
    kind: str  # "highlight", "underline", "strikeout"
    text: str
    page: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@dataclass
class Page:
    number: int
    blocks: list[Block] = field(default_factory=list)
    headings: list[Heading] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    text: str = ""  # raw text for backward compat


# ---------------------------------------------------------------------------
# Sections & Clauses
# ---------------------------------------------------------------------------


@dataclass
class Section:
    heading: Heading
    subsections: list[Section] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0


@dataclass
class Clause:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ""
    section_id: str | None = None
    parent_clause_id: str | None = None
    page: int = 0
    kind: str = "clause"  # "clause", "definition", "obligation", "right", "exclusion"
    metadata: dict = field(default_factory=dict)


@dataclass
class ClauseEdge:
    parent_section: str | None = None
    parent_clause: str | None = None
    previous_clause: str | None = None
    next_clause: str | None = None
    cross_references: list[str] = field(default_factory=list)


@dataclass
class ClauseGraph:
    clauses: dict[str, Clause] = field(default_factory=dict)
    edges: dict[str, ClauseEdge] = field(default_factory=dict)

    def add_clause(self, clause: Clause, edge: ClauseEdge | None = None) -> None:
        self.clauses[clause.id] = clause
        self.edges[clause.id] = edge or ClauseEdge()

    def link_chain(self, clause_ids: list[str]) -> None:
        """Link clauses in sequential order (previous/next)."""
        for i, cid in enumerate(clause_ids):
            if cid not in self.edges:
                self.edges[cid] = ClauseEdge()
            if i > 0:
                self.edges[cid].previous_clause = clause_ids[i - 1]
            if i < len(clause_ids) - 1:
                self.edges[cid].next_clause = clause_ids[i + 1]


# ---------------------------------------------------------------------------
# Entities & Extracted Metadata
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    name: str
    kind: str  # "person", "organization", "date", "money", "jurisdiction", "deadline"
    value: str | None = None
    page: int | None = None
    chunk_id: str | None = None


@dataclass
class ExtractedMetadata:
    document_type: str | None = None  # "contract", "nda", "agreement", "policy"
    entities: list[Entity] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    rights: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    money: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    parties: list[str] = field(default_factory=list)
    jurisdictions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Semantic Metadata (optional LLM enrichment)
# ---------------------------------------------------------------------------


@dataclass
class DerivedMetadata:
    summary: str | None = None
    keywords: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    deadlines: list[str] = field(default_factory=list)
    rights: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    parties: list[str] = field(default_factory=list)
    jurisdictions: list[str] = field(default_factory=list)
    document_type: str | None = None
    party_sentences: dict[str, list[str]] = field(default_factory=dict)
    obligations_by_party: list[dict] = field(default_factory=list)
    risks_by_party: list[dict] = field(default_factory=list)

SemanticMetadata = DerivedMetadata


# ---------------------------------------------------------------------------
# Chunk (the unit that gets embedded and stored)
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    index: int = 0
    page: int = 0
    section: str | None = None
    subsection: str | None = None
    clause: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    clause_id: str | None = None
    derived_metadata: DerivedMetadata | None = None
    content_type: str | None = None
    metadata: dict = field(default_factory=dict)
    semantic_metadata: DerivedMetadata | None = None

    def __post_init__(self):
        if self.semantic_metadata is not None and self.derived_metadata is None:
            self.derived_metadata = self.semantic_metadata
        elif self.derived_metadata is not None and self.semantic_metadata is None:
            self.semantic_metadata = self.derived_metadata

    def to_payload(self) -> dict:
        """Serialize to a flat dict for Qdrant payload storage."""
        payload: dict = {
            "id": self.id,
            "content": self.content,
            "index": self.index,
            "page": self.page,
            "section": self.section,
            "subsection": self.subsection,
            "clause": self.clause,
            "previous_chunk": self.previous_chunk_id,
            "next_chunk": self.next_chunk_id,
            "content_type": self.content_type,
            **self.metadata,
        }
        if self.clause_id:
            payload["clause_id"] = self.clause_id
        if self.derived_metadata:
            dm = self.derived_metadata
            if dm.summary:
                payload["summary"] = dm.summary
            if dm.keywords:
                payload["keywords"] = dm.keywords
            if dm.obligations:
                payload["obligations"] = dm.obligations
            if dm.risks:
                payload["risks"] = dm.risks
            if dm.entities:
                payload["entities"] = dm.entities
            if dm.topics:
                payload["topics"] = dm.topics
            if dm.deadlines:
                payload["deadlines"] = dm.deadlines
            if dm.rights:
                payload["rights"] = dm.rights
            if dm.exclusions:
                payload["exclusions"] = dm.exclusions
            if dm.definitions:
                payload["definitions"] = dm.definitions
            if dm.parties:
                payload["parties"] = dm.parties
            if dm.jurisdictions:
                payload["jurisdictions"] = dm.jurisdictions
            if dm.document_type:
                payload["document_type"] = dm.document_type
            if dm.party_sentences:
                payload["party_sentences"] = dm.party_sentences
            if dm.obligations_by_party:
                payload["obligations_by_party"] = dm.obligations_by_party
            if dm.risks_by_party:
                payload["risks_by_party"] = dm.risks_by_party
        return payload


@dataclass
class AnalysisChunk:
    id: str
    content: str
    index: int
    page: int = 0
    section: str | None = None
    subsection: str | None = None
    clause: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    clause_id: str | None = None
    derived_metadata: DerivedMetadata | None = None
    content_type: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "content": self.content,
            "index": self.index,
            "page": self.page,
            "section": self.section,
            "subsection": self.subsection,
            "clause": self.clause,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "clause_id": self.clause_id,
            "content_type": self.content_type,
            "metadata": self.metadata,
        }
        if self.derived_metadata:
            dm = self.derived_metadata
            d["derived_metadata"] = {
                "summary": dm.summary,
                "keywords": dm.keywords,
                "obligations": dm.obligations,
                "risks": dm.risks,
                "entities": dm.entities,
                "topics": dm.topics,
                "deadlines": dm.deadlines,
                "rights": dm.rights,
                "exclusions": dm.exclusions,
                "definitions": dm.definitions,
                "parties": dm.parties,
                "jurisdictions": dm.jurisdictions,
                "document_type": dm.document_type,
                "party_sentences": dm.party_sentences,
                "obligations_by_party": dm.obligations_by_party,
                "risks_by_party": dm.risks_by_party,
            }
        else:
            d["derived_metadata"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'AnalysisChunk':
        dm_dict = d.get("derived_metadata")
        dm = None
        if dm_dict:
            dm = DerivedMetadata(
                summary=dm_dict.get("summary"),
                keywords=dm_dict.get("keywords", []),
                obligations=dm_dict.get("obligations", []),
                risks=dm_dict.get("risks", []),
                entities=dm_dict.get("entities", []),
                topics=dm_dict.get("topics", []),
                deadlines=dm_dict.get("deadlines", []),
                rights=dm_dict.get("rights", []),
                exclusions=dm_dict.get("exclusions", []),
                definitions=dm_dict.get("definitions", []),
                parties=dm_dict.get("parties", []),
                jurisdictions=dm_dict.get("jurisdictions", []),
                document_type=dm_dict.get("document_type"),
                party_sentences=dm_dict.get("party_sentences", {}),
                obligations_by_party=dm_dict.get("obligations_by_party", []),
                risks_by_party=dm_dict.get("risks_by_party", []),
            )
        return cls(
            id=d["id"],
            content=d["content"],
            index=d["index"],
            page=d.get("page", 0),
            section=d.get("section"),
            subsection=d.get("subsection"),
            clause=d.get("clause"),
            previous_chunk_id=d.get("previous_chunk_id"),
            next_chunk_id=d.get("next_chunk_id"),
            clause_id=d.get("clause_id"),
            content_type=d.get("content_type"),
            metadata=d.get("metadata", {}),
            derived_metadata=dm,
        )


# ---------------------------------------------------------------------------
# Document (the root IR)
# ---------------------------------------------------------------------------


@dataclass
class Document:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    filename: str = ""
    metadata: dict = field(default_factory=dict)
    pages: list[Page] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    clause_graph: ClauseGraph = field(default_factory=ClauseGraph)
    entities: list[Entity] = field(default_factory=list)
    extracted_metadata: ExtractedMetadata = field(default_factory=ExtractedMetadata)
    chunks: list[Chunk] = field(default_factory=list)
    raw_text: str = ""  # backward compat — full concatenated text

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    def get_page_text(self, page_number: int) -> str:
        for page in self.pages:
            if page.number == page_number:
                return page.text
        return ""
