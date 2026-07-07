"""Hierarchical semantic chunker.

Chunk hierarchy:
  Heading → Section → Clause → Paragraph → Sentence → Character fallback

Preserves legal clauses, numbered lists, bullets, definitions, and section
boundaries.  Never splits clauses unless unavoidable.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from src.core.document import Chunk, Document, Heading, Page, Paragraph


# ---------------------------------------------------------------------------
# Patterns for structural boundary detection
# ---------------------------------------------------------------------------

# Heading detection (markdown-style, numbered sections, all-caps lines)
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:#{1,6})\s+"  # markdown headings
    r"|(?:ARTICLE|SECTION|CLAUSE|PART|CHAPTER)\s+\S+"  # legal headings
    r"|(?:\d{1,3}(?:\.\d{1,3}){0,3})\s+\S+"  # numbered: "1. Introduction", "1.1 Scope"
    r"|[A-Z][A-Z\s]{3,}$"  # ALL CAPS line (likely a heading)
    r")",
    re.MULTILINE,
)

# Clause / legal-structure boundaries
_CLAUSE_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:\d{1,3}(?:\.\d{1,3}){0,2})\.\s+"  # "1. ", "1.1. "
    r"|(?:\(\s*[a-z]\s*\))\s+"  # "(a) "
    r"|(?:\(\s*\d{1,2}\s*\))\s+"  # "(1) "
    r"|(?:ARTICLE\s+\S+)"  # "ARTICLE IV"
    r"|(?:SECTION\s+\S+)"  # "SECTION 3.1"
    r"|(?:CLAUSE\s+\S+)"
    r"|(?:WHEREAS|NOTWITHSTANDING|PROVIDED\s+THAT|HEREINAFTER|NOW\s+THEREFORE)"
    r")",
    re.IGNORECASE,
)

# List-item boundaries (bullets, numbered)
_LIST_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*•]\s+|(?:\d{1,3})[.)]\s+)",
    re.MULTILINE,
)

# Sentence boundary
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])")

# Paragraph boundary (double newline)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


# ---------------------------------------------------------------------------
# Segment: an atomic piece of content with structural annotations
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    text: str
    page: int
    kind: str  # "heading", "paragraph", "list_item", "clause", "raw"
    heading_level: int | None = None
    section_title: str | None = None


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------


class SemanticChunker:
    """Produce semantically meaningful chunks from a Document IR.

    Parameters
    ----------
    max_chunk_size:
        Target maximum characters per chunk.  Clauses that exceed this are
        split at paragraph / sentence / character boundaries, but never at
        the clause boundary itself unless the clause alone exceeds the limit.
    min_chunk_size:
        Chunks smaller than this are merged with the previous chunk (if possible)
        to avoid tiny fragments.
    chunk_overlap:
        Overlap in characters when falling back to sentence or character splitting.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        chunk_overlap: int = 100,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Produce chunks from a Document IR and attach them to ``document.chunks``."""
        segments = self._extract_segments(document)
        if not segments:
            return []

        sections = self._build_sections(segments)
        raw_chunks = self._hierarchical_split(sections)
        chunks = self._assign_ids_and_metadata(raw_chunks, document)
        document.chunks = chunks
        return chunks

    # ------------------------------------------------------------------
    # Segment extraction from Document IR
    # ------------------------------------------------------------------

    def _extract_segments(self, document: Document) -> list[_Segment]:
        """Walk Document IR pages and extract ordered structural segments."""
        segments: list[_Segment] = []

        for page in document.pages:
            # Use pre-parsed blocks if available
            if page.blocks:
                for block in page.blocks:
                    segments.append(
                        _Segment(
                            text=block.content,
                            page=page.number,
                            kind=block.kind,
                            heading_level=block.level,
                        )
                    )
                continue

            # Fall back to heading / paragraph lists
            if page.headings:
                for h in page.headings:
                    segments.append(
                        _Segment(
                            text=h.text,
                            page=page.number,
                            kind="heading",
                            heading_level=h.level,
                        )
                    )

            if page.paragraphs:
                for p in page.paragraphs:
                    segments.append(
                        _Segment(
                            text=p.text,
                            page=page.number,
                            kind=p.kind,
                        )
                    )
                continue

            # Last resort: split raw page text into paragraphs
            if page.text:
                for para_text in _PARAGRAPH_RE.split(page.text):
                    para_text = para_text.strip()
                    if para_text:
                        segments.append(
                            _Segment(text=para_text, page=page.number, kind="paragraph")
                        )

        return segments

    # ------------------------------------------------------------------
    # Section building from segments
    # ------------------------------------------------------------------

    def _build_sections(self, segments: list[_Segment]) -> list[dict]:
        """Group segments into sections delimited by headings.

        Returns a list of dicts with keys:
          heading, level, page, segments[]
        """
        sections: list[dict] = []
        current: dict | None = None

        for seg in segments:
            if seg.kind == "heading":
                if current is not None:
                    sections.append(current)
                current = {
                    "heading": seg.text,
                    "level": seg.heading_level or 1,
                    "page": seg.page,
                    "segments": [],
                }
            else:
                if current is None:
                    current = {
                        "heading": "",
                        "level": 0,
                        "page": seg.page,
                        "segments": [],
                    }
                current["segments"].append(seg)

        if current is not None:
            sections.append(current)

        return sections

    # ------------------------------------------------------------------
    # Hierarchical splitting
    # ------------------------------------------------------------------

    def _hierarchical_split(self, sections: list[dict]) -> list[dict]:
        """Split each section into chunks respecting the hierarchy.

        Returns list of chunk dicts with keys:
          content, page, section, subsection, clause
        """
        result: list[dict] = []
        prev_chunk_id: str | None = None

        for section in sections:
            section_title = section["heading"]
            section_page = section["page"]

            # Detect clause label from section heading (e.g., "1. Definitions")
            section_clause = self._detect_clause_label_text(section_title) if section_title else None

            clause_groups = self._group_into_clauses(section["segments"])

            for clause_group in clause_groups:
                clause_text = "\n\n".join(s.text for s in clause_group)
                clause_page = clause_group[0].page if clause_group else section_page

                # Prefer clause label from the segment, fall back to section heading label
                clause_label = self._detect_clause_label(clause_group[0]) or section_clause
                group_kind = clause_group[0].kind if clause_group else "paragraph"

                if len(clause_text) <= self.max_chunk_size:
                    chunk_id = uuid.uuid4().hex
                    result.append(
                        {
                            "id": chunk_id,
                            "content": clause_text,
                            "page": clause_page,
                            "section": section_title or None,
                            "subsection": None,
                            "clause": clause_label,
                            "previous_chunk_id": prev_chunk_id,
                            "content_type": group_kind,
                        }
                    )
                    if result and prev_chunk_id:
                        # Update previous chunk's next pointer
                        for r in result:
                            if r["id"] == prev_chunk_id:
                                r["next_chunk_id"] = chunk_id
                                break
                    prev_chunk_id = chunk_id
                else:
                    # Split at paragraph / sentence / character boundaries
                    sub_chunks = self._split_oversized(
                        clause_text, clause_page, section_title
                    )
                    for sc in sub_chunks:
                        chunk_id = uuid.uuid4().hex
                        sc["id"] = chunk_id
                        sc["clause"] = clause_label
                        sc["previous_chunk_id"] = prev_chunk_id
                        sc["content_type"] = group_kind
                        if result and prev_chunk_id:
                            for r in result:
                                if r["id"] == prev_chunk_id:
                                    r["next_chunk_id"] = chunk_id
                                    break
                        prev_chunk_id = chunk_id
                        result.append(sc)

        return result

    def _group_into_clauses(self, segments: list[_Segment]) -> list[list[_Segment]]:
        """Group segments into clause-level units.

        A new clause group starts when we encounter a clause boundary pattern
        or a page change (to preserve per-page tracking).
        """
        if not segments:
            return []

        groups: list[list[_Segment]] = []
        current_group: list[_Segment] = []

        for seg in segments:
            if seg.kind == "heading":
                # Headings always start a new group
                if current_group:
                    groups.append(current_group)
                groups.append([seg])
                current_group = []
            elif self._is_clause_boundary(seg):
                if current_group:
                    groups.append(current_group)
                current_group = [seg]
            elif current_group and seg.page != current_group[0].page:
                # Page change — start a new group to preserve page tracking
                groups.append(current_group)
                current_group = [seg]
            else:
                current_group.append(seg)

        if current_group:
            groups.append(current_group)

        return groups

    def _is_clause_boundary(self, seg: _Segment) -> bool:
        """Check if a segment starts a new clause."""
        text = seg.text.strip()
        if not text:
            return False

        # Check against clause pattern
        if _CLAUSE_RE.match("\n" + text):
            return True

        # Numbered list items at the start of a line
        if _LIST_RE.match(text):
            return True

        # Legal clause markers
        first_words = text.split(None, 1)[:1]
        if first_words and first_words[0].upper() in {
            "WHEREAS", "NOTWITHSTANDING", "PROVIDED", "HEREINAFTER",
            "NOW", "ARTICLE", "SECTION", "CLAUSE",
        }:
            return True

        return False

    def _detect_clause_label(self, seg: _Segment) -> str | None:
        """Extract the clause label from a segment (e.g., '1.1', '(a)', 'ARTICLE IV')."""
        text = seg.text.strip()
        return self._detect_clause_label_text(text)

    def _detect_clause_label_text(self, text: str) -> str | None:
        """Extract the clause label from raw text (e.g., '1. Definitions')."""
        text = text.strip()
        m = re.match(
            r"^((?:\d{1,3}(?:\.\d{1,3}){0,2})\.?|(?:\(\s*[a-z]\s*\))|(?:\(\s*\d{1,2}\s*\))|(?:ARTICLE\s+\S+)|(?:SECTION\s+\S+)|(?:CLAUSE\s+\S+))",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return None

    # ------------------------------------------------------------------
    # Oversized chunk splitting
    # ------------------------------------------------------------------

    def _split_oversized(
        self, text: str, page: int, section: str | None
    ) -> list[dict]:
        """Split text that exceeds max_chunk_size into smaller chunks.

        Priority: paragraph → sentence → character fallback.
        Never produces chunks smaller than min_chunk_size (unless the text is smaller).
        """
        chunks: list[dict] = []

        # Level 1: split at paragraph boundaries
        paragraphs = _PARAGRAPH_RE.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if len(paragraphs) > 1:
            merged = self._merge_small_parts(paragraphs)
            for para in merged:
                if len(para) <= self.max_chunk_size:
                    chunks.append(
                        {
                            "content": para,
                            "page": page,
                            "section": section,
                            "subsection": None,
                            "clause": None,
                        }
                    )
                else:
                    # Level 2: split at sentence boundaries
                    chunks.extend(
                        self._split_by_sentence(para, page, section)
                    )
        else:
            # Single paragraph — try sentence split
            single = paragraphs[0] if paragraphs else text
            if len(single) <= self.max_chunk_size:
                chunks.append(
                    {
                        "content": single,
                        "page": page,
                        "section": section,
                        "subsection": None,
                        "clause": None,
                    }
                )
            else:
                chunks.extend(
                    self._split_by_sentence(single, page, section)
                )

        return chunks

    def _split_by_sentence(
        self, text: str, page: int, section: str | None
    ) -> list[dict]:
        """Split text at sentence boundaries with overlap."""
        sentences = _SENTENCE_RE.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [
                {
                    "content": text[: self.max_chunk_size],
                    "page": page,
                    "section": section,
                    "subsection": None,
                    "clause": None,
                }
            ]

        merged = self._merge_small_parts(sentences)
        chunks: list[dict] = []

        for part in merged:
            if len(part) <= self.max_chunk_size:
                chunks.append(
                    {
                        "content": part,
                        "page": page,
                        "section": section,
                        "subsection": None,
                        "clause": None,
                    }
                )
            else:
                # Level 3: character fallback with overlap
                chunks.extend(
                    self._split_by_character(part, page, section)
                )

        return chunks

    def _split_by_character(
        self, text: str, page: int, section: str | None
    ) -> list[dict]:
        """Last resort: split at character boundaries with overlap."""
        chunks: list[dict] = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk_text = text[start:end]

            # Try to break at a word boundary
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                if last_space > self.max_chunk_size * 0.5:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunks.append(
                {
                    "content": chunk_text.strip(),
                    "page": page,
                    "section": section,
                    "subsection": None,
                    "clause": None,
                }
            )

            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def _merge_small_parts(self, parts: list[str]) -> list[str]:
        """Merge consecutive parts that are too small into single chunks."""
        if not parts:
            return []

        merged: list[str] = []
        current = parts[0]

        for part in parts[1:]:
            candidate = current + "\n\n" + part
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                merged.append(current)
                current = part

        merged.append(current)
        return merged

    # ------------------------------------------------------------------
    # ID assignment and metadata propagation
    # ------------------------------------------------------------------

    def _assign_ids_and_metadata(
        self, raw_chunks: list[dict], document: Document
    ) -> list[Chunk]:
        """Convert raw chunk dicts into typed Chunk objects with final metadata."""
        chunks: list[Chunk] = []

        for i, rc in enumerate(raw_chunks):
            chunk = Chunk(
                id=rc.get("id", uuid.uuid4().hex),
                content=rc["content"],
                index=i,
                page=rc.get("page", 0),
                section=rc.get("section"),
                subsection=rc.get("subsection"),
                clause=rc.get("clause"),
                previous_chunk_id=rc.get("previous_chunk_id"),
                next_chunk_id=rc.get("next_chunk_id"),
                content_type=rc.get("content_type", "paragraph"),
                metadata={
                    "job_id": document.metadata.get("job_id", ""),
                    "filename": document.filename,
                },
            )
            chunks.append(chunk)

        # Patch next_chunk_id for the last chunk
        for i, chunk in enumerate(chunks):
            if i > 0 and chunk.previous_chunk_id is None:
                chunk.previous_chunk_id = chunks[i - 1].id
            if i < len(chunks) - 1 and chunk.next_chunk_id is None:
                chunk.next_chunk_id = chunks[i + 1].id

        return chunks


# ---------------------------------------------------------------------------
# Legacy adapter: satisfies the old Chunker ABC interface so existing code
# that calls chunker.chunk(text, metadata) still works.
# ---------------------------------------------------------------------------


class TextChunker:
    """Drop-in replacement that wraps SemanticChunker.

    Accepts the old ``chunk(text, metadata)`` interface and delegates to
    the semantic chunker via a synthetic Document IR.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        min_chunk_size: int = 100,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators  # accepted but ignored by semantic chunker
        self._semantic = SemanticChunker(
            max_chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, text: str, metadata: dict | None = None) -> list:
        """Legacy interface: accepts raw text, returns list of Chunk objects."""
        from src.core.chunkers.base import Chunk as LegacyChunk

        if not text.strip():
            return []

        doc = Document(
            filename=metadata.get("filename", "") if metadata else "",
            metadata=metadata or {},
            pages=[
                Page(
                    number=1,
                    text=text,
                )
            ],
        )
        self._semantic.chunk_document(doc)

        # Convert to legacy Chunk format for backward compat
        result: list[LegacyChunk] = []
        for c in doc.chunks:
            result.append(
                LegacyChunk(
                    content=c.content,
                    index=c.index,
                    metadata={**c.metadata, "page": c.page, **(metadata or {})},
                )
            )
        return result
