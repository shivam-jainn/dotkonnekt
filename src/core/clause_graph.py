"""Clause graph construction during ingestion.

Builds a graph linking chunks to their parent section, parent clause,
previous/next clause, and cross-references.  Stored as graph metadata
for retrieval and navigation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.document import Chunk, Clause, ClauseEdge, ClauseGraph, Document


def build_clause_graph(document: Document) -> ClauseGraph:
    """Construct a clause graph from a Document IR's chunks.

    Each chunk that has a ``clause`` label becomes a node in the graph.
    Edges encode parent section, previous/next clause, and cross-references.
    """
    graph = ClauseGraph()

    # Index chunks by clause label
    clause_chunks: dict[str, list[Chunk]] = {}
    for chunk in document.chunks:
        label = chunk.clause
        if label:
            clause_chunks.setdefault(label, []).append(chunk)

    # Create clause nodes
    clause_ids: list[str] = []
    for label, chunks in clause_chunks.items():
        clause = Clause(
            id=chunks[0].id,
            text="\n\n".join(c.content for c in chunks),
            section_id=chunks[0].section,
            page=chunks[0].page,
            kind=_classify_clause_kind(chunks[0].content),
        )
        graph.add_clause(clause)
        clause_ids.append(clause.id)

        # Link chunks to clause
        for chunk in chunks:
            chunk.clause_id = clause.id

    # Link clauses sequentially within the same section
    section_groups: dict[str | None, list[str]] = {}
    for cid in clause_ids:
        clause = graph.clauses[cid]
        section_groups.setdefault(clause.section_id, []).append(cid)

    for section_id, group_ids in section_groups.items():
        graph.link_chain(group_ids)
        for cid in group_ids:
            graph.edges[cid].parent_section = section_id

    # Detect cross-references (e.g., "see Section 3.1", "as defined in Clause 4")
    _detect_cross_references(graph, document)

    document.clause_graph = graph
    return graph


def _classify_clause_kind(text: str) -> str:
    """Classify a clause into its semantic kind."""
    lower = text.lower()

    if re.search(r"\b(?:means?|shall\s+mean|refers?\s+to)\b", lower):
        return "definition"
    # Check exclusions BEFORE obligations since "shall not" should not match "shall"
    if re.search(r"\b(?:shall\s+not|does\s+not|excluding|except|not\s+(?:be\s+)?liable|disclaim)\b", lower):
        return "exclusion"
    if re.search(r"\b(?:shall|must|will\s+be\s+required|is\s+obligated)\b", lower):
        return "obligation"
    if re.search(r"\b(?:may|is\s+entitled|has\s+the\s+right|reserves?\s+the\s+right)\b", lower):
        return "right"
    return "clause"


def _detect_cross_references(graph: ClauseGraph, document: Document) -> None:
    """Find cross-references between clauses (e.g., 'see Section 3.1')."""
    ref_pattern = re.compile(
        r"(?:see|as\s+defined\s+in|pursuant\s+to|subject\s+to|in\s+accordance\s+with)\s+"
        r"((?:Section|Clause|Article|Paragraph)\s+[\d.]+)",
        re.IGNORECASE,
    )

    # Build a lookup from section titles to clause IDs
    section_to_clause: dict[str, str] = {}
    for cid, clause in graph.clauses.items():
        if clause.section_id:
            section_to_clause[clause.section_id.lower()] = cid

    for cid, clause in graph.clauses.items():
        for match in ref_pattern.finditer(clause.text):
            ref_label = match.group(1).strip().lower()
            # Try to find the referenced clause
            for ref_cid, ref_clause in graph.clauses.items():
                if ref_cid == cid:
                    continue
                if ref_label in ref_clause.section_id.lower() if ref_clause.section_id else False:
                    if ref_cid not in graph.edges[cid].cross_references:
                        graph.edges[cid].cross_references.append(ref_cid)
