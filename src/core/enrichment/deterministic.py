"""Deterministic metadata enrichment using regex and heuristics.

Extracts: document type, entities, dates, money, deadlines, parties,
jurisdictions, obligations, rights, exclusions, definitions, risks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.document import Document, Entity, ExtractedMetadata, DerivedMetadata
from src.document_analyzers import get_analyzer



# Dates
_DATE_PATTERNS = [
    # ISO: 2024-01-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # US: 01/15/2024, 1/15/2024
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    # Written: January 15, 2024 / 15 January 2024
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Relative: "within 30 days", "within sixty (60) days"
    re.compile(
        r"\bwithin\s+(\d+\s+(?:days|business\s+days|calendar\s+days|months|years))\b",
        re.IGNORECASE,
    ),
]

# Money / monetary amounts
_MONEY_RE = re.compile(
    r"(?:"
    r"[$€£¥]\s*\d[\d,]*(?:\.\d{2})?"  # $1,000.00
    r"|\d[\d,]*(?:\.\d{2})?\s*(?:dollars?|euros?|pounds?|yen|USD|EUR|GBP|JPY)"
    r"|(?:USD|EUR|GBP|JPY)\s*\d[\d,]*(?:\.\d{2})?"
    r")",
    re.IGNORECASE,
)

# Deadlines / time-bound obligations
_DEADLINE_RE = re.compile(
    r"(?:"
    r"(?:shall|must|will|should)\s+(?:be\s+)?(?:completed|delivered|paid|submitted|executed|received|notified|provided|made)\s+"
    r"(?:within|by|before|after|on|no\s+later\s+than)\s+"
    r"[^.]{5,60}"
    r"|within\s+\d+\s+(?:days|business\s+days|months|years)"
    r"|no\s+later\s+than\s+[^.]{5,60}"
    r"|by\s+(?:the\s+)?(?:end|close)\s+of\s+[^.]{5,40}"
    r")",
    re.IGNORECASE,
)

# Parties (X Corp, ABC Inc., Ltd., LLC, etc.)
_PARTY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\s.,]{1,50}?\s+(?:Corp(?:oration)?|Inc\.?|Ltd\.?|LLC|L\.?L\.?C\.?|Co\.?|Company|Partners?|Associates?|Group|Holdings?|PLC|GmbH|S\.?A\.?|B\.?V\.?))\b"
)

# Jurisdictions
_JURISDICTION_RE = re.compile(
    r"(?:"
    r"(?:governed\s+by|subject\s+to|under\s+the\s+laws?\s+of|jurisdiction\s+of|in\s+the\s+state\s+of|courts?\s+of)\s+"
    r"([A-Z][A-Za-z\s,]{2,40})"
    r"|(?:State\s+of|Commonwealth\s+of|Province\s+of)\s+([A-Z][A-Za-z\s]{2,30})"
    r"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*(?:USA|United\s+States|U\.S\.A?\.)"
    r")",
    re.IGNORECASE,
)

# Obligations
_OBLIGATION_RE = re.compile(
    r"(?:shall|must|will\s+be\s+required\s+to|is\s+obligated\s+to|agrees?\s+to|undertakes?\s+to|covenants?\s+to|warrants?\s+that)[^.]{10,200}\.",
    re.IGNORECASE,
)

# Rights
_RIGHT_RE = re.compile(
    r"(?:may|is\s+entitled\s+to|has\s+the\s+right\s+to|is\s+authorized\s+to|reserves?\s+the\s+right)[^.]{10,200}\.",
    re.IGNORECASE,
)

# Exclusions / limitations
_EXCLUSION_RE = re.compile(
    r"(?:"
    r"shall\s+not|does\s+not|excluding|except\s+(?:as|for|when)|"
    r"in\s+no\s+event\s+shall|no\s+liability|not\s+responsible|"
    r"disclaim|limitation\s+of\s+liability|not\s+liable|"
    r"not\s+be\s+liable"
    r")[^.]{10,200}\.",
    re.IGNORECASE,
)

# Definitions (often in "Definitions" section or "X means" / "X shall mean")
_DEFINITION_RE = re.compile(
    r"(?:"
    r'"([^"]+)"\s+(?:means?|shall\s+mean|refers?\s+to)\s+[^.]{10,200}\.'
    r"|([A-Z][A-Za-z]+)\s+(?:means?|shall\s+mean)\s+[^.]{10,200}\."
    r")",
    re.IGNORECASE,
)

# Risky terms
_RISK_RE = re.compile(
    r"(?:"
    r"indemnif|hold\s+harmless|consequential\s+damages?|penalt|liquidated\s+damages?|"
    r"terminate\s+immediately|breach|default|enjoin|injunctive|"
    r"representations?\s+and\s+warranties?|at\s+its\s+sole\s+discretion|"
    r"may\s+terminate|without\s+notice|non[- ]?compete|non[- ]?solicitation|"
    r"confidential|trade\s+secret|intellectual\s+property|"
    r"not\s+be\s+liable|not\s+liable"
    r")",
    re.IGNORECASE,
)

# Document type detection
_DOC_TYPE_PATTERNS = {
    "nda": re.compile(r"\b(?:non[- ]?disclosure|NDA|confidential(?:ity)?)\b", re.IGNORECASE),
    "contract": re.compile(r"\b(?:agreement|contract|terms\s+and\s+conditions)\b", re.IGNORECASE),
    "employment": re.compile(r"\b(?:employment|employee|employer|salary|compensation)\b", re.IGNORECASE),
    "lease": re.compile(r"\b(?:lease|tenant|landlord|rental|lessor|lessee)\b", re.IGNORECASE),
    "license": re.compile(r"\b(?:license|licensor|licensee|royalt)\b", re.IGNORECASE),
    "policy": re.compile(r"\b(?:policy|guidelines?|procedure|compliance)\b", re.IGNORECASE),
    "invoice": re.compile(r"\b(?:invoice|billing|payment\s+due|amount\s+due)\b", re.IGNORECASE),
    "power_of_attorney": re.compile(r"\b(?:power\s+of\s+attorney|POA|attorney[- ]in[- ]fact)\b", re.IGNORECASE),
}


_DOC_TYPE_ROLES_MAP = {
    "nda": ["Disclosing Party", "Receiving Party", "Party A", "Party B"],
    "employment": ["Employer", "Employee", "Company"],
    "lease": ["Landlord", "Tenant", "Lessor", "Lessee"],
    "license": ["Licensor", "Licensee"],
    "policy": ["Insurer", "Insured", "Insuree", "Policyholder", "Company"],
    "contract": ["Party A", "Party B", "Party C", "Contractor", "Client", "Customer", "Vendor", "Supplier", "Buyer", "Seller"],
    "generic": ["Party A", "Party B", "Party C"],
}


def _attribute_chunk_content(
    content: str,
    doc_type: str | None,
    document_parties: list[str],
) -> tuple[dict[str, list[str]], list[dict], list[dict]]:
    """Attributes sentences, obligations, and risks to specific party roles."""
    party_sentences = {}
    obligations_by_party = []
    risks_by_party = []

    # Get roles to scan for
    roles = _DOC_TYPE_ROLES_MAP.get(doc_type or "generic", ["Party A", "Party B", "Party C"])
    
    # Split text into sentences/paragraphs
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', content) if s.strip()]

    # Attribute sentences
    for sentence in sentences:
        matched_roles = []
        for role in roles:
            role_pat = r'\b' + re.escape(role.lower()) + r'\b'
            if re.search(role_pat, sentence.lower()):
                matched_roles.append(role)
        
        for p_name in document_parties:
            if p_name.lower() in sentence.lower():
                matched_roles.append(p_name)

        for role_or_name in set(matched_roles):
            party_sentences.setdefault(role_or_name, []).append(sentence)

    # Attribute obligations
    obligations = list({m.group(0).strip() for m in _OBLIGATION_RE.finditer(content)})
    for ob in obligations:
        matched_roles = []
        for role in roles:
            role_pat = r'\b' + re.escape(role.lower()) + r'\b'
            if re.search(role_pat, ob.lower()):
                matched_roles.append(role)
        for p_name in document_parties:
            if p_name.lower() in ob.lower():
                matched_roles.append(p_name)
        
        if matched_roles:
            for role_or_name in set(matched_roles):
                obligations_by_party.append({"party_role": role_or_name, "text": ob})
        else:
            obligations_by_party.append({"party_role": "General", "text": ob})

    # Attribute risks
    risks = list({m.group(0).strip() for m in _RISK_RE.finditer(content)})
    for rk in risks:
        matched_roles = []
        for role in roles:
            role_pat = r'\b' + re.escape(role.lower()) + r'\b'
            if re.search(role_pat, rk.lower()):
                matched_roles.append(role)
        for p_name in document_parties:
            if p_name.lower() in rk.lower():
                matched_roles.append(p_name)
        
        if matched_roles:
            for role_or_name in set(matched_roles):
                risks_by_party.append({"party_role": role_or_name, "description": rk})
        else:
            risks_by_party.append({"party_role": "General", "description": rk})

    return party_sentences, obligations_by_party, risks_by_party


# ---------------------------------------------------------------------------
# DeterministicEnricher
# ---------------------------------------------------------------------------


class DeterministicEnricher:
    """Extract structured metadata from Document IR using regex and heuristics.

    Prefer this over LLM-based extraction for deterministic, fast, and
    cost-free metadata.  Only fall back to LLM when heuristic extraction
    cannot infer the metadata.
    """

    def enrich_document(self, document: Document) -> ExtractedMetadata:
        """Run full deterministic enrichment on a Document IR."""
        text = document.raw_text
        metadata = ExtractedMetadata()

        doc_type = self._detect_document_type(text)
        metadata.document_type = doc_type
        metadata.entities = self._extract_entities(text, document)
        metadata.dates = self._extract_dates(text)
        metadata.money = self._extract_money(text)
        metadata.deadlines = self._extract_deadlines(text)
        metadata.parties = self._extract_parties(text)
        metadata.jurisdictions = self._extract_jurisdictions(text)
        metadata.obligations = self._extract_obligations(text)
        metadata.rights = self._extract_rights(text)
        metadata.exclusions = self._extract_exclusions(text)
        metadata.definitions = self._extract_definitions(text)
        metadata.risks = self._extract_risks(text)

        document.extracted_metadata = metadata
        document.entities = metadata.entities

        # Enrich individual chunks using DocumentAnalyzer based on document_type
        analyzer = get_analyzer(doc_type)
        document_parties = metadata.parties
        for chunk in document.chunks:
            analysis = analyzer.analyze_text(chunk.content)
            
            # Sentence/paragraph-level party attribution
            party_sentences, ob_by_party, rk_by_party = _attribute_chunk_content(
                chunk.content, doc_type, document_parties
            )
            
            chunk.derived_metadata = DerivedMetadata(
                summary=None, # summary is built separately if needed
                keywords=analysis.get("keywords", []),
                obligations=analysis.get("obligations", []),
                risks=analysis.get("risks", []),
                entities=analysis.get("parties", []) + analysis.get("dates", []) + analysis.get("jurisdictions", []),
                topics=analysis.get("topics", []),
                deadlines=analysis.get("deadlines", []),
                rights=analysis.get("rights", []),
                exclusions=analysis.get("exclusions", []),
                definitions=analysis.get("definitions", []),
                parties=analysis.get("parties", []),
                jurisdictions=analysis.get("jurisdictions", []),
                document_type=doc_type,
                party_sentences=party_sentences,
                obligations_by_party=ob_by_party,
                risks_by_party=rk_by_party,
            )

        return metadata

    def enrich_chunk_text(self, text: str, document_type: str | None = None) -> dict:
        """Extract metadata from a single chunk's text."""
        analyzer = get_analyzer(document_type)
        return analyzer.analyze_text(text)

    # ------------------------------------------------------------------
    # Document type
    # ------------------------------------------------------------------

    def _detect_document_type(self, text: str) -> str | None:
        scores: dict[str, int] = {}
        for dtype, pattern in _DOC_TYPE_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                scores[dtype] = len(matches)
        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]
        return None

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str, document: Document) -> list[Entity]:
        entities: list[Entity] = []

        # Organizations
        for match in _PARTY_RE.finditer(text):
            entities.append(
                Entity(name=match.group(1).strip(), kind="organization")
            )

        # Dates
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                val = match.group(1) if match.lastindex else match.group(0)
                entities.append(Entity(name=val, kind="date", value=val))

        # Money
        for match in _MONEY_RE.finditer(text):
            entities.append(Entity(name=match.group(0).strip(), kind="money"))

        # Jurisdictions
        for match in _JURISDICTION_RE.finditer(text):
            name = match.group(1) or match.group(2) or match.group(3)
            if name:
                entities.append(Entity(name=name.strip(), kind="jurisdiction"))

        return entities

    # ------------------------------------------------------------------
    # Individual extractors
    # ------------------------------------------------------------------

    def _extract_dates(self, text: str) -> list[str]:
        dates: list[str] = []
        for pattern in _DATE_PATTERNS:
            for match in pattern.finditer(text):
                val = match.group(1) if match.lastindex else match.group(0)
                if val not in dates:
                    dates.append(val.strip())
        return dates

    def _extract_money(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _MONEY_RE.finditer(text)})

    def _extract_deadlines(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _DEADLINE_RE.finditer(text)})

    def _extract_parties(self, text: str) -> list[str]:
        return list({m.group(1).strip() for m in _PARTY_RE.finditer(text)})

    def _extract_jurisdictions(self, text: str) -> list[str]:
        jurisdictions: list[str] = []
        for match in _JURISDICTION_RE.finditer(text):
            name = match.group(1) or match.group(2) or match.group(3)
            if name and name.strip() not in jurisdictions:
                jurisdictions.append(name.strip())
        return jurisdictions

    def _extract_obligations(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _OBLIGATION_RE.finditer(text)})

    def _extract_rights(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _RIGHT_RE.finditer(text)})

    def _extract_exclusions(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _EXCLUSION_RE.finditer(text)})

    def _extract_definitions(self, text: str) -> list[str]:
        defs: list[str] = []
        for match in _DEFINITION_RE.finditer(text):
            name = match.group(1) or match.group(2)
            if name:
                defs.append(name.strip())
        return defs

    def _extract_risks(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _RISK_RE.finditer(text)})
