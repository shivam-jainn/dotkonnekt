import re
from src.document_analyzers.base import BaseAnalyzer

# Dates
_DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
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

class GenericAnalyzer(BaseAnalyzer):
    def analyze_text(self, text: str) -> dict:
        return {
            "obligations": [],
            "rights": [],
            "exclusions": [],
            "definitions": [],
            "risks": [],
            "dates": self._extract_dates(text),
            "money": self._extract_money(text),
            "deadlines": [],
            "parties": self._extract_parties(text),
            "jurisdictions": self._extract_jurisdictions(text),
        }

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

    def _extract_parties(self, text: str) -> list[str]:
        return list({m.group(1).strip() for m in _PARTY_RE.finditer(text)})

    def _extract_jurisdictions(self, text: str) -> list[str]:
        jurisdictions: list[str] = []
        for match in _JURISDICTION_RE.finditer(text):
            name = match.group(1) or match.group(2) or match.group(3)
            if name and name.strip() not in jurisdictions:
                jurisdictions.append(name.strip())
        return jurisdictions
