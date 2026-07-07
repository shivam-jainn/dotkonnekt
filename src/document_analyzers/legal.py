import re
from src.document_analyzers.generic import GenericAnalyzer

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

class LegalAnalyzer(GenericAnalyzer):
    def analyze_text(self, text: str) -> dict:
        # Get base generic analysis first (dates, money, parties, jurisdictions)
        analysis = super().analyze_text(text)
        
        # Add legal-specific annotations
        analysis.update({
            "obligations": self._extract_obligations(text),
            "rights": self._extract_rights(text),
            "exclusions": self._extract_exclusions(text),
            "definitions": self._extract_definitions(text),
            "risks": self._extract_risks(text),
            "deadlines": self._extract_deadlines(text),
        })
        return analysis

    def _extract_deadlines(self, text: str) -> list[str]:
        return list({m.group(0).strip() for m in _DEADLINE_RE.finditer(text)})

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
