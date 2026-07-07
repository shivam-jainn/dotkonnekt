import re
from src.document_analyzers.legal import LegalAnalyzer

_COMPENSATION_RE = re.compile(
    r"(?:salary|compensation|benefits|commission|bonus|vesting|severance|stock\s+options)[^.]{5,100}",
    re.IGNORECASE,
)

class EmploymentAnalyzer(LegalAnalyzer):
    def analyze_text(self, text: str) -> dict:
        analysis = super().analyze_text(text)
        comp = list({m.group(0).strip() for m in _COMPENSATION_RE.finditer(text)})
        if comp:
            analysis["obligations"].extend(comp)
        return analysis
