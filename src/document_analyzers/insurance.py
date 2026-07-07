import re
from src.document_analyzers.legal import LegalAnalyzer

_DEDUCTIBLE_RE = re.compile(
    r"(?:deductible|copay|co-insurance|out-of-pocket|premium|limit\s+of\s+liability)[^.]{5,100}",
    re.IGNORECASE,
)

class InsuranceAnalyzer(LegalAnalyzer):
    def analyze_text(self, text: str) -> dict:
        analysis = super().analyze_text(text)
        # Extra insurance-specific annotations (deductibles, policy rules)
        deductibles = list({m.group(0).strip() for m in _DEDUCTIBLE_RE.finditer(text)})
        if deductibles:
            analysis["risks"].extend(deductibles)
        return analysis
