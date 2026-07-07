import re
from src.document_analyzers.legal import LegalAnalyzer

_FINANCE_RE = re.compile(
    r"(?:interest\s+rate|principal|amortization|collateral|maturity\s+date|default\s+rate|loan|credit|debt|borrower|lender)[^.]{5,100}",
    re.IGNORECASE,
)

class FinanceAnalyzer(LegalAnalyzer):
    def analyze_text(self, text: str) -> dict:
        analysis = super().analyze_text(text)
        finance_terms = list({m.group(0).strip() for m in _FINANCE_RE.finditer(text)})
        if finance_terms:
            analysis["obligations"].extend(finance_terms)
        return analysis
