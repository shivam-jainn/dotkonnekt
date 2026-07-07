import re
from src.document_analyzers.generic import GenericAnalyzer

_TECH_RE = re.compile(
    r"(?:specification|protocol|rfc|architecture|interface|api|latency|throughput|memory|cpu|bandwidth|error\s+rate|schema)[^.]{5,100}",
    re.IGNORECASE,
)

class TechnicalAnalyzer(GenericAnalyzer):
    def analyze_text(self, text: str) -> dict:
        analysis = super().analyze_text(text)
        tech_terms = list({m.group(0).strip() for m in _TECH_RE.finditer(text)})
        # For technical documents, "risks" might mean performance constraints or bottlenecks
        if tech_terms:
            analysis["risks"].extend(tech_terms)
        return analysis
