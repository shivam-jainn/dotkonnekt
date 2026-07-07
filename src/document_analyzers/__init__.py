from src.document_analyzers.base import BaseAnalyzer
from src.document_analyzers.generic import GenericAnalyzer
from src.document_analyzers.legal import LegalAnalyzer
from src.document_analyzers.insurance import InsuranceAnalyzer
from src.document_analyzers.employment import EmploymentAnalyzer
from src.document_analyzers.finance import FinanceAnalyzer
from src.document_analyzers.technical import TechnicalAnalyzer

def get_analyzer(document_type: str | None) -> BaseAnalyzer:
    if not document_type:
        return GenericAnalyzer()
        
    dt = document_type.lower().strip()
    if dt in ("contract", "nda", "lease", "license", "power_of_attorney", "legal"):
        return LegalAnalyzer()
    elif dt == "insurance":
        return InsuranceAnalyzer()
    elif dt == "employment":
        return EmploymentAnalyzer()
    elif dt in ("invoice", "finance", "billing", "financial"):
        return FinanceAnalyzer()
    elif dt in ("technical", "rfc", "specification", "manual", "research_paper"):
        return TechnicalAnalyzer()
        
    # Default heuristics
    if "policy" in dt or "agreement" in dt:
        return LegalAnalyzer()
        
    return GenericAnalyzer()

__all__ = [
    "BaseAnalyzer",
    "GenericAnalyzer",
    "LegalAnalyzer",
    "InsuranceAnalyzer",
    "EmploymentAnalyzer",
    "FinanceAnalyzer",
    "TechnicalAnalyzer",
    "get_analyzer",
]
