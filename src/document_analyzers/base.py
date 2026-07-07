from abc import ABC, abstractmethod

class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze_text(self, text: str) -> dict:
        """Extract metadata from a chunk of text.
        
        Returns a dictionary containing:
            obligations, rights, exclusions, definitions, risks,
            dates, money, deadlines, parties, jurisdictions, etc.
        """
        pass
