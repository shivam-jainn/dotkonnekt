from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Legacy parsed document — kept for backward compat."""

    content: str
    metadata: dict = field(default_factory=dict)
    extracted_images: list[dict] = field(default_factory=list)


class Parser(ABC):
    @abstractmethod
    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...

    @abstractmethod
    def supported_extensions(self) -> list[str]: ...
