from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    content: str
    metadata: dict = field(default_factory=dict)


class Parser(ABC):
    @abstractmethod
    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...

    @abstractmethod
    def supported_extensions(self) -> list[str]: ...
