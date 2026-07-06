from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    index: int
    metadata: dict = field(default_factory=dict)


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]: ...
