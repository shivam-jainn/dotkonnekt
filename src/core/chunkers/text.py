from src.core.chunkers.base import Chunk, Chunker


class TextChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text.strip():
            return []

        raw_chunks = self._split_text(text)
        chunks: list[Chunk] = []

        for idx, raw_chunk in enumerate(raw_chunks):
            cleaned = raw_chunk.strip()
            if cleaned:
                chunk_metadata = {**(metadata or {}), "chunk_index": idx}
                chunks.append(
                    Chunk(
                        content=cleaned,
                        index=idx,
                        metadata=chunk_metadata,
                    )
                )

        return chunks

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        if len(text) <= self.chunk_size:
            return [text]

        sep = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(sep)
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    if self.chunk_overlap > 0 and current:
                        overlap_text = current[-self.chunk_overlap :]
                        current = overlap_text + sep + part
                        if len(current) <= self.chunk_size:
                            continue
                        else:
                            chunks.append(current)
                            current = part
                    else:
                        current = part
                else:
                    if len(part) > self.chunk_size:
                        chunks.extend(self._recursive_split(part, remaining_separators))
                    else:
                        current = part

        if current:
            chunks.append(current)

        return chunks
