import logging
from dataclasses import dataclass, field

from src.core.chunkers.base import Chunk
from src.core.chunkers.text import TextChunker
from src.core.embedders.embedder import EmbeddedChunk, Embedder
from src.core.parsers.base import ParsedDocument
from src.core.parsers.pdf import PDFParser
from src.storage import create_storage

logger = logging.getLogger(__name__)

PARSERS = {
    ".pdf": PDFParser(),
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 100


@dataclass
class PipelineResult:
    job_id: str
    total_chunks: int = 0
    embedded_chunks: list[EmbeddedChunk] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self) -> None:
        self.storage = create_storage()
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.embedder = Embedder(batch_size=EMBED_BATCH_SIZE)

    def _get_parser(self, filename: str):
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        parser = PARSERS.get(ext)
        if parser is None:
            raise ValueError(f"No parser found for extension '{ext}'")
        return parser

    def _parse(self, data: bytes, filename: str) -> ParsedDocument:
        parser = self._get_parser(filename)
        return parser.parse(data, filename)

    def _chunk(self, document: ParsedDocument) -> list[Chunk]:
        return self.chunker.chunk(document.content, document.metadata)

    async def _embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return await self.embedder.embed_chunks(chunks)

    async def run(self, job_id: str, files: list[dict]) -> PipelineResult:
        result = PipelineResult(job_id=job_id)

        for file_info in files:
            try:
                filename = file_info["filename"]
                storage_path = file_info["storage_path"]

                logger.info("Downloading %s from storage", filename)
                data = self.storage.download_bytes(storage_path)

                logger.info("Parsing %s", filename)
                document = self._parse(data, filename)

                logger.info("Chunking %s", filename)
                chunks = self._chunk(document)
                result.total_chunks += len(chunks)

                logger.info("Embedding %d chunks from %s", len(chunks), filename)
                embedded = await self._embed(chunks)
                result.embedded_chunks.extend(embedded)

                logger.info("Processed %s: %d chunks embedded", filename, len(embedded))

            except Exception as e:
                error_msg = (
                    f"Error processing {file_info.get('filename', 'unknown')}: {e}"
                )
                logger.exception(error_msg)
                result.errors.append(error_msg)

        return result
