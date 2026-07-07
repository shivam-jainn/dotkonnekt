import logging
import asyncio
from dataclasses import dataclass, field

from src.core.chunkers.semantic import SemanticChunker, TextChunker
from src.core.clause_graph import build_clause_graph
from src.core.document import Document
from src.core.embedders.embedder import EmbeddedChunk, Embedder
from src.core.enrichment.deterministic import DeterministicEnricher
from src.core.enrichment.semantic import SemanticEnricher
from src.core.parsers.base import ParsedDocument
from src.core.parsers.pdf import PDFParser
from src.storage import create_storage

logger = logging.getLogger(__name__)

PARSERS = {
    ".pdf": PDFParser(),
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNK_MIN_SIZE = 100
EMBED_BATCH_SIZE = 100


@dataclass
class PipelineResult:
    job_id: str
    total_chunks: int = 0
    embedded_chunks: list[EmbeddedChunk] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(
        self,
        enable_semantic_enrichment: bool = False,
    ) -> None:
        self.storage = create_storage()
        self.semantic_chunker = SemanticChunker(
            max_chunk_size=CHUNK_SIZE,
            min_chunk_size=CHUNK_MIN_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        # Legacy chunker adapter (wraps SemanticChunker with old interface)
        self.chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_size=CHUNK_MIN_SIZE,
        )
        self.embedder = Embedder(batch_size=EMBED_BATCH_SIZE)
        self.deterministic_enricher = DeterministicEnricher()
        self.semantic_enricher = SemanticEnricher(enabled=enable_semantic_enrichment)

    def _get_parser(self, filename: str):
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        parser = PARSERS.get(ext)
        if parser is None:
            raise ValueError(f"No parser found for extension '{ext}'")
        return parser

    def _parse(self, data: bytes, filename: str) -> ParsedDocument:
        parser = self._get_parser(filename)
        return parser.parse(data, filename)

    def _parse_to_document(self, data: bytes, filename: str) -> Document:
        """Parse directly into Document IR."""
        parser = self._get_parser(filename)
        if hasattr(parser, "parse_to_document"):
            return parser.parse_to_document(data, filename)
        # Fallback: use legacy parse and wrap in Document
        parsed = parser.parse(data, filename)
        return Document(
            filename=filename,
            metadata=parsed.metadata,
            raw_text=parsed.content,
        )

    def _chunk_document(self, document: Document) -> list:
        """Chunk a Document IR using the semantic chunker."""
        return self.semantic_chunker.chunk_document(document)

    def _chunk_legacy(self, document: ParsedDocument) -> list:
        """Legacy chunking interface for backward compat."""
        return self.chunker.chunk(document.content, document.metadata)

    def _enrich_deterministic(self, document: Document) -> None:
        """Run deterministic metadata enrichment."""
        self.deterministic_enricher.enrich_document(document)

    async def _enrich_semantic(self, document: Document) -> None:
        """Run optional LLM semantic enrichment."""
        await self.semantic_enricher.enrich_document(document)

    async def _embed(self, chunks) -> list[EmbeddedChunk]:
        return await self.embedder.embed_chunks(chunks)

    async def _embed_document_chunks(
        self, chunks: list
    ) -> tuple[list, list[list[float]]]:
        """Embed Document IR chunks."""
        return await self.embedder.embed_document_chunks(chunks)

    async def _process_file(self, file_info: dict) -> tuple[list[EmbeddedChunk], int, str | None, Document | None]:
        filename = file_info["filename"]
        storage_path = file_info["storage_path"]
        job_id = file_info.get("job_id", "")
        try:
            logger.info("Downloading %s from storage", filename)
            data = await self.storage.download_bytes(storage_path)

            logger.info("Parsing %s to Document IR", filename)
            document = await asyncio.to_thread(self._parse_to_document, data, filename)
            document.metadata["job_id"] = job_id

            logger.info("Chunking %s with semantic chunker", filename)
            doc_chunks = await asyncio.to_thread(self._chunk_document, document)

            logger.info("Enriching metadata for %s", filename)
            await asyncio.to_thread(self._enrich_deterministic, document)

            if self.semantic_enricher.enabled:
                logger.info("Running semantic enrichment for %s", filename)
                await self._enrich_semantic(document)

            logger.info("Building clause graph for %s", filename)
            await asyncio.to_thread(build_clause_graph, document)

            logger.info("Embedding %d chunks from %s", len(doc_chunks), filename)
            _, embeddings = await self._embed_document_chunks(doc_chunks)

            # Convert Document IR chunks to EmbeddedChunk for backward compat
            embedded = self._to_embedded_chunks(doc_chunks, embeddings, document)

            logger.info("Processed %s: %d chunks embedded", filename, len(embedded))
            return embedded, len(doc_chunks), None, document
        except Exception as e:
            error_msg = f"Error processing {filename}: {e}"
            logger.exception(error_msg)
            return [], 0, error_msg, None

    def _to_embedded_chunks(
        self, doc_chunks: list, embeddings: list[list[float]], document: Document
    ) -> list[EmbeddedChunk]:
        """Convert Document IR chunks + embeddings to EmbeddedChunk list."""
        embedded: list[EmbeddedChunk] = []
        for chunk, emb in zip(doc_chunks, embeddings):
            sm = None
            if chunk.derived_metadata:
                sm = {
                    "summary": chunk.derived_metadata.summary,
                    "keywords": chunk.derived_metadata.keywords,
                    "obligations": chunk.derived_metadata.obligations,
                    "risks": chunk.derived_metadata.risks,
                    "entities": chunk.derived_metadata.entities,
                    "topics": chunk.derived_metadata.topics,
                    "deadlines": chunk.derived_metadata.deadlines,
                    "rights": chunk.derived_metadata.rights,
                    "exclusions": chunk.derived_metadata.exclusions,
                    "definitions": chunk.derived_metadata.definitions,
                    "parties": chunk.derived_metadata.parties,
                    "jurisdictions": chunk.derived_metadata.jurisdictions,
                    "document_type": chunk.derived_metadata.document_type,
                }
            embedded.append(
                EmbeddedChunk(
                    content=chunk.content,
                    embedding=emb,
                    index=chunk.index,
                    metadata={
                        **chunk.metadata,
                        "job_id": document.metadata.get("job_id", ""),
                        "filename": document.filename,
                    },
                    id=chunk.id,
                    page=chunk.page,
                    section=chunk.section,
                    subsection=chunk.subsection,
                    clause=chunk.clause,
                    previous_chunk_id=chunk.previous_chunk_id,
                    next_chunk_id=chunk.next_chunk_id,
                    clause_id=chunk.clause_id,
                    semantic_metadata=sm,
                    content_type=chunk.content_type,
                )
            )
        return embedded

    async def run(self, job_id: str, files: list[dict]) -> PipelineResult:
        result = PipelineResult(job_id=job_id)

        # Inject job_id into file_info dicts
        enriched_files = [{**f, "job_id": job_id} for f in files]

        # Process all files concurrently
        tasks = [self._process_file(file_info) for file_info in enriched_files]
        res_list = await asyncio.gather(*tasks)

        for embedded, total_chunks, error_msg, document in res_list:
            if error_msg:
                result.errors.append(error_msg)
            result.total_chunks += total_chunks
            result.embedded_chunks.extend(embedded)
            if document:
                result.documents.append(document)

        return result
