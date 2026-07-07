import asyncio
import logging

from src.configs import settings
from src.core.embedders.embedder import EmbeddedChunk
from src.core.storer import VectorStorer
from src.models.storage import StorageMessage
from src.queue import queue

logger = logging.getLogger(__name__)


class StorageWorker:
    def __init__(self) -> None:
        self.storer = VectorStorer()
        self._running = False
        self._stop_event = asyncio.Event()

    async def _store_chunks(self, raw_message: bytes) -> None:
        try:
            msg = StorageMessage.model_validate_json(raw_message)

            chunks = [
                EmbeddedChunk(
                    content=c.content,
                    embedding=c.embedding,
                    index=c.index,
                    metadata=c.metadata,
                    id=c.id,
                    page=c.page,
                    section=c.section,
                    subsection=c.subsection,
                    clause=c.clause,
                    previous_chunk_id=c.previous_chunk,
                    next_chunk_id=c.next_chunk,
                    semantic_metadata={
                        "summary": c.summary,
                        "keywords": c.keywords,
                        "obligations": c.obligations,
                        "entities": c.entities,
                        "risks": c.risks,
                        "party_sentences": c.party_sentences,
                        "obligations_by_party": c.obligations_by_party,
                        "risks_by_party": c.risks_by_party,
                    }
                    if c.summary or c.keywords or c.obligations or c.entities or c.risks or c.party_sentences or c.obligations_by_party or c.risks_by_party
                    else None,
                )
                for c in msg.chunks
            ]

            await self.storer.store_batch(msg.collection, chunks)

            # Store the legal relationship graph in PostgreSQL
            await self._save_relational_graph(msg)

            logger.info(
                "Stored %d chunks from job %s to collection '%s' and updated relational graph",
                len(chunks),
                msg.job_id,
                msg.collection,
            )

        except Exception:
            logger.exception("Failed to store chunk batch — message will be requeued")
            raise

    async def _save_relational_graph(self, msg: StorageMessage) -> None:
        if not msg.chunks:
            return

        import uuid
        from src.database import db
        from src.database.models import DocumentModel, PartyModel, RelationModel, ObligationModel, RiskModel
        from src.core.storer import QDRANT_NS

        # Group chunks by filename
        chunks_by_file = {}
        for c in msg.chunks:
            fn = c.metadata.get("filename", "unknown_document")
            chunks_by_file.setdefault(fn, []).append(c)

        async with db.pool() as session:
            for filename, chunks in chunks_by_file.items():
                # 1. Determine doc ID and type
                doc_type = chunks[0].document_type or "generic"
                doc_id = str(uuid.uuid5(QDRANT_NS, f"{msg.job_id}_{filename}"))

                # Create document if not exists
                doc_model = await session.get(DocumentModel, doc_id)
                if not doc_model:
                    doc_model = DocumentModel(
                        id=doc_id,
                        job_id=msg.job_id,
                        filename=filename,
                        document_type=doc_type,
                    )
                    session.add(doc_model)

                # 2. Extract unique parties & roles
                unique_parties = set()
                for c in chunks:
                    if c.party_sentences:
                        for party_name in c.party_sentences.keys():
                            unique_parties.add(party_name)
                    if c.obligations_by_party:
                        for ob in c.obligations_by_party:
                            unique_parties.add(ob.get("party_role"))
                    if c.risks_by_party:
                        for rk in c.risks_by_party:
                            unique_parties.add(rk.get("party_role"))

                unique_parties.discard("General")
                unique_parties.discard("")

                party_id_map = {}
                for party in unique_parties:
                    party_id = str(uuid.uuid5(QDRANT_NS, f"{doc_id}_{party}"))
                    party_id_map[party] = party_id
                    
                    party_model = await session.get(PartyModel, party_id)
                    if not party_model:
                        party_model = PartyModel(
                            id=party_id,
                            document_id=doc_id,
                            name=party,
                            role=party,
                        )
                        session.add(party_model)

                # 3. Add default relations based on doc_type
                relations = []
                if doc_type == "nda":
                    if "Disclosing Party" in party_id_map and "Receiving Party" in party_id_map:
                        relations.append(("Disclosing Party", "Receiving Party", "discloses to"))
                elif doc_type == "lease":
                    if "Landlord" in party_id_map and "Tenant" in party_id_map:
                        relations.append(("Landlord", "Tenant", "leases to"))
                elif doc_type == "policy":
                    if "Insurer" in party_id_map and "Insured" in party_id_map:
                        relations.append(("Insurer", "Insured", "insures"))
                    elif "Insurer" in party_id_map and "Insuree" in party_id_map:
                        relations.append(("Insurer", "Insuree", "insures"))

                for source, target, rel_type in relations:
                    rel_id = str(uuid.uuid5(QDRANT_NS, f"{doc_id}_{source}_{target}_{rel_type}"))
                    rel_model = await session.get(RelationModel, rel_id)
                    if not rel_model:
                        rel_model = RelationModel(
                            id=rel_id,
                            document_id=doc_id,
                            source_party_id=party_id_map[source],
                            target_party_id=party_id_map[target],
                            relation_type=rel_type,
                        )
                        session.add(rel_model)

                # 4. Insert Obligations & Risks
                for c in chunks:
                    chunk_point_id = c.id if c.id else str(
                        uuid.uuid5(
                            QDRANT_NS,
                            f"{c.metadata.get('job_id', 'doc')}_{c.metadata.get('filename', 'doc')}_{c.index}",
                        )
                    )

                    if c.obligations_by_party:
                        for ob in c.obligations_by_party:
                            role = ob.get("party_role")
                            if role in party_id_map:
                                text = ob.get("text")
                                ob_id = str(uuid.uuid5(QDRANT_NS, f"{chunk_point_id}_{role}_{text[:30]}"))
                                ob_model = await session.get(ObligationModel, ob_id)
                                if not ob_model:
                                    ob_model = ObligationModel(
                                        id=ob_id,
                                        party_id=party_id_map[role],
                                        chunk_id=chunk_point_id,
                                        text=text,
                                    )
                                    session.add(ob_model)

                    if c.risks_by_party:
                        for rk in c.risks_by_party:
                            role = rk.get("party_role")
                            if role in party_id_map:
                                desc = rk.get("description")
                                rk_id = str(uuid.uuid5(QDRANT_NS, f"{chunk_point_id}_{role}_{desc[:30]}"))
                                rk_model = await session.get(RiskModel, rk_id)
                                if not rk_model:
                                    rk_model = RiskModel(
                                        id=rk_id,
                                        party_id=party_id_map[role],
                                        chunk_id=chunk_point_id,
                                        description=desc,
                                        risk_level="Medium",
                                    )
                                    session.add(rk_model)

            await session.commit()

    async def start(self) -> None:
        if self._running:
            logger.warning("Storage worker is already running")
            return

        self._running = True
        logger.info(
            "Starting storage worker, consuming from queue: %s",
            settings.storage_queue,
        )

        await self.storer.initialize()

        await queue.consume(settings.storage_queue, self._store_chunks)

        logger.info("Storage worker is now listening for messages")
        self._stop_event.clear()
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        await self.storer.close()
        logger.info("Storage worker stopped")
