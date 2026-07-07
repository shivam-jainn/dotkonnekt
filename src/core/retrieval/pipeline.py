from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from src.configs import settings
from src.core.embedders.embedder import Embedder
from src.core.retrieval.intent import detect_intent, RetrievalIntent
from src.core.retrieval.context import build_context_prompt

class RetrievalPipeline:
    def __init__(self):
        self.qdrant = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=False,
            check_compatibility=False,
        )
        self.embedder = Embedder()

    async def execute(
        self,
        job_id: str,
        query: str,
        top_k: int = 5,
        collection: str | None = None,
    ) -> dict:
        collection = collection or settings.qdrant_collection
        
        # 1. Intent Detection
        intent = detect_intent(query)
        
        # Adjust top_k based on intent
        if intent == RetrievalIntent.SUMMARY:
            effective_top_k = max(top_k, 10)
        else:
            effective_top_k = top_k

        # Check if query references specific party role
        party_filter = None
        lower_query = query.lower()
        if "insurer" in lower_query:
            party_filter = "Insurer"
        elif "insured" in lower_query or "insuree" in lower_query:
            party_filter = "Insured"
        elif "landlord" in lower_query or "lessor" in lower_query:
            party_filter = "Landlord"
        elif "tenant" in lower_query or "lessee" in lower_query:
            party_filter = "Tenant"
        elif "employer" in lower_query:
            party_filter = "Employer"
        elif "employee" in lower_query:
            party_filter = "Employee"
        elif "disclosing" in lower_query:
            party_filter = "Disclosing Party"
        elif "receiving" in lower_query:
            party_filter = "Receiving Party"

        matching_chunk_ids = []
        if party_filter:
            try:
                from src.database import db
                from src.database.models import PartyModel, ObligationModel, RiskModel
                from sqlalchemy import select
                async with db.pool() as session:
                    stmt = select(PartyModel.id).where(
                        PartyModel.role.ilike(f"%{party_filter}%")
                    )
                    party_ids = (await session.execute(stmt)).scalars().all()
                    if party_ids:
                        ob_stmt = select(ObligationModel.chunk_id).where(ObligationModel.party_id.in_(party_ids))
                        rk_stmt = select(RiskModel.chunk_id).where(RiskModel.party_id.in_(party_ids))
                        chunk_ids = (await session.execute(ob_stmt)).scalars().all() + \
                                    (await session.execute(rk_stmt)).scalars().all()
                        matching_chunk_ids = list(set(chunk_ids))
            except Exception:
                logger.exception("Error querying relational graph filters from PostgreSQL")

        # 2. Vector Search
        try:
            query_vector = await self.embedder.embed_query(query)
        except Exception:
            return {"context_text": "", "chunks": [], "intent": intent}

        must_filters = [FieldCondition(key="job_id", match=MatchValue(value=job_id))]
        
        try:
            search_result = await self.qdrant.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=Filter(must=must_filters),
                limit=effective_top_k,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return {"context_text": "", "chunks": [], "intent": intent}

        chunks = []
        for hit in search_result.points:
            score = hit.score
            if hit.payload.get("id") in matching_chunk_ids:
                score += 2.0

            chunks.append({
                "id": hit.payload.get("id", ""),
                "content": hit.payload.get("content", ""),
                "index": hit.payload.get("index", 0),
                "score": score,
                "page": hit.payload.get("page"),
                "section": hit.payload.get("section"),
                "subsection": hit.payload.get("subsection"),
                "clause": hit.payload.get("clause"),
                "previous_chunk": hit.payload.get("previous_chunk"),
                "next_chunk": hit.payload.get("next_chunk"),
                "content_type": hit.payload.get("content_type"),
                "summary": hit.payload.get("summary"),
                "keywords": hit.payload.get("keywords", []),
                "obligations": hit.payload.get("obligations", []),
                "entities": hit.payload.get("entities", []),
                "risks": hit.payload.get("risks", []),
                "rights": hit.payload.get("rights", []),
                "exclusions": hit.payload.get("exclusions", []),
                "definitions": hit.payload.get("definitions", []),
                "document_type": hit.payload.get("document_type"),
                "party_sentences": hit.payload.get("party_sentences", {}),
                "obligations_by_party": hit.payload.get("obligations_by_party", []),
                "risks_by_party": hit.payload.get("risks_by_party", []),
            })

        chunks.sort(key=lambda x: x["score"], reverse=True)

        # 3. Neighbor Expansion & 4. Deduplication
        all_chunks = await self._expand_and_deduplicate(collection, chunks)

        # 5. Build prompt context using Locality-aware Prompting
        context_text = build_context_prompt(all_chunks)

        return {
            "context_text": context_text,
            "chunks": all_chunks,
            "intent": intent,
        }

    async def _expand_and_deduplicate(self, collection: str, chunks: list[dict]) -> list[dict]:
        neighbor_ids = set()
        for chunk in chunks:
            prev_id = chunk.get("previous_chunk")
            next_id = chunk.get("next_chunk")
            if prev_id:
                neighbor_ids.add(prev_id)
            if next_id:
                neighbor_ids.add(next_id)

        retrieved_ids = {c["id"] for c in chunks}
        ids_to_fetch = list(neighbor_ids - retrieved_ids)

        if not ids_to_fetch:
            # Just sort the retrieved chunks chronologically
            chunks.sort(key=lambda x: x["index"])
            return chunks

        try:
            neighbor_result = await self.qdrant.retrieve(
                collection_name=collection,
                ids=ids_to_fetch,
                with_payload=True,
                with_vectors=False,
            )
            
            neighbor_map = {}
            for point in neighbor_result:
                neighbor_map[point.id] = {
                    "id": point.payload.get("id", ""),
                    "content": point.payload.get("content", ""),
                    "index": point.payload.get("index", 0),
                    "score": 0.0,
                    "page": point.payload.get("page"),
                    "section": point.payload.get("section"),
                    "subsection": point.payload.get("subsection"),
                    "clause": point.payload.get("clause"),
                    "previous_chunk": point.payload.get("previous_chunk"),
                    "next_chunk": point.payload.get("next_chunk"),
                    "content_type": point.payload.get("content_type"),
                    "summary": point.payload.get("summary"),
                    "keywords": point.payload.get("keywords", []),
                    "obligations": point.payload.get("obligations", []),
                    "entities": point.payload.get("entities", []),
                    "risks": point.payload.get("risks", []),
                    "rights": point.payload.get("rights", []),
                    "exclusions": point.payload.get("exclusions", []),
                    "definitions": point.payload.get("definitions", []),
                    "document_type": point.payload.get("document_type"),
                    "party_sentences": point.payload.get("party_sentences", {}),
                    "obligations_by_party": point.payload.get("obligations_by_party", []),
                    "risks_by_party": point.payload.get("risks_by_party", []),
                }

            combined = chunks + list(neighbor_map.values())
            combined.sort(key=lambda x: x["index"])
            return combined
            
        except Exception:
            chunks.sort(key=lambda x: x["index"])
            return chunks
