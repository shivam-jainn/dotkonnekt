from typing import Optional, List, Dict
from pydantic import BaseModel

class DerivedMetadataModel(BaseModel):
    summary: Optional[str] = None
    keywords: List[str] = []
    obligations: List[str] = []
    risks: List[str] = []
    entities: List[str] = []
    topics: List[str] = []
    deadlines: List[str] = []
    rights: List[str] = []
    exclusions: List[str] = []
    definitions: List[str] = []
    parties: List[str] = []
    jurisdictions: List[str] = []
    document_type: Optional[str] = None
    party_sentences: Dict[str, List[str]] = {}
    obligations_by_party: List[dict] = []
    risks_by_party: List[dict] = []

class AnalysisChunkModel(BaseModel):
    id: str
    content: str
    index: int
    page: int = 0
    section: Optional[str] = None
    subsection: Optional[str] = None
    clause: Optional[str] = None
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None
    clause_id: Optional[str] = None
    derived_metadata: Optional[DerivedMetadataModel] = None
    content_type: Optional[str] = None
    metadata: dict = {}

class LangGraphMessage(BaseModel):
    job_id: str
    collection: str
    chunks: List[AnalysisChunkModel]
