from typing import Dict, Any, List
import uuid
from datetime import datetime


class Document:
    def __init__(self, content: str, metadata: Dict[str, Any], owner_id: str):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata
        self.owner_id = owner_id
        self.created_at = datetime.utcnow()
        self.chunks: List[DocumentChunk] = []


class DocumentChunk:
    def __init__(self, content: str, embedding: List[float], doc_id: str):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.doc_id = doc_id
