from abc import ABC, abstractmethod
from typing import List
from document import DocumentChunk


class BaseVectorStore(ABC):
    @abstractmethod
    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks and their embeddings"""
        pass

    @abstractmethod
    def query_similar(self, query_embedding: List[float], k: int, owner_id: str) -> List[DocumentChunk]:
        """Find similar chunks"""
        pass
