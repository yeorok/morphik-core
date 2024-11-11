from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

# Base Classes and Interfaces

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

class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Parse content into chunks"""
        pass


class BasePlanner(ABC):
    @abstractmethod
    def plan_retrieval(self, query: str, **kwargs) -> Dict[str, Any]:
        """Create execution plan for retrieval"""
        pass


class BaseVectorStore(ABC):
    @abstractmethod
    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks and their embeddings"""
        pass

    @abstractmethod
    def query_similar(self, query_embedding: List[float], k: int, owner_id: str) -> List[DocumentChunk]:
        """Find similar chunks"""
        pass

# Concrete Implementations

class SimpleParser(BaseParser):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        # Simple implementation - split by chunk_size
        chunks = []
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            chunk = content[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks

class SimpleRAGPlanner(BasePlanner):
    def __init__(self, k: int = 3):
        self.k = k

    def plan_retrieval(self, query: str, **kwargs) -> Dict[str, Any]:
        return {
            "strategy": "simple_rag",
            "k": kwargs.get("k", self.k),
            "query": query
        }

# Main DataBridge Class

class DataBridge:
    def __init__(
        self,
        parser: BaseParser,
        planner: BasePlanner,
        vector_store: BaseVectorStore,
        embedding_model: Any  # This would be your chosen embedding model
    ):
        self.parser = parser
        self.planner = planner
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        owner_id: str
    ) -> Document:
        # Create document
        doc = Document(content, metadata, owner_id)

        # Parse into chunks
        chunk_texts = self.parser.parse(content, metadata)

        # Create embeddings and chunks
        for chunk_text in chunk_texts:
            embedding = await self.embedding_model.embed(chunk_text)
            chunk = DocumentChunk(chunk_text, embedding, doc.id)
            doc.chunks.append(chunk)

        # Store in vector store
        success = self.vector_store.store_embeddings(doc.chunks)
        if not success:
            raise Exception("Failed to store embeddings")

        return doc

    async def query(
        self,
        query: str,
        owner_id: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        # Create plan
        plan = self.planner.plan_retrieval(query, **kwargs)

        # Get query embedding
        query_embedding = await self.embedding_model.embed(query)

        # Execute plan
        chunks = self.vector_store.query_similar(
            query_embedding,
            k=plan["k"],
            owner_id=owner_id
        )

        # Format results
        results = []
        for chunk in chunks:
            results.append({
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.id,
                "score": chunk.score if hasattr(chunk, "score") else None
            })

        return results

# Example usage
"""
# Initialize components
parser = SimpleParser()
planner = SimpleRAGPlanner()
vector_store = YourVectorStore()  # Implement with chosen backend
embedding_model = YourEmbeddingModel()  # Implement with chosen model

# Create DataBridge instance
db = DataBridge(parser, planner, vector_store, embedding_model)

# Ingest a document
doc = await db.ingest_document(
    content="Your document content here",
    metadata={"source": "pdf", "title": "Example Doc"},
    owner_id="user123"
)

# Query the system
results = await db.query(
    query="Your query here",
    owner_id="user123",
    k=5  # optional override
)
"""