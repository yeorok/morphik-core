from typing import List, Dict, Any
from pymongo import MongoClient
from .base_vector_store import BaseVectorStore
from core.document import DocumentChunk


class MongoDBAtlasVectorStore(BaseVectorStore):
    def __init__(
        self,
        connection_string: str,
        database_name: str,
        collection_name: str = "kb_chunked_embeddings",
        index_name: str = "vector_index"
    ):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        self.index_name = index_name

        # Ensure vector search index exists
        # self._ensure_index()

    def _ensure_index(self):
        """Ensure the vector search index exists"""
        try:
            # Check if index exists
            indexes = self.collection.list_indexes()
            index_exists = any(index.get('name') == self.index_name for index in indexes)

            if not index_exists:
                # Create the vector search index if it doesn't exist
                self.collection.create_index(
                    [("embedding", "vectorSearch")],
                    name=self.index_name,
                    vectorSearchOptions={
                        "dimensions": 1536,  # For OpenAI embeddings
                        "similarity": "dotProduct"
                    }
                )
        except Exception as e:
            print(f"Warning: Could not create vector index: {str(e)}")

    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        try:
            documents = []
            for chunk in chunks:
                doc = {
                    "_id": chunk.id,
                    "text": chunk.content,
                    "embedding": chunk.embedding,
                    "doc_id": chunk.doc_id,
                    "owner_id": chunk.metadata.get("owner_id"),
                    "metadata": chunk.metadata
                }

                documents.append(doc)

            if documents:
                # Use ordered=False to continue even if some inserts fail
                result = self.collection.insert_many(documents, ordered=False)
                return len(result.inserted_ids) > 0
            return True

        except Exception as e:
            print(f"Error storing embeddings: {str(e)}")
            return False

    def query_similar(
        self,
        query_embedding: List[float],
        k: int,
        owner_id: str,
        filters: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """Find similar chunks using MongoDB Atlas Vector Search."""
        base_filter = {"owner_id": owner_id}
        if filters:
            filters.update(base_filter)

        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k,
                        "filter": filters if filters else base_filter
                    }
                },
                {
                    "$project": {
                        "score": {"$meta": "vectorSearchScore"},
                        "text": 1,
                        "embedding": 1,
                        "doc_id": 1,
                        "metadata": 1
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            chunks = []

            for result in results:
                chunk = DocumentChunk(
                    content=result["text"],
                    embedding=result["embedding"],
                    doc_id=result["doc_id"]
                )
                chunk.score = result.get("score", 0)
                # Add metadata back to chunk
                chunk.metadata = result.get("metadata", {})
                chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"Error querying similar documents: {str(e)}")
            return []
