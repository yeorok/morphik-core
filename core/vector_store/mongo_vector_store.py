from typing import List, Dict, Any
from fastapi.logger import logger
from pymongo import MongoClient

from .base_vector_store import BaseVectorStore
from core.document import AuthType, DocumentChunk, Source, SystemMetadata, AuthContext


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

    def store_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        try:
            documents = [chunk.to_dict() for chunk in chunks]

            if documents:
                # Use ordered=False to continue even if some inserts fail
                result = self.collection.insert_many(documents, ordered=False)
                return len(result.inserted_ids) > 0
            return True

        except Exception as e:
            print(f"Error storing embeddings: {str(e)}")
            return False

    # TODO a natural language interface for filtering would be great (langchanin self querying, etc).
    def _build_metadata_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build MongoDB filter for metadata fields.
        
        Converts user-provided filters into proper MongoDB metadata filters
        and validates the filter values.
        """
        if not filters:
            return {}
            
        metadata_filter = {}
        for key, value in filters.items():
            # Only allow filtering on metadata fields
            metadata_key = f"metadata.{key}"
            
            # Handle different types of filter values
            if isinstance(value, (str, int, float, bool)):
                # Simple equality match
                metadata_filter[metadata_key] = value
            elif isinstance(value, list):
                # Array contains or in operator
                metadata_filter[metadata_key] = {"$in": value}
            elif isinstance(value, dict):
                # Handle comparison operators
                valid_ops = {
                    "gt": "$gt",
                    "gte": "$gte", 
                    "lt": "$lt",
                    "lte": "$lte",
                    "ne": "$ne"
                }
                mongo_ops = {}
                for op, val in value.items():
                    if op not in valid_ops:
                        raise ValueError(f"Invalid operator: {op}")
                    mongo_ops[valid_ops[op]] = val
                metadata_filter[metadata_key] = mongo_ops
            else:
                raise ValueError(f"Unsupported filter value type for key {key}: {type(value)}")
                
        return metadata_filter

    def query_similar(
            self,
            query_embedding: List[float],
            k: int,
            auth: AuthContext,
            filters: Dict[str, Any] = None
        ) -> List[DocumentChunk]:
        """Find similar chunks using MongoDB Atlas Vector Search."""
        try:
            # Build access filter based on auth context
            if auth.type == AuthType.DEVELOPER:
                base_filter = {
                    "$or": [
                        {"system_metadata.dev_id": auth.dev_id},  # Dev's own docs
                        {"permissions": {"$in": [auth.app_id]}}  # Docs app has access to
                    ]
                }
            else:
                base_filter = {"system_metadata.eu_id": auth.eu_id}  # User's docs
            metadata_filter = self._build_metadata_filter(filters)
            # Combine with any additional filters
            filter_query = base_filter
            if metadata_filter:
                filter_query = {"$and": [base_filter, metadata_filter]}

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k,
                        "filter": filter_query
                    }
                },
                {
                    "$project": {
                        "score": {"$meta": "vectorSearchScore"},
                        "content": 1,
                        "metadata": 1,
                        "system_metadata": 1,
                        "source": 1,
                        "permissions": 1,
                        "_id": 0  # Don't need MongoDB's _id
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            chunks = []

            for result in results:
                chunk = DocumentChunk(
                    content=result["content"],
                    embedding=[],  # We don't need to send embeddings back
                    metadata=result["metadata"],
                    system_metadata=SystemMetadata(**result["system_metadata"]),
                    source=Source(result["source"]),
                    permissions=result.get("permissions", {})
                )
                # Add score from vector search
                chunk.score = result.get("score", 0)
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}")
            return []
