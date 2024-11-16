from typing import Dict, Any, List
from .databridge_uri import DataBridgeURI
from .document import Document, DocumentChunk
from .vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from .embedding_model.openai_embedding_model import OpenAIEmbeddingModel
from .parser.unstructured_parser import UnstructuredAPIParser
from .planner.simple_planner import SimpleRAGPlanner


class DataBridge:
    """
    DataBridge with owner authentication and authorization.
    Configured via URI containing owner credentials.
    """

    def __init__(self, uri: str):
        # Parse URI and initialize configuration
        self.config = DataBridgeURI(uri)

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all required components using the URI configuration"""
        self.embedding_model = OpenAIEmbeddingModel(
            api_key=self.config.openai_api_key,
            model_name=self.config.embedding_model
        )

        self.parser = UnstructuredAPIParser(
            api_key=self.config.unstructured_api_key,
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vector_store = MongoDBAtlasVectorStore(
            connection_string=self.config.mongo_uri,
            database_name=self.config.db_name,
            collection_name=self.config.collection_name
        )

        self.planner = SimpleRAGPlanner(default_k=4)

    async def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Document:
        """
        Ingest a document using the owner ID from the URI configuration.
        """
        # Add owner_id to metadata
        metadata['owner_id'] = self.config.owner_id

        # Create document
        doc = Document(content, metadata, self.config.owner_id)

        # Parse into chunks
        chunk_texts = self.parser.parse(content, metadata)

        # Create embeddings and chunks
        for chunk_text in chunk_texts:
            embedding = await self.embedding_model.embed(chunk_text)
            chunk = DocumentChunk(chunk_text, embedding, doc.id)
            chunk.metadata = {'owner_id': self.config.owner_id}
            doc.chunks.append(chunk)

        # Store in vector store
        success = self.vector_store.store_embeddings(doc.chunks)
        if not success:
            raise Exception("Failed to store embeddings")

        return doc

    async def query(
        self,
        query: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query the document store using the owner ID from the URI configuration.
        """
        # Create plan
        plan = self.planner.plan_retrieval(query, **kwargs)

        # Get query embedding
        query_embedding = await self.embedding_model.embed(query)

        # Execute plan
        chunks = self.vector_store.query_similar(
            query_embedding,
            k=plan["k"],
            owner_id=self.config.owner_id
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
