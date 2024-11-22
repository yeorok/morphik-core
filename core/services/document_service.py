from collections import defaultdict
from typing import Dict, List, Union, Optional

import logging
from core.api import IngestRequest, QueryRequest
from core.database.base_database import BaseDatabase
from core.embedding_model.base_embedding_model import BaseEmbeddingModel
from core.parser.base_parser import BaseParser
from core.storage.base_storage import BaseStorage
from core.vector_store.base_vector_store import BaseVectorStore
from ..models.documents import (
    Document, DocumentChunk, ChunkResult, DocumentContent, DocumentResult,
    QueryReturnType
)
from ..models.auth import AuthContext


logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        database: BaseDatabase,
        vector_store: BaseVectorStore,
        storage: BaseStorage,
        parser: BaseParser,
        embedding_model: BaseEmbeddingModel
    ):
        self.db = database
        self.vector_store = vector_store
        self.storage = storage
        self.parser = parser
        self.embedding_model = embedding_model

    async def ingest_document(
        self,
        request: IngestRequest,
        auth: AuthContext
    ) -> Document:
        """Ingest a new document with chunks."""
        try:
            # 1. Create document record
            doc = Document(
                content_type=request.content_type,
                filename=request.filename,
                metadata=request.metadata,
                access_control={
                    "owner": {
                        "type": auth.entity_type,
                        "id": auth.entity_id
                    },
                    "readers": {auth.entity_id},
                    "writers": {auth.entity_id},
                    "admins": {auth.entity_id}
                }
            )

            # 2. Store file in storage if it's not text
            if request.content_type != "text/plain":
                storage_info = await self.storage.upload_from_base64(
                    request.content,
                    doc.external_id,
                    request.content_type
                )
                doc.storage_info = {
                    "bucket": storage_info[0],
                    "key": storage_info[1]
                }

            # 3. Parse content into chunks
            chunks = await self.parser.parse(request.content)
            
            # 4. Generate embeddings for chunks
            embeddings = await self.embedding_model.embed_for_ingestion(chunks)

            # 5. Create and store chunks with embeddings
            chunk_objects = []
            for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = DocumentChunk(
                    document_id=doc.external_id,
                    content=content,
                    embedding=embedding,
                    chunk_number=i,
                    metadata=doc.metadata  # Inherit document metadata
                )
                chunk_objects.append(chunk)

            # 6. Store chunks in vector store
            success = await self.vector_store.store_embeddings(chunk_objects)
            if not success:
                raise Exception("Failed to store chunk embeddings")

            # 7. Store document metadata
            if not await self.db.store_document(doc):
                raise Exception("Failed to store document metadata")

            return doc

        except Exception as e:
            # TODO: Clean up any stored data on failure
            raise Exception(f"Document ingestion failed: {str(e)}")

    async def query(
        self,
        request: QueryRequest,
        auth: AuthContext
    ) -> Union[List[ChunkResult], List[DocumentResult]]:
        """Query documents with specified return type."""
        try:
            # 1. Get embedding for query
            query_embedding = await self.embedding_model.embed_for_query(request.query)

            # 2. Find authorized documents
            doc_ids = await self.db.find_documents(auth, request.filters)
            if not doc_ids:
                return []

            # 3. Search chunks with vector similarity
            chunks = await self.vector_store.query_similar(
                query_embedding,
                k=request.k,
                auth=auth,
                filters={"document_id": {"$in": doc_ids}}
            )

            # 4. Return results in requested format
            if request.return_type == QueryReturnType.CHUNKS:
                return await self._create_chunk_results(auth, chunks)
            else:
                return await self._create_document_results(auth, chunks)

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise e

    async def _create_chunk_results(self, auth: AuthContext, chunks: List[DocumentChunk]) -> List[ChunkResult]:
        """Create ChunkResult objects with document metadata."""
        results = []
        for chunk in chunks:
            # Get document metadata
            doc = await self.db.get_document(chunk.document_id, auth)
            if not doc:
                continue

            # Generate download URL if needed
            download_url = None
            if doc.storage_info:
                download_url = await self.storage.get_download_url(
                    doc.storage_info["bucket"],
                    doc.storage_info["key"]
                )

            results.append(ChunkResult(
                content=chunk.content,
                score=chunk.score,
                document_id=chunk.document_id,
                chunk_number=chunk.chunk_number,
                metadata=doc.metadata,
                content_type=doc.content_type,
                filename=doc.filename,
                download_url=download_url
            ))

        return results

    async def _create_document_results(self, auth: AuthContext, chunks: List[DocumentChunk]) -> List[DocumentResult]:
        """Group chunks by document and create DocumentResult objects."""
        # Group chunks by document and get highest scoring chunk per doc
        doc_chunks: Dict[str, DocumentChunk] = {}
        for chunk in chunks:
            if chunk.document_id not in doc_chunks or chunk.score > doc_chunks[chunk.document_id].score:
                doc_chunks[chunk.document_id] = chunk

        results = []
        for doc_id, chunk in doc_chunks.items():
            # Get document metadata
            doc = await self.db.get_document(doc_id, auth)
            if not doc:
                continue

            # Create DocumentContent based on content type
            if doc.content_type == "text/plain":
                content = DocumentContent(
                    type="string",
                    value=chunk.content,
                    filename=None
                )
            else:
                # Generate download URL for file types
                download_url = await self.storage.get_download_url(
                    doc.storage_info["bucket"],
                    doc.storage_info["key"]
                )
                content = DocumentContent(
                    type="url",
                    value=download_url,
                    filename=doc.filename
                )

            results.append(DocumentResult(
                score=chunk.score,
                document_id=doc_id,
                metadata=doc.metadata,
                content=content
            ))

        return results
