from fastapi import FastAPI, HTTPException, Depends, Header, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional, Annotated
from pydantic import BaseModel, Field
import jwt
import os
from datetime import datetime, UTC
import logging
from .vector_store.mongo_vector_store import MongoDBAtlasVectorStore
from .embedding_model.openai_embedding_model import OpenAIEmbeddingModel
from .parser.unstructured_parser import UnstructuredAPIParser
from .planner.simple_planner import SimpleRAGPlanner
from .document import Document, DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DataBridge API",
    description="REST API for DataBridge document ingestion and querying",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataBridgeException(HTTPException):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)


class AuthenticationError(DataBridgeException):
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(detail=detail, status_code=status.HTTP_401_UNAUTHORIZED)


class ServiceConfig:
    """Service-wide configuration and component management"""
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET_KEY environment variable not set")

        # Required environment variables
        required_vars = {
            "MONGODB_URI": "MongoDB connection string",
            "OPENAI_API_KEY": "OpenAI API key",
            "UNSTRUCTURED_API_KEY": "Unstructured API key"
        }

        missing = [f"{var} ({desc})" for var, desc in required_vars.items() if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Initialize core components
        self._init_components()

    def _init_components(self):
        """Initialize service components"""
        try:
            self.vector_store = MongoDBAtlasVectorStore(
                connection_string=os.getenv("MONGODB_URI"),
                database_name=os.getenv("DB_NAME", "databridge"),
                collection_name=os.getenv("COLLECTION_NAME", "embeddings")
            )

            self.embedding_model = OpenAIEmbeddingModel(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            )

            self.parser = UnstructuredAPIParser(
                api_key=os.getenv("UNSTRUCTURED_API_KEY"),
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
            )

            self.planner = SimpleRAGPlanner(
                default_k=int(os.getenv("DEFAULT_K", "4"))
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize components: {str(e)}")

    async def verify_token(self, token: str, owner_id: str) -> bool:
        """Verify JWT token and owner_id"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            if payload.get("owner_id") != owner_id:
                raise AuthenticationError("Owner ID mismatch")
            if datetime.fromtimestamp(payload["exp"], UTC) < datetime.now(UTC):
                raise AuthenticationError("Token has expired")
            return True
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            raise AuthenticationError(f"Authentication failed: {str(e)}")


# Initialize service
service = ServiceConfig()


# Request/Response Models
class IngestRequest(BaseModel):
    content: str = Field(..., description="Document content (text or base64)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string")
    k: Optional[int] = Field(default=4, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, 
                                            description="Optional metadata filters")


class IngestResponse(BaseModel):
    document_id: str = Field(..., description="Ingested document ID")
    message: str = Field(default="Document ingested successfully")


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Query results")
    total_results: int = Field(..., description="Total number of results")


# Authentication dependency
async def verify_auth(
    owner_id: Annotated[str, Header(alias="X-Owner-ID")],
    auth_token: Annotated[str, Header(alias="X-Auth-Token")]
) -> str:
    """Verify authentication headers"""
    await service.verify_token(auth_token, owner_id)
    return owner_id


# Error handler middleware
@app.middleware("http")
async def error_handler(request: Request, call_next):
    try:
        return await call_next(request)
    except DataBridgeException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.exception("Unexpected error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"}
        )


# API Routes
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    owner_id: str = Depends(verify_auth)
) -> IngestResponse:
    """
    Ingest a document into DataBridge.
    All configuration and credentials are handled server-side.
    """
    logger.info(f"Ingesting document for owner {owner_id}")

    # Add owner_id to metadata
    request.metadata['owner_id'] = owner_id

    # Create document
    doc = Document(request.content, request.metadata, owner_id)

    # Parse into chunks
    chunk_texts = service.parser.parse(request.content, request.metadata)
    # Create embeddings and chunks
    chunks = []
    for chunk_text in chunk_texts:
        embedding = await service.embedding_model.embed(chunk_text)
        chunk = DocumentChunk(chunk_text, embedding, doc.id)
        chunk.metadata = {
            'owner_id': owner_id,
            **request.metadata
        }
        chunks.append(chunk)

    # Store in vector store
    if not service.vector_store.store_embeddings(chunks):
        raise DataBridgeException(
            "Failed to store embeddings",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    return IngestResponse(document_id=doc.id)


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    owner_id: str = Depends(verify_auth)
) -> QueryResponse:
    """
    Query documents in DataBridge.
    All configuration and credentials are handled server-side.
    """
    logger.info(f"Processing query for owner {owner_id}")
    print("ADILOG ")
    # Create plan
    plan = service.planner.plan_retrieval(request.query, k=request.k)

    # Get query embedding
    query_embedding = await service.embedding_model.embed(request.query)

    # Query vector store
    chunks = service.vector_store.query_similar(
        query_embedding,
        k=plan["k"],
        owner_id=owner_id,
        filters=request.filters
    )

    # Format results
    results = [
        {
            "content": chunk.content,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.id,
            "score": getattr(chunk, "score", None),
            "metadata": {k:v for k,v in chunk.metadata.items() if k != 'owner_id'}
        }
        for chunk in chunks
    ]

    return QueryResponse(
        results=results,
        total_results=len(results)
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check service health"""
    try:
        # Verify MongoDB connection
        service.vector_store.collection.find_one({})
        return {"status": "healthy"}
    except Exception as e:
        raise DataBridgeException(
            f"Service unhealthy: {str(e)}", 
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Verify all connections on startup"""
    logger.info("Starting DataBridge service")
    await health_check()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DataBridge service")
