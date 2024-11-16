import sys; sys.path.append('.')

from datetime import datetime, timedelta, UTC
import base64
from core.databridge import DataBridge
import jwt
import os
from dotenv import load_dotenv


def create_databridge_uri() -> str:
    """Create DataBridge URI from environment variables"""
    load_dotenv()

    # Get credentials from environment
    mongo_uri = os.getenv("MONGODB_URI")
    openai_key = os.getenv("OPENAI_API_KEY")
    unstructured_key = os.getenv("UNSTRUCTURED_API_KEY")
    owner_id = os.getenv("DATABRIDGE_OWNER", "admin")

    # Validate required credentials
    if not all([mongo_uri, openai_key, unstructured_key]):
        raise ValueError("Missing required environment variables")

    # Generate auth token
    auth_token = jwt.encode(
        {
            'owner_id': owner_id,
            'exp': datetime.now(UTC) + timedelta(days=30)
        },
        'your-secret-key',  # In production, use proper secret
        algorithm='HS256'
    )

    # For DataBridge URI, use any host identifier (it won't affect MongoDB connection)
    uri = (
        f"databridge://{owner_id}:{auth_token}@databridge.local"
        f"?openai_key={openai_key}"
        f"&unstructured_key={unstructured_key}"
        f"&db=brandsyncaidb"
        f"&collection=kb_chunked_embeddings"
    )

    return uri


async def main():
    # Initialize DataBridge
    bridge = DataBridge(create_databridge_uri())

    # Example: Ingest a PDF document
    with open("examples/sample.pdf", "rb") as f:
        pdf_content = base64.b64encode(f.read()).decode()

    await bridge.ingest_document(
        content=pdf_content,
        metadata={
            "content_type": "application/pdf",
            "is_base64": True,
            "title": "Sample PDF"
        }
    )

    # Example: Query documents
    results = await bridge.query(
        query="What is machine learning?",
        k=4
    )

    for result in results:
        print(f"Content: {result['content'][:200]}...")
        print(f"Score: {result['score']}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
