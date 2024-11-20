import asyncio
import os
import sys
from pathlib import Path
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv
import jwt

# pip install -e ./sdks/python
from databridge import DataBridge, DataBridgeError


def create_developer_test_uri():
    """Create a test URI for developer"""
    token = jwt.encode(
        {
            'type': 'developer',
            'exp': datetime.now(UTC) + timedelta(days=30)
        },
        "your-secret-key-for-signing-tokens",
        algorithm='HS256'
    )
    return f"databridge://673b64dcb6e40d739a9b6e2a:{token}@localhost:8000"


def create_user_test_uri():
    """Create a test URI for end user"""
    token = jwt.encode(
        {
            'type': 'user',
            'exp': datetime.now(UTC) + timedelta(days=30)
        },
        "your-secret-key-for-signing-tokens",
        algorithm='HS256'
    )
    return f"databridge://user_789:{token}@localhost:8000"


async def example_text():
    """Example of ingesting and querying text documents"""
    print("\n=== Text Document Example ===")
    load_dotenv()
    uri = os.getenv("DATABRIDGE_URI")
    if not uri:
        raise ValueError("Please set DATABRIDGE_URI environment variable")

    db = DataBridge(create_developer_test_uri())
    
    try:
        # Ingest a simple text document
        content = """
        Machine learning (ML) is a type of artificial intelligence (AI) that allows 
        software applications to become more accurate at predicting outcomes without 
        being explicitly programmed to do so. Machine learning algorithms use historical 
        data as input to predict new output values.
        """
        
        doc_id = await db.ingest_document(
            content=content,
            metadata={
                "title": "ML Introduction",
                "category": "tech",
                "tags": ["ML", "AI", "technology"]
            }
        )
        print(f"✓ Document ingested successfully (ID: {doc_id})")

        # Query the document
        results = await db.query(
            query="What is machine learning?",
            k=1  # Get top result,
        )
        
        print("\nQuery Results:")
        for result in results:
            print(f"Content: {result.content.strip()}")
            print(f"Score: {result.score:.2f}")
            print(f"Metadata: {result.metadata}")

    except DataBridgeError as e:
        print(f"× Error: {str(e)}")
    
    finally:
        await db.close()


async def example_pdf():
    """Example of ingesting and querying PDF documents"""
    print("\n=== PDF Document Example ===")

    # pdf_path = Path(__file__).parent / "sample.pdf"
    pdf_path = Path(__file__).parent / "trial.png"
    if not pdf_path.exists():
        print("× sample.pdf not found in examples directory")
        return

    db = DataBridge(create_developer_test_uri())
    
    try:
        # Read and ingest PDF
        with open(pdf_path, "rb") as f:
            pdf_content = f.read()

        doc_id = await db.ingest_document(
            content=pdf_content,
            # metadata={
            #     "title": "Sample Document",
            #     "source": "examples",
            #     "file_type": "pdf"
            # }
        )
        print(f"✓ PDF ingested successfully (ID: {doc_id})")

        # Query the PDF content
        results = await db.query(
            query="Brandsync repair!",
            k=2,  # Get top 2 results
            # filters={"file_type": "pdf"}  # Only search PDF documents
        )

        print("\nQuery Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {result.content[:200]}...")
            print(f"Score: {result.score:.2f}")
            print(f"Document ID: {result.doc_id}")

    except DataBridgeError as e:
        print(f"× Error: {str(e)}")

    finally:
        await db.close()


async def example_batch():
    """Example of batch operations"""
    print("\n=== Batch Operations Example ===")
    
    uri = os.getenv("DATABRIDGE_URI")
    if not uri:
        raise ValueError("Please set DATABRIDGE_URI environment variable")

    db = DataBridge(create_developer_test_uri())
    
    try:
        # Prepare multiple documents
        documents = [
            {
                "content": "Python is a programming language.",
                "metadata": {"category": "programming", "level": "basic"}
            },
            {
                "content": "JavaScript runs in the browser.",
                "metadata": {"category": "programming", "level": "basic"}
            },
            {
                "content": "Docker containers package applications.",
                "metadata": {"category": "devops", "level": "intermediate"}
            }
        ]

        # Ingest multiple documents
        doc_ids = []
        for doc in documents:
            doc_id = await db.ingest_document(
                content=doc["content"],
                metadata=doc["metadata"]
            )
            doc_ids.append(doc_id)
        print(f"✓ Ingested {len(doc_ids)} documents")

        # Query with filters
        results = await db.query(
            query="What is Python?",
            filters={"category": "programming"}
        )
        
        print("\nQuery Results (Programming category only):")
        for result in results:
            print(f"\nContent: {result.content}")
            print(f"Category: {result.metadata['category']}")
            print(f"Level: {result.metadata['level']}")

    except DataBridgeError as e:
        print(f"× Error: {str(e)}")
    
    finally:
        await db.close()


async def example_get_documents():
    """Example of getting documents"""
    print("\n=== Get Documents Example ===")

    db = DataBridge(create_developer_test_uri())
    documents = await db.get_documents()
    print(documents)

async def example_get_document_by_id(id: str):
    """Example of getting documents"""
    print("\n=== Get Documents Example ===")

    db = DataBridge(create_developer_test_uri())
    documents = await db.get_document_by_id(id)
    print(documents)


async def main():
    """Run all examples"""
    try:
        # await example_text()
        # await example_pdf()
        # await example_batch()
        # await example_get_documents()
        await example_get_document_by_id('673cb75886809b44b5c9d553');
    except Exception as e:
        print(f"× Main error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())