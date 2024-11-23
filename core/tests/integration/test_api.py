import base64
import pytest
from fastapi.testclient import TestClient
from core.api import app
import jwt
from datetime import datetime, UTC, timedelta


def create_auth_header(entity_type: str = "developer", permissions: list = None):
    """Create auth header with test token"""
    token = jwt.encode(
        {
            "type": entity_type,
            "entity_id": "test_user",
            "permissions": permissions or ["read", "write"],
            "exp": datetime.now(UTC) + timedelta(days=1)
        },
        # TODO: Use settings.JWT_SECRET_KEY
        "your-secret-key-for-signing-tokens"
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    return TestClient(app)


def test_ingest_document(client):
    """Test document ingestion endpoint"""
    headers = create_auth_header()

    # Test text document
    response = client.post(
        "/documents",
        json={
            "content": "Test content",
            "content_type": "text/plain",
            "metadata": {"test": True}
        },
        headers=headers
    )
    print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert data["external_id"]

    # Test binary document
    with open("test.pdf", "rb") as f:
        content = base64.b64encode(f.read()).decode()

    response = client.post(
        "/documents",
        json={
            "content": content,
            "content_type": "application/pdf",
            "filename": "test.pdf"
        },
        headers=headers
    )
    assert response.status_code == 200


def test_query_endpoints(client):
    """Test query endpoints"""
    headers = create_auth_header()

    # Test chunk query
    response = client.post(
        "/query",
        json={
            "query": "test",
            "return_type": "chunks",
            "k": 2
        },
        headers=headers
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) <= 2

    # Test document query
    response = client.post(
        "/query",
        json={
            "query": "test",
            "return_type": "documents"
        },
        headers=headers
    )
    assert response.status_code == 200


def test_document_management(client):
    """Test document management endpoints"""
    headers = create_auth_header()

    # List documents
    response = client.get("/documents", headers=headers)
    assert response.status_code == 200
    docs = response.json()

    if docs:
        # Get specific document
        doc_id = docs[0]["external_id"]
        response = client.get(f"/documents/{doc_id}", headers=headers)
        assert response.status_code == 200


def test_auth_errors(client):
    """Test authentication error cases"""
    # No auth header
    response = client.get("/documents")
    assert response.status_code == 401

    # Invalid token
    headers = {"Authorization": "Bearer invalid"}
    response = client.get("/documents", headers=headers)
    assert response.status_code == 401

    # Expired token
    token = jwt.encode(
        {
            "type": "developer",
            "entity_id": "test",
            "exp": datetime.now(UTC) - timedelta(days=1)
        },
        "test-secret"
    )
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/documents", headers=headers)
    assert response.status_code == 401


def main():
    """Run API endpoint tests"""
    client = TestClient(app)

    try:
        test_ingest_document(client)
        print("✓ Document ingestion tests passed")

        test_query_endpoints(client)
        print("✓ Query endpoint tests passed")

        test_document_management(client)
        print("✓ Document management tests passed")

        test_auth_errors(client)
        print("✓ Auth error tests passed")

    except Exception as e:
        print(f"× API test failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
