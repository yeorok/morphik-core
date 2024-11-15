from urllib.parse import urlparse, parse_qs
from typing import Optional, Dict, Any
import os
import jwt
from datetime import datetime, timedelta


class DataBridgeURI:
    """
    Handles parsing and validation of DataBridge URIs with owner authentication
    Format: databridge://<owner_id>:<auth_token>@host/path?params
    """
    def __init__(self, uri: str):
        self.uri = uri
        self._parse_uri()

    def _parse_uri(self):
        parsed = urlparse(self.uri)
        query_params = parse_qs(parsed.query)

        # Parse authentication info from netloc
        auth_parts = parsed.netloc.split('@')[0].split(':')
        if len(auth_parts) != 2:
            raise ValueError("URI must include owner_id and auth_token")

        self.owner_id = auth_parts[0]
        self.auth_token = auth_parts[1]

        # Validate and decode auth token
        try:
            self._validate_auth_token()
        except Exception as e:
            raise ValueError(f"Invalid auth token: {str(e)}")

        # Get the original MongoDB URI from environment - use it as is
        self.mongo_uri = os.getenv("MONGODB_URI")
        if not self.mongo_uri:
            raise ValueError("MONGODB_URI environment variable not set")

        # Get configuration from query parameters
        self.openai_api_key = query_params.get('openai_key', [os.getenv('OPENAI_API_KEY', '')])[0]
        self.unstructured_api_key = query_params.get('unstructured_key', [os.getenv('UNSTRUCTURED_API_KEY', '')])[0]
        self.db_name = query_params.get('db', ['brandsyncaidb'])[0]
        self.collection_name = query_params.get('collection', ['kb_chunked_embeddings'])[0]
        self.embedding_model = query_params.get('embedding_model', ['text-embedding-3-small'])[0]

        # Validate required fields
        if not all([self.mongo_uri, self.openai_api_key, self.unstructured_api_key]):
            raise ValueError("Missing required configuration in DataBridge URI")

    def _validate_auth_token(self):
        """Validate the auth token and extract any additional claims"""
        try:
            decoded = jwt.decode(self.auth_token, 'your-secret-key', algorithms=['HS256'])
            if decoded.get('owner_id') != self.owner_id:
                raise ValueError("Token owner_id mismatch")
            self.auth_claims = decoded
        except jwt.ExpiredSignatureError:
            raise ValueError("Auth token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid auth token")
