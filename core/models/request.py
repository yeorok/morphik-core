from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .documents import QueryReturnType


class IngestRequest(BaseModel):
    content: str
    # TODO: We should infer this, not request it
    content_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    filename: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    return_type: QueryReturnType = QueryReturnType.CHUNKS
    filters: Optional[Dict[str, Any]] = None
    k: int = 4
    min_score: float = 0.0
