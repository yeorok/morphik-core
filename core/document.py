from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class AuthType(str, Enum):
    DEVELOPER = "developer"
    USER = "user"


@dataclass
class AuthContext:
    """Authentication context for request"""
    type: AuthType
    eu_id: Optional[str] = None
    dev_id: Optional[str] = None
    app_id: Optional[str] = None


class Permission(str, Enum):
    READ = "R"
    WRITE = "W"
    DELETE = "D"


class Source(str, Enum):
    APP = "app"  # TODO name of app
    SELF_UPLOADED = "self_uploaded"
    GOOGLE_DRIVE = "google_drive"
    NOTION = "notion"


@dataclass
class SystemMetadata:
    """Metadata controlled by our system"""
    dev_id: Optional[str] = None
    app_id: Optional[str] = None
    eu_id: Optional[str] = None
    doc_id: str = None
    s3_bucket: str = None
    s3_key: str = None


class DocumentChunk:
    def __init__(
        self,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],  # user has control over
        system_metadata: SystemMetadata,  # set by us
        source: Source,
        permissions: Dict[str, Set[Permission]] = None
    ):
        self.content = content
        self.embedding = embedding
        self.metadata = metadata
        self.system_metadata = system_metadata
        self.source = source
        self.permissions = permissions or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "system_metadata": vars(self.system_metadata),
            "source": self.source.value,
            "permissions": {
                app_id: [p.value for p in perms]
                for app_id, perms in self.permissions.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from stored dictionary"""
        # Convert stored permission values back to Permission enums
        permissions = {
            app_id: {Permission(p) for p in perms}
            for app_id, perms in data.get("permissions", {}).items()
        }

        return cls(
            content=data["content"],
            embedding=data["embedding"],
            metadata=data["metadata"],
            system_metadata=SystemMetadata(**data["system_metadata"]),
            source=Source(data["source"]),
            permissions=permissions
        )
