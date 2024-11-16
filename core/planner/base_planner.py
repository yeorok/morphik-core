from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePlanner(ABC):
    @abstractmethod
    def plan_retrieval(self, query: str, **kwargs) -> Dict[str, Any]:
        """Create execution plan for retrieval"""
        pass
