from typing import Dict, Any
from base_planner import BasePlanner


class SimpleRAGPlanner(BasePlanner):
    def __init__(self, default_k: int = 3):
        self.default_k = default_k

    def plan_retrieval(self, query: str, **kwargs) -> Dict[str, Any]:
        """Create a simple retrieval plan."""
        return {
            "strategy": "simple_retrieval",
            "k": kwargs.get("k", self.default_k),
            "query": query,
            "filters": kwargs.get("filters", {}),
            "min_score": kwargs.get("min_score", 0.0)
        }
