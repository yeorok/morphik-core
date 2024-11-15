from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Parse content into chunks"""
        pass
