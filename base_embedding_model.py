from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, text: Union[str, List[str]]) -> List[float]:
        """Generate embeddings for input text"""
        pass
