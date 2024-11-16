from typing import List, Union
import openai
from .base_embedding_model import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.client = openai.Client(api_key=api_key)
        self.model_name = model_name

    async def embed(self, text: Union[str, List[str]]) -> List[float]:
        if isinstance(text, str):
            text = [text]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )

        if len(text) == 1:
            return response.data[0].embedding

        return [item.embedding for item in response.data]
