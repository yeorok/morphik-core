from typing import Dict, Any, List
from .base_parser import BaseParser
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import base64
import nltk
import logging

logger = logging.getLogger(__name__)


class UnstructuredAPIParser(BaseParser):
    def __init__(
        self,
        api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        api_url: str = "https://api.unstructuredapp.io"
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def parse(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Parse content using Unstructured API and split into chunks."""
        try:
            # Create temporary file for content
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(metadata)) as temp_file:
                if metadata.get("is_base64", False):
                    try:
                        decoded_content = base64.b64decode(content)
                    except Exception as e:
                        logger.error(f"Failed to decode base64 content: {str(e)}")
                        # If base64 decode fails, try treating as plain text
                        decoded_content = content.encode('utf-8')
                else:
                    decoded_content = content.encode('utf-8')
                
                temp_file.write(decoded_content)
                temp_file_path = temp_file.name

            try:
                # Use Unstructured API for parsing
                elements = partition(
                    filename=temp_file_path,
                    api_key=self.api_key,
                    api_url=self.api_url,
                    partition_via_api=True
                )

                # Combine elements and split into chunks
                full_text = "\n\n".join(str(element) for element in elements)
                chunks = self.text_splitter.split_text(full_text)

                if not chunks:
                    # If no chunks were created, use the full text as a single chunk
                    logger.warning("No chunks created, using full text as single chunk")
                    return [full_text] if full_text.strip() else []

                return chunks
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {str(e)}")

        except Exception as e:
            logger.error(f"Error parsing document: {str(e)}")
            raise Exception(f"Error parsing document: {str(e)}")

    def _get_file_extension(self, metadata: Dict[str, Any]) -> str:
        """Get appropriate file extension based on content type."""
        content_type_mapping = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'text/plain': '.txt',
            'text/html': '.html'
        }
        return content_type_mapping.get(metadata.get('content_type'), '.txt')
