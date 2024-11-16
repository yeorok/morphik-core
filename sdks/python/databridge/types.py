from enum import Enum


class ContentType(str, Enum):
    """Supported content types"""
    TEXT = "text/plain"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"