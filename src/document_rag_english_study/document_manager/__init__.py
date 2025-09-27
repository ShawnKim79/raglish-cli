"""
Document management module for the RAG English Study system.

This module provides document parsing and management functionality
for handling various file formats and indexing operations.
"""

from .parser import DocumentParser, DocumentParsingError
from .manager import DocumentManager

__all__ = [
    'DocumentParser',
    'DocumentParsingError',
    'DocumentManager'
]