"""
Document-related data models for the RAG English Study system.

This module contains data classes for managing documents, indexing results,
and document processing information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib


@dataclass
class Document:
    """Represents a document in the system.
    
    Attributes:
        id: Unique identifier for the document
        title: Title of the document
        file_path: Path to the source file
        content: Text content of the document
        file_type: Type of file (pdf, docx, txt, md)
        created_at: When the document was processed
        word_count: Number of words in the document
        language: Detected language of the document
        file_hash: Hash of the file for change detection
    """
    id: str
    title: str
    file_path: str
    content: str
    file_type: str
    created_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    language: str = "english"
    file_hash: str = ""
    
    def __post_init__(self):
        """Validate and process document data after initialization."""
        if not self.id.strip():
            raise ValueError("Document ID cannot be empty")
        if not self.title.strip():
            raise ValueError("Document title cannot be empty")
        if not self.file_path.strip():
            raise ValueError("File path cannot be empty")
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")
        if self.file_type not in ['pdf', 'docx', 'txt', 'md']:
            raise ValueError(f"Unsupported file type: {self.file_type}")
        
        # Calculate word count if not provided
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        
        # Generate file hash if not provided
        if not self.file_hash:
            self.file_hash = self._generate_file_hash()
    
    def _generate_file_hash(self) -> str:
        """Generate a hash of the file content for change detection."""
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()
    
    def get_file_size(self) -> int:
        """Get the size of the source file in bytes."""
        try:
            return Path(self.file_path).stat().st_size
        except (OSError, FileNotFoundError):
            return 0
    
    def get_summary(self, max_length: int = 200) -> str:
        """Get a summary of the document content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length].rsplit(' ', 1)[0] + "..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for serialization."""
        return {
            'id': self.id,
            'title': self.title,
            'file_path': self.file_path,
            'content': self.content,
            'file_type': self.file_type,
            'created_at': self.created_at.isoformat(),
            'word_count': self.word_count,
            'language': self.language,
            'file_hash': self.file_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create document from dictionary."""
        return cls(
            id=data['id'],
            title=data['title'],
            file_path=data['file_path'],
            content=data['content'],
            file_type=data['file_type'],
            created_at=datetime.fromisoformat(data['created_at']),
            word_count=data.get('word_count', 0),
            language=data.get('language', 'english'),
            file_hash=data.get('file_hash', '')
        )


@dataclass
class IndexingResult:
    """Represents the result of a document indexing operation.
    
    Attributes:
        success: Whether the indexing was successful
        documents_processed: Number of documents processed
        total_chunks: Total number of text chunks created
        processing_time: Time taken for processing in seconds
        errors: List of error messages encountered
        indexed_files: List of successfully indexed file paths
        failed_files: List of files that failed to index
    """
    success: bool
    documents_processed: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    indexed_files: List[str] = field(default_factory=list)
    failed_files: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error message to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_indexed_file(self, file_path: str, chunks: int = 0) -> None:
        """Add a successfully indexed file."""
        self.indexed_files.append(file_path)
        self.documents_processed += 1
        self.total_chunks += chunks
    
    def add_failed_file(self, file_path: str, error: str) -> None:
        """Add a file that failed to index."""
        self.failed_files.append(file_path)
        self.add_error(f"Failed to index {file_path}: {error}")
    
    def get_success_rate(self) -> float:
        """Get the success rate as a percentage."""
        total_files = len(self.indexed_files) + len(self.failed_files)
        if total_files == 0:
            return 0.0
        return (len(self.indexed_files) / total_files) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert indexing result to dictionary."""
        return {
            'success': self.success,
            'documents_processed': self.documents_processed,
            'total_chunks': self.total_chunks,
            'processing_time': self.processing_time,
            'errors': self.errors,
            'indexed_files': self.indexed_files,
            'failed_files': self.failed_files,
            'success_rate': self.get_success_rate()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexingResult':
        """Create indexing result from dictionary."""
        return cls(
            success=data['success'],
            documents_processed=data.get('documents_processed', 0),
            total_chunks=data.get('total_chunks', 0),
            processing_time=data.get('processing_time', 0.0),
            errors=data.get('errors', []),
            indexed_files=data.get('indexed_files', []),
            failed_files=data.get('failed_files', [])
        )


@dataclass
class IndexingStatus:
    """Represents the current status of document indexing.
    
    Attributes:
        is_indexing: Whether indexing is currently in progress
        total_documents: Total number of documents to index
        processed_documents: Number of documents already processed
        current_file: Currently processing file (if any)
        start_time: When indexing started
        estimated_completion: Estimated completion time
    """
    is_indexing: bool = False
    total_documents: int = 0
    processed_documents: int = 0
    current_file: Optional[str] = None
    start_time: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    def get_progress_percentage(self) -> float:
        """Get indexing progress as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds since indexing started."""
        if not self.start_time:
            return None
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert indexing status to dictionary."""
        return {
            'is_indexing': self.is_indexing,
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'current_file': self.current_file,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'progress_percentage': self.get_progress_percentage(),
            'elapsed_time': self.get_elapsed_time()
        }


@dataclass
class DocumentSummary:
    """Represents a summary of all indexed documents.
    
    Attributes:
        total_documents: Total number of indexed documents
        total_words: Total word count across all documents
        file_types: Dictionary of file types and their counts
        languages: Dictionary of languages and their counts
        last_indexed: When the last document was indexed
        storage_size: Total storage size in bytes
    """
    total_documents: int = 0
    total_words: int = 0
    file_types: Dict[str, int] = field(default_factory=dict)
    languages: Dict[str, int] = field(default_factory=dict)
    last_indexed: Optional[datetime] = None
    storage_size: int = 0
    
    def add_document(self, document: Document) -> None:
        """Add a document to the summary statistics."""
        self.total_documents += 1
        self.total_words += document.word_count
        
        # Update file type count
        self.file_types[document.file_type] = self.file_types.get(document.file_type, 0) + 1
        
        # Update language count
        self.languages[document.language] = self.languages.get(document.language, 0) + 1
        
        # Update last indexed time
        if not self.last_indexed or document.created_at > self.last_indexed:
            self.last_indexed = document.created_at
        
        # Update storage size
        self.storage_size += document.get_file_size()
    
    def get_average_words_per_document(self) -> float:
        """Get average word count per document."""
        if self.total_documents == 0:
            return 0.0
        return self.total_words / self.total_documents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document summary to dictionary."""
        return {
            'total_documents': self.total_documents,
            'total_words': self.total_words,
            'file_types': self.file_types,
            'languages': self.languages,
            'last_indexed': self.last_indexed.isoformat() if self.last_indexed else None,
            'storage_size': self.storage_size,
            'average_words_per_document': self.get_average_words_per_document()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentSummary':
        """Create document summary from dictionary."""
        return cls(
            total_documents=data.get('total_documents', 0),
            total_words=data.get('total_words', 0),
            file_types=data.get('file_types', {}),
            languages=data.get('languages', {}),
            last_indexed=datetime.fromisoformat(data['last_indexed']) if data.get('last_indexed') else None,
            storage_size=data.get('storage_size', 0)
        )