"""
Configuration-related data models for the RAG English Study system.

This module contains data classes for managing system configuration,
setup status, and user preferences.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for Language Model providers.
    
    Attributes:
        provider: The LLM provider (openai, gemini, ollama)
        api_key: API key for the provider (if required)
        model_name: Specific model to use
        host: Host URL for local models (e.g., Ollama)
        temperature: Temperature setting for response generation
        max_tokens: Maximum tokens for responses
        additional_params: Additional provider-specific parameters
    """
    provider: str
    api_key: Optional[str] = None
    model_name: str = ""
    host: str = "localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1000
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate LLM configuration."""
        if self.provider not in ['openai', 'gemini', 'ollama']:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        if self.provider in ['openai', 'gemini'] and not self.api_key:
            raise ValueError(f"API key is required for {self.provider}")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        
        # Set default model names
        if not self.model_name:
            if self.provider == 'openai':
                self.model_name = 'gpt-3.5-turbo'
            elif self.provider == 'gemini':
                self.model_name = 'gemini-pro'
            elif self.provider == 'ollama':
                self.model_name = 'llama2'
    
    def is_configured(self) -> bool:
        """Check if the LLM is properly configured."""
        if self.provider in ['openai', 'gemini']:
            return bool(self.api_key and self.model_name)
        elif self.provider == 'ollama':
            return bool(self.model_name and self.host)
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert LLM config to dictionary."""
        return {
            'provider': self.provider,
            'api_key': self.api_key,
            'model_name': self.model_name,
            'host': self.host,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'additional_params': self.additional_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """Create LLM config from dictionary."""
        return cls(
            provider=data['provider'],
            api_key=data.get('api_key'),
            model_name=data.get('model_name', ''),
            host=data.get('host', 'localhost:11434'),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 1000),
            additional_params=data.get('additional_params', {})
        )


@dataclass
class DocumentConfig:
    """Configuration for document processing.
    
    Attributes:
        document_directory: Path to the directory containing documents
        supported_formats: List of supported file formats
        chunk_size: Size of text chunks for indexing
        chunk_overlap: Overlap between chunks
        exclude_patterns: Patterns to exclude when scanning files
        max_file_size: Maximum file size to process (in bytes)
    """
    document_directory: Optional[str] = None
    supported_formats: List[str] = field(default_factory=lambda: ['pdf', 'docx', 'txt', 'md'])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    exclude_patterns: List[str] = field(default_factory=lambda: ['*.tmp', '*.log', '.*'])
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    def __post_init__(self):
        """Validate document configuration."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if self.max_file_size <= 0:
            raise ValueError("Max file size must be positive")
    
    def is_configured(self) -> bool:
        """Check if document configuration is complete."""
        return bool(self.document_directory and Path(self.document_directory).exists())
    
    def get_document_directory_path(self) -> Optional[Path]:
        """Get the document directory as a Path object."""
        if self.document_directory:
            return Path(self.document_directory)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document config to dictionary."""
        return {
            'document_directory': self.document_directory,
            'supported_formats': self.supported_formats,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'exclude_patterns': self.exclude_patterns,
            'max_file_size': self.max_file_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentConfig':
        """Create document config from dictionary."""
        return cls(
            document_directory=data.get('document_directory'),
            supported_formats=data.get('supported_formats', ['pdf', 'docx', 'txt', 'md']),
            chunk_size=data.get('chunk_size', 1000),
            chunk_overlap=data.get('chunk_overlap', 200),
            exclude_patterns=data.get('exclude_patterns', ['*.tmp', '*.log', '.*']),
            max_file_size=data.get('max_file_size', 50 * 1024 * 1024)
        )


@dataclass
class UserConfig:
    """Configuration for user preferences.
    
    Attributes:
        native_language: User's native language for explanations
        target_language: Target language for learning (default: english)
        learning_level: User's current learning level
        preferred_topics: List of preferred learning topics
        feedback_level: Level of feedback detail (minimal, normal, detailed)
        session_timeout: Session timeout in minutes
    """
    native_language: str = "korean"
    target_language: str = "english"
    learning_level: str = "intermediate"
    preferred_topics: List[str] = field(default_factory=list)
    feedback_level: str = "normal"
    session_timeout: int = 30
    
    def __post_init__(self):
        """Validate user configuration."""
        if not self.native_language.strip():
            raise ValueError("Native language cannot be empty")
        
        if not self.target_language.strip():
            raise ValueError("Target language cannot be empty")
        
        if self.learning_level not in ['beginner', 'intermediate', 'advanced']:
            raise ValueError(f"Invalid learning level: {self.learning_level}")
        
        if self.feedback_level not in ['minimal', 'normal', 'detailed']:
            raise ValueError(f"Invalid feedback level: {self.feedback_level}")
        
        if self.session_timeout <= 0:
            raise ValueError("Session timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user config to dictionary."""
        return {
            'native_language': self.native_language,
            'target_language': self.target_language,
            'learning_level': self.learning_level,
            'preferred_topics': self.preferred_topics,
            'feedback_level': self.feedback_level,
            'session_timeout': self.session_timeout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserConfig':
        """Create user config from dictionary."""
        return cls(
            native_language=data.get('native_language', 'korean'),
            target_language=data.get('target_language', 'english'),
            learning_level=data.get('learning_level', 'intermediate'),
            preferred_topics=data.get('preferred_topics', []),
            feedback_level=data.get('feedback_level', 'normal'),
            session_timeout=data.get('session_timeout', 30)
        )


@dataclass
class Configuration:
    """Main configuration class containing all system settings.
    
    Attributes:
        llm: Language model configuration
        document: Document processing configuration
        user: User preferences configuration
        version: Configuration version for compatibility
        created_at: When the configuration was created
        updated_at: When the configuration was last updated
    """
    llm: Optional[LLMConfig] = None
    document: DocumentConfig = field(default_factory=DocumentConfig)
    user: UserConfig = field(default_factory=UserConfig)
    version: str = "1.0.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def is_setup_complete(self) -> bool:
        """Check if the initial setup is complete."""
        return (
            self.llm is not None and 
            self.llm.is_configured() and 
            self.document.is_configured()
        )
    
    def get_setup_status(self) -> 'SetupStatus':
        """Get detailed setup status."""
        return SetupStatus(
            llm_configured=self.llm is not None and self.llm.is_configured(),
            documents_configured=self.document.is_configured(),
            user_configured=True,  # User config has defaults
            overall_complete=self.is_setup_complete()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'llm': self.llm.to_dict() if self.llm else None,
            'document': self.document.to_dict(),
            'user': self.user.to_dict(),
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Configuration':
        """Create configuration from dictionary."""
        config = cls(
            document=DocumentConfig.from_dict(data.get('document', {})),
            user=UserConfig.from_dict(data.get('user', {})),
            version=data.get('version', '1.0.0'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
        
        if data.get('llm'):
            config.llm = LLMConfig.from_dict(data['llm'])
        
        return config


@dataclass
class SetupStatus:
    """Represents the current setup status of the system.
    
    Attributes:
        llm_configured: Whether LLM is configured
        documents_configured: Whether document directory is configured
        user_configured: Whether user preferences are configured
        overall_complete: Whether overall setup is complete
    """
    llm_configured: bool = False
    documents_configured: bool = False
    user_configured: bool = False
    overall_complete: bool = False
    
    def get_completion_percentage(self) -> float:
        """Get setup completion as a percentage."""
        completed = sum([
            self.llm_configured,
            self.documents_configured,
            self.user_configured
        ])
        return (completed / 3) * 100
    
    def get_missing_steps(self) -> List[str]:
        """Get list of missing setup steps."""
        missing = []
        if not self.llm_configured:
            missing.append("LLM configuration")
        if not self.documents_configured:
            missing.append("Document directory setup")
        if not self.user_configured:
            missing.append("User preferences")
        return missing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert setup status to dictionary."""
        return {
            'llm_configured': self.llm_configured,
            'documents_configured': self.documents_configured,
            'user_configured': self.user_configured,
            'overall_complete': self.overall_complete,
            'completion_percentage': self.get_completion_percentage(),
            'missing_steps': self.get_missing_steps()
        }