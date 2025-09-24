"""
Data models for the RAG English Study system.

This package contains all data classes and models used throughout the system
for representing conversations, documents, configurations, and responses.
"""

# Conversation models
from .conversation import (
    Message,
    LearningPoint,
    ConversationSession
)

# Response models
from .response import (
    Correction,
    GrammarTip,
    VocabSuggestion,
    LearningFeedback,
    SearchResult,
    ConversationResponse
)

# Document models
from .document import (
    Document,
    IndexingResult,
    IndexingStatus,
    DocumentSummary
)

# Configuration models
from .config import (
    LLMConfig,
    DocumentConfig,
    UserConfig,
    Configuration,
    SetupStatus
)

__all__ = [
    # Conversation models
    'Message',
    'LearningPoint',
    'ConversationSession',
    
    # Response models
    'Correction',
    'GrammarTip',
    'VocabSuggestion',
    'LearningFeedback',
    'SearchResult',
    'ConversationResponse',
    
    # Document models
    'Document',
    'IndexingResult',
    'IndexingStatus',
    'DocumentSummary',
    
    # Configuration models
    'LLMConfig',
    'DocumentConfig',
    'UserConfig',
    'Configuration',
    'SetupStatus',
]