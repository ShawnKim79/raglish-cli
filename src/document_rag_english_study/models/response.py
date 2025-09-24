"""
Response-related data models for the RAG English Study system.

This module contains data classes for managing conversation responses,
learning feedback, and related information.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Correction:
    """Represents a grammar or language correction.
    
    Attributes:
        original_text: The original incorrect text
        corrected_text: The corrected version
        explanation: Explanation of the correction
        error_type: Type of error (grammar, vocabulary, etc.)
    """
    original_text: str
    corrected_text: str
    explanation: str
    error_type: str = "grammar"
    
    def __post_init__(self):
        """Validate correction data."""
        if not self.original_text.strip():
            raise ValueError("Original text cannot be empty")
        if not self.corrected_text.strip():
            raise ValueError("Corrected text cannot be empty")
        if not self.explanation.strip():
            raise ValueError("Explanation cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert correction to dictionary."""
        return {
            'original_text': self.original_text,
            'corrected_text': self.corrected_text,
            'explanation': self.explanation,
            'error_type': self.error_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Correction':
        """Create correction from dictionary."""
        return cls(
            original_text=data['original_text'],
            corrected_text=data['corrected_text'],
            explanation=data['explanation'],
            error_type=data.get('error_type', 'grammar')
        )


@dataclass
class GrammarTip:
    """Represents a grammar tip or explanation.
    
    Attributes:
        rule: The grammar rule being explained
        explanation: Detailed explanation
        examples: List of example sentences
        difficulty_level: Difficulty level of the grammar point
    """
    rule: str
    explanation: str
    examples: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    
    def __post_init__(self):
        """Validate grammar tip data."""
        if not self.rule.strip():
            raise ValueError("Grammar rule cannot be empty")
        if not self.explanation.strip():
            raise ValueError("Grammar explanation cannot be empty")
        if self.difficulty_level not in ['beginner', 'intermediate', 'advanced']:
            raise ValueError(f"Invalid difficulty level: {self.difficulty_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert grammar tip to dictionary."""
        return {
            'rule': self.rule,
            'explanation': self.explanation,
            'examples': self.examples,
            'difficulty_level': self.difficulty_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrammarTip':
        """Create grammar tip from dictionary."""
        return cls(
            rule=data['rule'],
            explanation=data['explanation'],
            examples=data.get('examples', []),
            difficulty_level=data.get('difficulty_level', 'intermediate')
        )


@dataclass
class VocabSuggestion:
    """Represents a vocabulary suggestion.
    
    Attributes:
        word: The vocabulary word
        definition: Definition of the word
        usage_example: Example of how to use the word
        synonyms: List of synonyms
        difficulty_level: Difficulty level of the vocabulary
    """
    word: str
    definition: str
    usage_example: str = ""
    synonyms: List[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    
    def __post_init__(self):
        """Validate vocabulary suggestion data."""
        if not self.word.strip():
            raise ValueError("Vocabulary word cannot be empty")
        if not self.definition.strip():
            raise ValueError("Definition cannot be empty")
        if self.difficulty_level not in ['beginner', 'intermediate', 'advanced']:
            raise ValueError(f"Invalid difficulty level: {self.difficulty_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vocabulary suggestion to dictionary."""
        return {
            'word': self.word,
            'definition': self.definition,
            'usage_example': self.usage_example,
            'synonyms': self.synonyms,
            'difficulty_level': self.difficulty_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VocabSuggestion':
        """Create vocabulary suggestion from dictionary."""
        return cls(
            word=data['word'],
            definition=data['definition'],
            usage_example=data.get('usage_example', ''),
            synonyms=data.get('synonyms', []),
            difficulty_level=data.get('difficulty_level', 'intermediate')
        )


@dataclass
class LearningFeedback:
    """Represents learning feedback for a user's input.
    
    Attributes:
        corrections: List of corrections for errors
        grammar_tips: List of grammar tips
        vocabulary_suggestions: List of vocabulary suggestions
        encouragement: Encouraging message for the user
    """
    corrections: List[Correction] = field(default_factory=list)
    grammar_tips: List[GrammarTip] = field(default_factory=list)
    vocabulary_suggestions: List[VocabSuggestion] = field(default_factory=list)
    encouragement: str = ""
    
    def has_feedback(self) -> bool:
        """Check if there's any feedback to provide."""
        return bool(self.corrections or self.grammar_tips or self.vocabulary_suggestions)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert learning feedback to dictionary."""
        return {
            'corrections': [c.to_dict() for c in self.corrections],
            'grammar_tips': [gt.to_dict() for gt in self.grammar_tips],
            'vocabulary_suggestions': [vs.to_dict() for vs in self.vocabulary_suggestions],
            'encouragement': self.encouragement
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningFeedback':
        """Create learning feedback from dictionary."""
        feedback = cls(encouragement=data.get('encouragement', ''))
        
        # Reconstruct corrections
        for corr_data in data.get('corrections', []):
            feedback.corrections.append(Correction.from_dict(corr_data))
        
        # Reconstruct grammar tips
        for tip_data in data.get('grammar_tips', []):
            feedback.grammar_tips.append(GrammarTip.from_dict(tip_data))
        
        # Reconstruct vocabulary suggestions
        for vocab_data in data.get('vocabulary_suggestions', []):
            feedback.vocabulary_suggestions.append(VocabSuggestion.from_dict(vocab_data))
        
        return feedback


@dataclass
class SearchResult:
    """Represents a search result from the RAG system.
    
    Attributes:
        content: The content text that was found
        source_file: Path to the source file
        relevance_score: Relevance score (0.0 to 1.0)
        metadata: Additional metadata about the result
    """
    content: str
    source_file: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate search result data."""
        if not self.content.strip():
            raise ValueError("Search result content cannot be empty")
        if not self.source_file.strip():
            raise ValueError("Source file cannot be empty")
        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError("Relevance score must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            'content': self.content,
            'source_file': self.source_file,
            'relevance_score': self.relevance_score,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create search result from dictionary."""
        return cls(
            content=data['content'],
            source_file=data['source_file'],
            relevance_score=data['relevance_score'],
            metadata=data.get('metadata', {})
        )


@dataclass
class ConversationResponse:
    """Represents a response in a conversation.
    
    Attributes:
        response_text: The main response text
        learning_feedback: Optional learning feedback
        suggested_topics: List of suggested follow-up topics
        context_sources: List of RAG search results used for context
    """
    response_text: str
    learning_feedback: Optional[LearningFeedback] = None
    suggested_topics: List[str] = field(default_factory=list)
    context_sources: List[SearchResult] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate conversation response data."""
        if not self.response_text.strip():
            raise ValueError("Response text cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation response to dictionary."""
        return {
            'response_text': self.response_text,
            'learning_feedback': self.learning_feedback.to_dict() if self.learning_feedback else None,
            'suggested_topics': self.suggested_topics,
            'context_sources': [cs.to_dict() for cs in self.context_sources]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationResponse':
        """Create conversation response from dictionary."""
        response = cls(
            response_text=data['response_text'],
            suggested_topics=data.get('suggested_topics', [])
        )
        
        # Reconstruct learning feedback
        if data.get('learning_feedback'):
            response.learning_feedback = LearningFeedback.from_dict(data['learning_feedback'])
        
        # Reconstruct context sources
        for source_data in data.get('context_sources', []):
            response.context_sources.append(SearchResult.from_dict(source_data))
        
        return response