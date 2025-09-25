"""
LLM 관련 데이터 모델.

이 모듈은 언어 모델과의 상호작용에 필요한 데이터 구조들을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ErrorType(Enum):
    """문법 오류 유형을 정의하는 열거형."""
    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    SPELLING = "spelling"
    PUNCTUATION = "punctuation"
    SYNTAX = "syntax"


@dataclass
class GrammarError:
    """문법 오류를 나타내는 데이터 클래스.
    
    Attributes:
        text: 오류가 있는 텍스트
        error_type: 오류 유형
        position: 텍스트 내에서의 위치 (시작, 끝)
        suggestion: 수정 제안
        explanation: 오류에 대한 설명
    """
    text: str
    error_type: ErrorType
    position: tuple[int, int]
    suggestion: str
    explanation: str
    
    def __post_init__(self):
        """데이터 검증."""
        if not self.text.strip():
            raise ValueError("Error text cannot be empty")
        if not self.suggestion.strip():
            raise ValueError("Suggestion cannot be empty")
        if not self.explanation.strip():
            raise ValueError("Explanation cannot be empty")
        if self.position[0] < 0 or self.position[1] < self.position[0]:
            raise ValueError("Invalid position range")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'text': self.text,
            'error_type': self.error_type.value,
            'position': self.position,
            'suggestion': self.suggestion,
            'explanation': self.explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrammarError':
        """딕셔너리에서 생성."""
        return cls(
            text=data['text'],
            error_type=ErrorType(data['error_type']),
            position=tuple(data['position']),
            suggestion=data['suggestion'],
            explanation=data['explanation']
        )


@dataclass
class ImprovementSuggestion:
    """개선 제안을 나타내는 데이터 클래스.
    
    Attributes:
        category: 개선 카테고리 (vocabulary, grammar, style 등)
        original: 원본 텍스트
        improved: 개선된 텍스트
        reason: 개선 이유
        confidence: 제안의 신뢰도 (0.0 ~ 1.0)
    """
    category: str
    original: str
    improved: str
    reason: str
    confidence: float = 0.8
    
    def __post_init__(self):
        """데이터 검증."""
        if not self.category.strip():
            raise ValueError("Category cannot be empty")
        if not self.original.strip():
            raise ValueError("Original text cannot be empty")
        if not self.improved.strip():
            raise ValueError("Improved text cannot be empty")
        if not self.reason.strip():
            raise ValueError("Reason cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'category': self.category,
            'original': self.original,
            'improved': self.improved,
            'reason': self.reason,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImprovementSuggestion':
        """딕셔너리에서 생성."""
        return cls(
            category=data['category'],
            original=data['original'],
            improved=data['improved'],
            reason=data['reason'],
            confidence=data.get('confidence', 0.8)
        )


@dataclass
class EnglishAnalysis:
    """영어 텍스트 분석 결과를 나타내는 데이터 클래스.
    
    Attributes:
        grammar_errors: 발견된 문법 오류 목록
        vocabulary_level: 어휘 수준 평가
        fluency_score: 유창성 점수 (0.0 ~ 1.0)
        suggestions: 개선 제안 목록
        complexity_score: 문장 복잡도 점수 (0.0 ~ 1.0)
    """
    grammar_errors: List[GrammarError] = field(default_factory=list)
    vocabulary_level: str = "intermediate"
    fluency_score: float = 0.7
    suggestions: List[ImprovementSuggestion] = field(default_factory=list)
    complexity_score: float = 0.5
    
    def __post_init__(self):
        """데이터 검증."""
        if self.vocabulary_level not in ['beginner', 'intermediate', 'advanced']:
            raise ValueError(f"Invalid vocabulary level: {self.vocabulary_level}")
        if not 0.0 <= self.fluency_score <= 1.0:
            raise ValueError("Fluency score must be between 0.0 and 1.0")
        if not 0.0 <= self.complexity_score <= 1.0:
            raise ValueError("Complexity score must be between 0.0 and 1.0")
    
    def has_errors(self) -> bool:
        """오류가 있는지 확인."""
        return len(self.grammar_errors) > 0
    
    def has_suggestions(self) -> bool:
        """개선 제안이 있는지 확인."""
        return len(self.suggestions) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'grammar_errors': [error.to_dict() for error in self.grammar_errors],
            'vocabulary_level': self.vocabulary_level,
            'fluency_score': self.fluency_score,
            'suggestions': [suggestion.to_dict() for suggestion in self.suggestions],
            'complexity_score': self.complexity_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnglishAnalysis':
        """딕셔너리에서 생성."""
        analysis = cls(
            vocabulary_level=data.get('vocabulary_level', 'intermediate'),
            fluency_score=data.get('fluency_score', 0.7),
            complexity_score=data.get('complexity_score', 0.5)
        )
        
        # 문법 오류 복원
        for error_data in data.get('grammar_errors', []):
            analysis.grammar_errors.append(GrammarError.from_dict(error_data))
        
        # 개선 제안 복원
        for suggestion_data in data.get('suggestions', []):
            analysis.suggestions.append(ImprovementSuggestion.from_dict(suggestion_data))
        
        return analysis


@dataclass
class LLMResponse:
    """LLM 응답을 나타내는 데이터 클래스.
    
    Attributes:
        content: 응답 내용
        model: 사용된 모델명
        usage: 토큰 사용량 정보
        metadata: 추가 메타데이터
    """
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """데이터 검증."""
        if not self.content.strip():
            raise ValueError("Response content cannot be empty")
        if not self.model.strip():
            raise ValueError("Model name cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환."""
        return {
            'content': self.content,
            'model': self.model,
            'usage': self.usage,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """딕셔너리에서 생성."""
        return cls(
            content=data['content'],
            model=data['model'],
            usage=data.get('usage'),
            metadata=data.get('metadata', {})
        )