"""
RAG 영어 학습 시스템의 대화 관련 데이터 모델.

이 모듈은 영어 학습 시스템에서 대화 세션, 메시지, 응답을 관리하기 위한
데이터 클래스들을 포함합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import uuid


@dataclass
class Message:
    """대화에서 하나의 메시지를 나타냅니다.
    
    Attributes:
        role: 메시지 발신자의 역할 ('user' 또는 'assistant')
        content: 메시지의 텍스트 내용
        timestamp: 메시지가 생성된 시간
        metadata: 메시지에 대한 추가 메타데이터
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """초기화 후 메시지 데이터를 검증합니다."""
        if self.role not in ['user', 'assistant']:
            raise ValueError(f"Invalid role: {self.role}. Must be 'user' or 'assistant'")
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """직렬화를 위해 메시지를 딕셔너리로 변환합니다."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """딕셔너리에서 메시지를 생성합니다."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


@dataclass
class LearningPoint:
    """대화 중 식별된 학습 포인트를 나타냅니다.
    
    Attributes:
        topic: 학습되는 주제 또는 개념
        description: 학습한 내용에 대한 설명
        example: 사용 예시 또는 맥락
        difficulty_level: 난이도 (beginner, intermediate, advanced)
        timestamp: 이 학습 포인트가 식별된 시간
    """
    topic: str
    description: str
    example: str = ""
    difficulty_level: str = "intermediate"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate learning point data."""
        if not self.topic.strip():
            raise ValueError("Learning point topic cannot be empty")
        if not self.description.strip():
            raise ValueError("Learning point description cannot be empty")
        if self.difficulty_level not in ['beginner', 'intermediate', 'advanced']:
            raise ValueError(f"Invalid difficulty level: {self.difficulty_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert learning point to dictionary."""
        return {
            'topic': self.topic,
            'description': self.description,
            'example': self.example,
            'difficulty_level': self.difficulty_level,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningPoint':
        """Create learning point from dictionary."""
        return cls(
            topic=data['topic'],
            description=data['description'],
            example=data.get('example', ''),
            difficulty_level=data.get('difficulty_level', 'intermediate'),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class ConversationSession:
    """Represents a complete conversation session.
    
    Attributes:
        session_id: Unique identifier for the session
        start_time: When the session started
        end_time: When the session ended (None if ongoing)
        messages: List of messages in the conversation
        topics_covered: List of topics discussed
        learning_points: List of learning points identified
        user_language: User's native language for explanations
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    messages: List[Message] = field(default_factory=list)
    topics_covered: List[str] = field(default_factory=list)
    learning_points: List[LearningPoint] = field(default_factory=list)
    user_language: str = "korean"
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def end_session(self) -> None:
        """Mark the session as ended."""
        self.end_time = datetime.now()
    
    def is_active(self) -> bool:
        """Check if the session is still active."""
        return self.end_time is None
    
    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'messages': [msg.to_dict() for msg in self.messages],
            'topics_covered': self.topics_covered,
            'learning_points': [lp.to_dict() for lp in self.learning_points],
            'user_language': self.user_language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """Create session from dictionary."""
        session = cls(
            session_id=data['session_id'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
            topics_covered=data.get('topics_covered', []),
            user_language=data.get('user_language', 'korean')
        )
        
        # Reconstruct messages
        for msg_data in data.get('messages', []):
            session.messages.append(Message.from_dict(msg_data))
        
        # Reconstruct learning points
        for lp_data in data.get('learning_points', []):
            session.learning_points.append(LearningPoint.from_dict(lp_data))
        
        return session
    
    def to_json(self) -> str:
        """Convert session to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationSession':
        """Create session from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)