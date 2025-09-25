"""
SessionTracker 클래스에 대한 단위 테스트.

이 모듈은 대화 세션 추적 및 관리 기능의 정확성을 검증합니다.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.document_rag_english_study.conversation.session_tracker import SessionTracker
from src.document_rag_english_study.models import (
    ConversationSession,
    ConversationSummary,
    Interaction,
    LearningPoint,
    Message
)


class TestSessionTracker:
    """SessionTracker 클래스의 단위 테스트."""
    
    @pytest.fixture
    def temp_dir(self):
        """테스트용 임시 디렉토리를 생성합니다."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def session_tracker(self, temp_dir):
        """테스트용 SessionTracker 인스턴스를 생성합니다."""
        return SessionTracker(sessions_dir=temp_dir)
    
    @pytest.fixture
    def sample_session(self):
        """테스트용 샘플 세션을 생성합니다."""
        session = ConversationSession(user_language="korean")
        
        # 샘플 메시지 추가
        user_msg = Message(role="user", content="Hello, how are you?")
        assistant_msg = Message(role="assistant", content="I'm doing well, thank you!")
        session.add_message(user_msg)
        session.add_message(assistant_msg)
        
        # 샘플 학습 포인트 추가
        learning_point = LearningPoint(
            topic="greeting",
            description="Basic greeting expressions",
            example="How are you? - I'm doing well"
        )
        session.learning_points.append(learning_point)
        session.topics_covered.append("greetings")
        
        return session
    
    @pytest.fixture
    def sample_interaction(self):
        """테스트용 샘플 상호작용을 생성합니다."""
        user_msg = Message(role="user", content="What's the weather like?")
        assistant_msg = Message(role="assistant", content="It's sunny today!")
        
        learning_point = LearningPoint(
            topic="weather vocabulary",
            description="Weather-related expressions",
            example="sunny, cloudy, rainy"
        )
        
        return Interaction(
            user_message=user_msg,
            assistant_message=assistant_msg,
            learning_points=[learning_point],
            topics=["weather"]
        )
    
    def test_init(self, temp_dir):
        """SessionTracker 초기화를 테스트합니다."""
        tracker = SessionTracker(sessions_dir=temp_dir)
        
        assert tracker.sessions_dir == Path(temp_dir)
        assert tracker.sessions_dir.exists()
        assert len(tracker._active_sessions) == 0
    
    def test_create_session(self, session_tracker):
        """새 세션 생성을 테스트합니다."""
        session = session_tracker.create_session(user_language="english")
        
        assert isinstance(session, ConversationSession)
        assert session.user_language == "english"
        assert session.session_id in session_tracker._active_sessions
        assert session.is_active()
        assert len(session.messages) == 0
    
    def test_create_session_default_language(self, session_tracker):
        """기본 언어로 세션 생성을 테스트합니다."""
        session = session_tracker.create_session()
        
        assert session.user_language == "korean"
    
    def test_update_session(self, session_tracker, sample_session, sample_interaction):
        """세션 업데이트를 테스트합니다."""
        # 세션을 활성 세션으로 등록
        session_tracker._active_sessions[sample_session.session_id] = sample_session
        
        initial_message_count = len(sample_session.messages)
        initial_learning_points = len(sample_session.learning_points)
        initial_topics = len(sample_session.topics_covered)
        
        session_tracker.update_session(sample_session, sample_interaction)
        
        # 메시지가 추가되었는지 확인
        assert len(sample_session.messages) == initial_message_count + 2
        
        # 학습 포인트가 추가되었는지 확인
        assert len(sample_session.learning_points) == initial_learning_points + 1
        
        # 주제가 추가되었는지 확인
        assert len(sample_session.topics_covered) == initial_topics + 1
        assert "weather" in sample_session.topics_covered
    
    def test_update_session_duplicate_topics(self, session_tracker, sample_session):
        """중복 주제 처리를 테스트합니다."""
        session_tracker._active_sessions[sample_session.session_id] = sample_session
        
        # 이미 존재하는 주제로 상호작용 생성
        user_msg = Message(role="user", content="Good morning!")
        assistant_msg = Message(role="assistant", content="Good morning to you too!")
        
        interaction = Interaction(
            user_message=user_msg,
            assistant_message=assistant_msg,
            topics=["greetings"]  # 이미 존재하는 주제
        )
        
        initial_topics_count = len(sample_session.topics_covered)
        session_tracker.update_session(sample_session, interaction)
        
        # 중복 주제는 추가되지 않아야 함
        assert len(sample_session.topics_covered) == initial_topics_count
    
    def test_save_session(self, session_tracker, sample_session):
        """세션 저장을 테스트합니다."""
        session_tracker.save_session(sample_session)
        
        session_file = session_tracker.sessions_dir / f"{sample_session.session_id}.json"
        assert session_file.exists()
        
        # 저장된 내용 확인
        with open(session_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data['session_id'] == sample_session.session_id
        assert saved_data['user_language'] == sample_session.user_language
    
    def test_save_session_io_error(self, session_tracker, sample_session):
        """세션 저장 중 IO 오류를 테스트합니다."""
        # 존재하지 않는 디렉토리로 설정
        session_tracker.sessions_dir = Path("/invalid/path")
        
        with pytest.raises(IOError):
            session_tracker.save_session(sample_session)
    
    def test_load_session(self, session_tracker, sample_session):
        """세션 로드를 테스트합니다."""
        # 먼저 세션을 저장
        session_tracker.save_session(sample_session)
        
        # 세션 로드
        loaded_session = session_tracker.load_session(sample_session.session_id)
        
        assert loaded_session is not None
        assert loaded_session.session_id == sample_session.session_id
        assert loaded_session.user_language == sample_session.user_language
        assert len(loaded_session.messages) == len(sample_session.messages)
        assert len(loaded_session.learning_points) == len(sample_session.learning_points)
    
    def test_load_nonexistent_session(self, session_tracker):
        """존재하지 않는 세션 로드를 테스트합니다."""
        loaded_session = session_tracker.load_session("nonexistent_id")
        assert loaded_session is None
    
    def test_load_session_invalid_json(self, session_tracker):
        """잘못된 JSON 파일 로드를 테스트합니다."""
        # 잘못된 JSON 파일 생성
        invalid_file = session_tracker.sessions_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        loaded_session = session_tracker.load_session("invalid")
        assert loaded_session is None
    
    def test_end_session(self, session_tracker, sample_session):
        """세션 종료를 테스트합니다."""
        # 세션을 활성 세션으로 등록
        session_tracker._active_sessions[sample_session.session_id] = sample_session
        
        assert sample_session.is_active()
        
        summary = session_tracker.end_session(sample_session)
        
        # 세션이 종료되었는지 확인
        assert not sample_session.is_active()
        assert sample_session.end_time is not None
        
        # 활성 세션에서 제거되었는지 확인
        assert sample_session.session_id not in session_tracker._active_sessions
        
        # 요약이 생성되었는지 확인
        assert isinstance(summary, ConversationSummary)
        assert summary.session_id == sample_session.session_id
        
        # 세션과 요약이 저장되었는지 확인
        session_file = session_tracker.sessions_dir / f"{sample_session.session_id}.json"
        summary_file = session_tracker.sessions_dir / f"{sample_session.session_id}_summary.json"
        assert session_file.exists()
        assert summary_file.exists()
    
    def test_get_session_summary(self, session_tracker, sample_session):
        """세션 요약 조회를 테스트합니다."""
        # 세션을 종료하여 요약 생성
        session_tracker._active_sessions[sample_session.session_id] = sample_session
        summary = session_tracker.end_session(sample_session)
        
        # 요약 조회
        retrieved_summary = session_tracker.get_session_summary(sample_session.session_id)
        
        assert retrieved_summary is not None
        assert retrieved_summary.session_id == summary.session_id
        assert retrieved_summary.total_messages == summary.total_messages
    
    def test_get_nonexistent_session_summary(self, session_tracker):
        """존재하지 않는 세션 요약 조회를 테스트합니다."""
        summary = session_tracker.get_session_summary("nonexistent_id")
        assert summary is None
    
    def test_get_active_sessions(self, session_tracker):
        """활성 세션 목록 조회를 테스트합니다."""
        # 여러 세션 생성
        session1 = session_tracker.create_session()
        session2 = session_tracker.create_session()
        
        active_sessions = session_tracker.get_active_sessions()
        
        assert len(active_sessions) == 2
        assert session1 in active_sessions
        assert session2 in active_sessions
    
    def test_list_all_sessions(self, session_tracker, sample_session):
        """모든 세션 목록 조회를 테스트합니다."""
        # 세션 저장
        session_tracker.save_session(sample_session)
        
        # 다른 세션도 생성하고 저장
        another_session = ConversationSession()
        session_tracker.save_session(another_session)
        
        session_ids = session_tracker.list_all_sessions()
        
        assert len(session_ids) == 2
        assert sample_session.session_id in session_ids
        assert another_session.session_id in session_ids
    
    def test_list_all_sessions_excludes_summaries(self, session_tracker, sample_session):
        """세션 목록에서 요약 파일이 제외되는지 테스트합니다."""
        # 세션과 요약 저장
        session_tracker._active_sessions[sample_session.session_id] = sample_session
        session_tracker.end_session(sample_session)
        
        session_ids = session_tracker.list_all_sessions()
        
        # 세션 ID만 포함되고 요약은 제외되어야 함
        assert len(session_ids) == 1
        assert sample_session.session_id in session_ids
    
    def test_get_user_progress_stats(self, session_tracker):
        """사용자 진행 통계 조회를 테스트합니다."""
        # 여러 세션 생성 및 저장
        for i in range(3):
            session = ConversationSession(user_language="korean")
            session.add_message(Message(role="user", content=f"Message {i}"))
            session.add_message(Message(role="assistant", content=f"Response {i}"))
            session.learning_points.append(LearningPoint(
                topic=f"topic_{i}",
                description=f"Description {i}"
            ))
            session.topics_covered.append(f"topic_{i}")
            session.end_session()
            session_tracker.save_session(session)
        
        stats = session_tracker.get_user_progress_stats("korean")
        
        assert stats['total_sessions'] == 3
        assert stats['total_messages'] == 6  # 각 세션당 2개 메시지
        assert stats['total_learning_points'] == 3
        assert len(stats['topics_covered']) == 3
        assert len(stats['recent_sessions']) == 3
    
    def test_get_user_progress_stats_language_filter(self, session_tracker):
        """언어별 진행 통계 필터링을 테스트합니다."""
        # 다른 언어의 세션 생성
        korean_session = ConversationSession(user_language="korean")
        english_session = ConversationSession(user_language="english")
        
        session_tracker.save_session(korean_session)
        session_tracker.save_session(english_session)
        
        korean_stats = session_tracker.get_user_progress_stats("korean")
        english_stats = session_tracker.get_user_progress_stats("english")
        
        assert korean_stats['total_sessions'] == 1
        assert english_stats['total_sessions'] == 1
    
    def test_generate_session_summary(self, session_tracker, sample_session):
        """세션 요약 생성을 테스트합니다."""
        # 세션에 더 많은 데이터 추가
        sample_session.add_message(Message(role="user", content="What's a verb?"))
        sample_session.add_message(Message(role="assistant", content="A verb is an action word."))
        
        # 문법 관련 학습 포인트 추가
        grammar_point = LearningPoint(
            topic="grammar: verbs",
            description="Understanding verbs as action words"
        )
        sample_session.learning_points.append(grammar_point)
        
        # 어휘 관련 학습 포인트 추가
        vocab_point = LearningPoint(
            topic="vocabulary: action words",
            description="Learning action-related vocabulary"
        )
        sample_session.learning_points.append(vocab_point)
        
        sample_session.end_session()
        
        summary = session_tracker._generate_session_summary(sample_session)
        
        assert isinstance(summary, ConversationSummary)
        assert summary.session_id == sample_session.session_id
        assert summary.total_messages == len(sample_session.messages)
        assert summary.duration_seconds == sample_session.get_duration()
        assert len(summary.learning_points) == len(sample_session.learning_points)
        assert len(summary.grammar_points) > 0  # 문법 포인트가 식별되어야 함
        assert len(summary.key_vocabulary) > 0  # 어휘가 식별되어야 함
        assert summary.user_progress != ""
        assert len(summary.recommendations) > 0
    
    def test_assess_user_progress_short_conversation(self, session_tracker):
        """짧은 대화에 대한 진행 상황 평가를 테스트합니다."""
        session = ConversationSession()
        # 5개 메시지만 추가 (짧은 대화)
        for i in range(5):
            session.add_message(Message(role="user", content=f"Message {i}"))
        
        progress = session_tracker._assess_user_progress(session)
        assert "짧은 대화" in progress
    
    def test_assess_user_progress_medium_conversation(self, session_tracker):
        """중간 길이 대화에 대한 진행 상황 평가를 테스트합니다."""
        session = ConversationSession()
        # 20개 메시지 추가 (중간 길이)
        for i in range(20):
            session.add_message(Message(role="user", content=f"Message {i}"))
        
        # 학습 포인트와 주제 추가
        session.learning_points.append(LearningPoint(topic="test", description="test"))
        session.topics_covered.append("test_topic")
        
        progress = session_tracker._assess_user_progress(session)
        assert "활발한 대화" in progress
    
    def test_assess_user_progress_long_conversation(self, session_tracker):
        """긴 대화에 대한 진행 상황 평가를 테스트합니다."""
        session = ConversationSession()
        # 40개 메시지 추가 (긴 대화)
        for i in range(40):
            session.add_message(Message(role="user", content=f"Message {i}"))
        
        # 여러 학습 포인트와 주제 추가
        for i in range(5):
            session.learning_points.append(LearningPoint(topic=f"topic_{i}", description="test"))
            session.topics_covered.append(f"topic_{i}")
        
        progress = session_tracker._assess_user_progress(session)
        assert "매우 활발한" in progress
    
    def test_generate_recommendations_many_learning_points(self, session_tracker):
        """많은 학습 포인트에 대한 권장사항 생성을 테스트합니다."""
        session = ConversationSession()
        # 6개 학습 포인트 추가
        for i in range(6):
            session.learning_points.append(LearningPoint(topic=f"topic_{i}", description="test"))
        
        recommendations = session_tracker._generate_recommendations(session)
        assert any("복습" in rec for rec in recommendations)
    
    def test_generate_recommendations_few_learning_points(self, session_tracker):
        """적은 학습 포인트에 대한 권장사항 생성을 테스트합니다."""
        session = ConversationSession()
        # 2개 학습 포인트만 추가
        for i in range(2):
            session.learning_points.append(LearningPoint(topic=f"topic_{i}", description="test"))
        
        recommendations = session_tracker._generate_recommendations(session)
        assert any("더 많은" in rec for rec in recommendations)
    
    def test_generate_recommendations_many_topics(self, session_tracker):
        """다양한 주제에 대한 권장사항 생성을 테스트합니다."""
        session = ConversationSession()
        # 4개 주제 추가
        for i in range(4):
            session.topics_covered.append(f"topic_{i}")
        
        recommendations = session_tracker._generate_recommendations(session)
        assert any("핵심 표현" in rec for rec in recommendations)
    
    def test_generate_recommendations_long_conversation(self, session_tracker):
        """긴 대화에 대한 권장사항 생성을 테스트합니다."""
        session = ConversationSession()
        # 25개 메시지 추가
        for i in range(25):
            session.add_message(Message(role="user", content=f"Message {i}"))
        
        recommendations = session_tracker._generate_recommendations(session)
        assert any("긴 대화" in rec for rec in recommendations)
    
    def test_save_session_summary(self, session_tracker):
        """세션 요약 저장을 테스트합니다."""
        summary = ConversationSummary(
            session_id="test_session",
            duration_seconds=300.0,
            total_messages=10,
            topics_covered=["test_topic"],
            learning_points=[]
        )
        
        session_tracker._save_session_summary(summary)
        
        summary_file = session_tracker.sessions_dir / f"{summary.session_id}_summary.json"
        assert summary_file.exists()
        
        # 저장된 내용 확인
        with open(summary_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data['session_id'] == summary.session_id
        assert saved_data['duration_seconds'] == summary.duration_seconds
    
    def test_save_session_summary_io_error(self, session_tracker):
        """세션 요약 저장 중 IO 오류를 테스트합니다."""
        summary = ConversationSummary(
            session_id="test_session",
            duration_seconds=300.0,
            total_messages=10,
            topics_covered=[],
            learning_points=[]
        )
        
        # 존재하지 않는 디렉토리로 설정
        session_tracker.sessions_dir = Path("/invalid/path")
        
        with pytest.raises(IOError):
            session_tracker._save_session_summary(summary)


if __name__ == "__main__":
    pytest.main([__file__])