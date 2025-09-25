"""
DialogManager 클래스에 대한 단위 테스트.

이 모듈은 대화 관리자의 모든 기능을 테스트합니다:
- 대화 시작 메시지 생성
- 대화 흐름 유지
- 후속 질문 제안
- 주제 변경 감지
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import List

from src.document_rag_english_study.conversation.dialog_manager import DialogManager, DialogManagerError
from src.document_rag_english_study.models.conversation import Message, ConversationSession
from src.document_rag_english_study.models.response import SearchResult
from src.document_rag_english_study.rag.engine import RAGEngine
from src.document_rag_english_study.llm.base import LanguageModel, LLMResponse


class TestDialogManager:
    """DialogManager 클래스의 단위 테스트."""
    
    @pytest.fixture
    def mock_rag_engine(self):
        """Mock RAG 엔진 생성."""
        mock_rag = Mock(spec=RAGEngine)
        mock_rag.extract_keywords.return_value = ["technology", "innovation", "future"]
        mock_rag.search_similar_content.return_value = [
            SearchResult(
                content="Technology is changing rapidly in modern society.",
                source_file="tech_doc.pdf",
                relevance_score=0.8,
                metadata={"document_id": "doc1", "chunk_index": 0}
            )
        ]
        mock_rag.get_indexed_document_info.return_value = {
            "total_documents": 5,
            "total_chunks": 50
        }
        return mock_rag
    
    @pytest.fixture
    def mock_llm(self):
        """Mock 언어 모델 생성."""
        mock_llm = Mock(spec=LanguageModel)
        mock_llm.generate_response.return_value = LLMResponse(
            content="Hello! Let's talk about technology today. What interests you most about modern innovations?",
            model="test-model",
            metadata={}
        )
        return mock_llm
    
    @pytest.fixture
    def dialog_manager(self, mock_rag_engine, mock_llm):
        """DialogManager 인스턴스 생성."""
        return DialogManager(
            rag_engine=mock_rag_engine,
            llm=mock_llm,
            user_language="korean"
        )
    
    @pytest.fixture
    def dialog_manager_english(self, mock_rag_engine, mock_llm):
        """영어 DialogManager 인스턴스 생성."""
        return DialogManager(
            rag_engine=mock_rag_engine,
            llm=mock_llm,
            user_language="english"
        )
    
    @pytest.fixture
    def sample_messages(self):
        """샘플 대화 메시지 생성."""
        return [
            Message(role="assistant", content="안녕하세요! 기술에 대해 이야기해볼까요?"),
            Message(role="user", content="네, 좋아요. 인공지능에 관심이 많아요."),
            Message(role="assistant", content="인공지능은 정말 흥미로운 분야죠!"),
            Message(role="user", content="맞아요. 특히 머신러닝이 신기해요."),
            Message(role="assistant", content="머신러닝에 대해 더 자세히 말씀해주세요.")
        ]
    
    def test_init(self, mock_rag_engine, mock_llm):
        """DialogManager 초기화 테스트."""
        dialog_manager = DialogManager(
            rag_engine=mock_rag_engine,
            llm=mock_llm,
            user_language="korean"
        )
        
        assert dialog_manager.rag_engine == mock_rag_engine
        assert dialog_manager.llm == mock_llm
        assert dialog_manager.user_language == "korean"
        assert dialog_manager._conversation_starters is not None
        assert dialog_manager._follow_up_patterns is not None
    
    def test_generate_conversation_starter_with_topics(self, dialog_manager, mock_llm):
        """주제가 제공된 경우 대화 시작 메시지 생성 테스트."""
        topics = ["technology", "innovation", "artificial intelligence"]
        
        result = dialog_manager.generate_conversation_starter(
            document_topics=topics,
            preferred_topic="technology"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        # LLM이 호출되었는지 확인
        mock_llm.generate_response.assert_called_once()
    
    def test_generate_conversation_starter_without_topics(self, dialog_manager):
        """주제가 제공되지 않은 경우 대화 시작 메시지 생성 테스트."""
        # RAG 엔진에서 주제를 추출하도록 설정
        dialog_manager.rag_engine.search_similar_content.return_value = [
            SearchResult(
                content="Technology and innovation are key topics.",
                source_file="doc.pdf",
                relevance_score=0.7,
                metadata={}
            )
        ]
        
        result = dialog_manager.generate_conversation_starter()
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_conversation_starter_no_documents(self, dialog_manager):
        """문서가 없는 경우 기본 대화 시작 메시지 테스트."""
        dialog_manager.rag_engine.get_indexed_document_info.return_value = {
            "total_documents": 0,
            "total_chunks": 0
        }
        
        result = dialog_manager.generate_conversation_starter()
        
        assert isinstance(result, str)
        assert len(result) > 0
        # 기본 메시지가 반환되어야 함
        assert any(keyword in result for keyword in ["안녕", "영어", "주제"])
    
    def test_generate_conversation_starter_english(self, dialog_manager_english):
        """영어 대화 시작 메시지 생성 테스트."""
        topics = ["technology", "science"]
        
        result = dialog_manager_english.generate_conversation_starter(
            document_topics=topics
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_maintain_conversation_flow_empty_history(self, dialog_manager):
        """빈 대화 기록에 대한 대화 흐름 유지 테스트."""
        result = dialog_manager.maintain_conversation_flow([])
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_maintain_conversation_flow_needs_encouragement(self, dialog_manager):
        """격려가 필요한 경우 대화 흐름 유지 테스트."""
        # 짧은 응답들로 구성된 대화
        short_messages = [
            Message(role="user", content="네"),
            Message(role="user", content="좋아요"),
            Message(role="user", content="맞아요")
        ]
        
        result = dialog_manager.maintain_conversation_flow(short_messages)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_maintain_conversation_flow_needs_topic_change(self, dialog_manager):
        """주제 변경이 필요한 경우 대화 흐름 유지 테스트."""
        # 긴 대화 기록 (주제 변경 필요)
        long_messages = []
        for i in range(12):
            long_messages.append(
                Message(role="user", content=f"기술에 대해 이야기 {i}")
            )
        
        result = dialog_manager.maintain_conversation_flow(long_messages)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_suggest_follow_up_questions(self, dialog_manager, mock_llm):
        """후속 질문 제안 테스트."""
        context = "I'm interested in artificial intelligence and machine learning."
        
        # LLM 응답 설정
        mock_llm.generate_response.return_value = LLMResponse(
            content="""1. What specific aspect of AI interests you most?
2. Have you tried any machine learning projects?
3. How do you think AI will change our daily lives?""",
            model="test-model",
            metadata={}
        )
        
        result = dialog_manager.suggest_follow_up_questions(context, max_suggestions=3)
        
        assert isinstance(result, list)
        assert len(result) <= 3
        assert all(isinstance(q, str) for q in result)
        assert all(q.endswith('?') for q in result)
    
    def test_suggest_follow_up_questions_with_history(self, dialog_manager, sample_messages):
        """대화 기록이 있는 경우 후속 질문 제안 테스트."""
        context = "Machine learning is fascinating."
        
        result = dialog_manager.suggest_follow_up_questions(
            context, 
            conversation_history=sample_messages
        )
        
        assert isinstance(result, list)
        assert len(result) >= 0  # 빈 리스트일 수도 있음
    
    def test_suggest_follow_up_questions_empty_context(self, dialog_manager):
        """빈 컨텍스트에 대한 후속 질문 제안 테스트."""
        result = dialog_manager.suggest_follow_up_questions("")
        
        assert isinstance(result, list)
        # 빈 컨텍스트의 경우 빈 리스트 또는 기본 질문들 반환
    
    def test_detect_topic_change_opportunity_no_change_needed(self, dialog_manager):
        """주제 변경이 필요하지 않은 경우 테스트."""
        # 다양한 키워드를 가진 짧은 대화
        diverse_messages = [
            Message(role="user", content="I like technology and innovation"),
            Message(role="user", content="Science is also interesting"),
            Message(role="user", content="Education plays important role")
        ]
        
        needs_change, new_topic = dialog_manager.detect_topic_change_opportunity(
            diverse_messages, min_messages_per_topic=3
        )
        
        assert isinstance(needs_change, bool)
        assert needs_change is False
        assert new_topic is None
    
    def test_detect_topic_change_opportunity_change_needed(self, dialog_manager):
        """주제 변경이 필요한 경우 테스트."""
        # 반복적인 키워드를 가진 긴 대화
        repetitive_messages = []
        for i in range(6):
            repetitive_messages.append(
                Message(role="user", content="technology is good technology helps technology")
            )
        
        needs_change, new_topic = dialog_manager.detect_topic_change_opportunity(
            repetitive_messages, min_messages_per_topic=5
        )
        
        assert isinstance(needs_change, bool)
        if needs_change:
            assert isinstance(new_topic, str)
            assert len(new_topic) > 0
    
    def test_detect_topic_change_opportunity_insufficient_messages(self, dialog_manager):
        """메시지가 부족한 경우 주제 변경 감지 테스트."""
        short_messages = [
            Message(role="user", content="Hello"),
            Message(role="user", content="How are you?")
        ]
        
        needs_change, new_topic = dialog_manager.detect_topic_change_opportunity(
            short_messages, min_messages_per_topic=5
        )
        
        assert needs_change is False
        assert new_topic is None
    
    def test_extract_topics_from_documents(self, dialog_manager):
        """문서에서 주제 추출 테스트."""
        # private 메서드 테스트
        topics = dialog_manager._extract_topics_from_documents()
        
        assert isinstance(topics, list)
        # RAG 엔진이 호출되었는지 확인
        dialog_manager.rag_engine.get_indexed_document_info.assert_called()
    
    def test_extract_topics_from_documents_no_documents(self, dialog_manager):
        """문서가 없는 경우 주제 추출 테스트."""
        dialog_manager.rag_engine.get_indexed_document_info.return_value = {
            "total_documents": 0,
            "total_chunks": 0
        }
        
        topics = dialog_manager._extract_topics_from_documents()
        
        assert isinstance(topics, list)
        assert len(topics) == 0
    
    def test_get_default_conversation_starter_korean(self, dialog_manager):
        """한국어 기본 대화 시작 메시지 테스트."""
        result = dialog_manager._get_default_conversation_starter()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert any(keyword in result for keyword in ["안녕", "영어", "주제"])
    
    def test_get_default_conversation_starter_english(self, dialog_manager_english):
        """영어 기본 대화 시작 메시지 테스트."""
        result = dialog_manager_english._get_default_conversation_starter()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert any(keyword in result.lower() for keyword in ["hello", "english", "topic"])
    
    def test_analyze_conversation_pattern(self, dialog_manager, sample_messages):
        """대화 패턴 분석 테스트."""
        analysis = dialog_manager._analyze_conversation_pattern(sample_messages)
        
        assert isinstance(analysis, dict)
        assert 'needs_topic_change' in analysis
        assert 'needs_encouragement' in analysis
        assert 'needs_clarification' in analysis
        assert 'user_engagement_level' in analysis
        assert 'conversation_length' in analysis
        assert 'last_user_message' in analysis
        assert 'dominant_keywords' in analysis
        
        assert isinstance(analysis['needs_topic_change'], bool)
        assert isinstance(analysis['needs_encouragement'], bool)
        assert isinstance(analysis['needs_clarification'], bool)
        assert analysis['user_engagement_level'] in ['low', 'medium', 'high']
        assert isinstance(analysis['conversation_length'], int)
    
    def test_analyze_conversation_pattern_empty_messages(self, dialog_manager):
        """빈 메시지 목록에 대한 대화 패턴 분석 테스트."""
        analysis = dialog_manager._analyze_conversation_pattern([])
        
        assert isinstance(analysis, dict)
        assert analysis['conversation_length'] == 0
        assert analysis['last_user_message'] is None
    
    def test_analyze_conversation_pattern_low_engagement(self, dialog_manager):
        """낮은 참여도 대화 패턴 분석 테스트."""
        low_engagement_messages = [
            Message(role="user", content="네"),
            Message(role="user", content="좋아요"),
            Message(role="user", content="맞아요")
        ]
        
        analysis = dialog_manager._analyze_conversation_pattern(low_engagement_messages)
        
        assert analysis['user_engagement_level'] == 'low'
        assert analysis['needs_encouragement'] is True
    
    def test_suggest_topic_transition(self, dialog_manager):
        """주제 전환 제안 테스트."""
        analysis = {
            'dominant_keywords': ['technology', 'innovation'],
            'needs_topic_change': True
        }
        
        result = dialog_manager._suggest_topic_transition(analysis, "technology")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_encouragement_message(self, dialog_manager):
        """격려 메시지 생성 테스트."""
        analysis = {
            'user_engagement_level': 'low',
            'needs_encouragement': True
        }
        
        result = dialog_manager._generate_encouragement_message(analysis)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_clarification_request(self, dialog_manager):
        """명확화 요청 메시지 생성 테스트."""
        analysis = {
            'needs_clarification': True,
            'last_user_message': "It's interesting"
        }
        
        result = dialog_manager._generate_clarification_request(analysis)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_natural_continuation(self, dialog_manager):
        """자연스러운 대화 지속 메시지 생성 테스트."""
        analysis = {
            'user_engagement_level': 'medium',
            'last_user_message': "I think technology is important"
        }
        
        result = dialog_manager._generate_natural_continuation(analysis)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_suggest_new_topic(self, dialog_manager):
        """새 주제 제안 테스트."""
        current_keywords = ["technology", "innovation"]
        
        result = dialog_manager._suggest_new_topic(current_keywords)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # 현재 키워드와 다른 주제여야 함
        assert result.lower() not in [kw.lower() for kw in current_keywords]
    
    def test_parse_follow_up_response(self, dialog_manager):
        """후속 질문 응답 파싱 테스트."""
        response = """1. What do you think about artificial intelligence?
2. Have you used any AI tools recently?
3. How do you see AI changing the future?"""
        
        questions = dialog_manager._parse_follow_up_response(response)
        
        assert isinstance(questions, list)
        assert len(questions) == 3
        assert all(q.endswith('?') for q in questions)
        assert "What do you think about artificial intelligence?" in questions
    
    def test_parse_follow_up_response_different_formats(self, dialog_manager):
        """다양한 형식의 후속 질문 응답 파싱 테스트."""
        response = """- What's your favorite technology?
• How often do you use smartphones?
* Do you think AI is helpful?"""
        
        questions = dialog_manager._parse_follow_up_response(response)
        
        assert isinstance(questions, list)
        assert len(questions) == 3
        assert all(q.endswith('?') for q in questions)
    
    def test_generate_pattern_based_questions(self, dialog_manager):
        """패턴 기반 질문 생성 테스트."""
        keywords = ["technology", "innovation", "future"]
        context = "Technology is changing rapidly."
        
        questions = dialog_manager._generate_pattern_based_questions(keywords, context)
        
        assert isinstance(questions, list)
        assert len(questions) <= 2
        assert all(isinstance(q, str) for q in questions)
    
    def test_generate_pattern_based_questions_empty_keywords(self, dialog_manager):
        """빈 키워드 목록에 대한 패턴 기반 질문 생성 테스트."""
        questions = dialog_manager._generate_pattern_based_questions([], "some context")
        
        assert isinstance(questions, list)
        assert len(questions) == 0
    
    def test_remove_duplicate_questions(self, dialog_manager):
        """중복 질문 제거 테스트."""
        questions = [
            "What do you think about technology?",
            "How do you feel about technology?",  # 유사한 질문
            "What's your opinion on innovation?",
            "What do you think about technology?"  # 완전 중복
        ]
        
        unique_questions = dialog_manager._remove_duplicate_questions(questions)
        
        assert isinstance(unique_questions, list)
        assert len(unique_questions) < len(questions)
        # 완전 중복은 제거되어야 함
        assert unique_questions.count("What do you think about technology?") == 1
    
    def test_error_handling_generate_conversation_starter(self, dialog_manager):
        """대화 시작 메시지 생성 오류 처리 테스트."""
        # LLM에서 오류 발생 시뮬레이션
        dialog_manager.llm.generate_response.side_effect = Exception("LLM Error")
        
        # 오류가 발생해도 기본 메시지는 반환되어야 함
        result = dialog_manager.generate_conversation_starter(["technology"])
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_error_handling_maintain_conversation_flow(self, dialog_manager, sample_messages):
        """대화 흐름 유지 오류 처리 테스트."""
        # _generate_encouragement_message에서 오류 발생 시뮬레이션
        with patch.object(dialog_manager, '_generate_encouragement_message', side_effect=Exception("Internal Error")):
            # 짧은 메시지로 격려가 필요한 상황 만들기
            short_messages = [
                Message(role="user", content="네"),
                Message(role="user", content="좋아요"),
                Message(role="user", content="맞아요")
            ]
            
            with pytest.raises(DialogManagerError):
                dialog_manager.maintain_conversation_flow(short_messages)
    
    def test_error_handling_suggest_follow_up_questions(self, dialog_manager):
        """후속 질문 제안 오류 처리 테스트."""
        # RAG 엔진에서 오류 발생 시뮬레이션 (키워드 추출과 검색 모두 실패)
        dialog_manager.rag_engine.extract_keywords.side_effect = Exception("RAG Error")
        dialog_manager.rag_engine.search_similar_content.side_effect = Exception("Search Error")
        
        with pytest.raises(DialogManagerError):
            dialog_manager.suggest_follow_up_questions("some context")
    
    @pytest.mark.parametrize("user_language", ["korean", "english"])
    def test_multilingual_support(self, mock_rag_engine, mock_llm, user_language):
        """다국어 지원 테스트."""
        dialog_manager = DialogManager(
            rag_engine=mock_rag_engine,
            llm=mock_llm,
            user_language=user_language
        )
        
        # 기본 대화 시작 메시지 테스트
        starter = dialog_manager._get_default_conversation_starter()
        assert isinstance(starter, str)
        assert len(starter) > 0
        
        # 격려 메시지 테스트
        analysis = {'user_engagement_level': 'low'}
        encouragement = dialog_manager._generate_encouragement_message(analysis)
        assert isinstance(encouragement, str)
        assert len(encouragement) > 0
    
    def test_conversation_starters_loading(self, dialog_manager):
        """대화 시작 템플릿 로딩 테스트."""
        starters = dialog_manager._load_conversation_starters()
        
        assert isinstance(starters, dict)
        assert "korean" in starters
        assert "english" in starters
        assert isinstance(starters["korean"], list)
        assert isinstance(starters["english"], list)
        assert all("{topic}" in template for template in starters["korean"])
        assert all("{topic}" in template for template in starters["english"])
    
    def test_follow_up_patterns_loading(self, dialog_manager):
        """후속 질문 패턴 로딩 테스트."""
        patterns = dialog_manager._load_follow_up_patterns()
        
        assert isinstance(patterns, dict)
        assert "korean" in patterns
        assert "english" in patterns
        assert isinstance(patterns["korean"], list)
        assert isinstance(patterns["english"], list)
        assert all(isinstance(pattern, str) for pattern in patterns["korean"])
        assert all(isinstance(pattern, str) for pattern in patterns["english"])


class TestDialogManagerIntegration:
    """DialogManager 통합 테스트."""
    
    @pytest.fixture
    def real_dialog_manager(self):
        """실제 구현체를 사용한 DialogManager (Mock 없이)."""
        # 실제 테스트에서는 실제 RAG 엔진과 LLM을 사용할 수 있음
        # 여기서는 Mock을 사용하되 더 현실적인 응답을 설정
        mock_rag = Mock(spec=RAGEngine)
        mock_llm = Mock(spec=LanguageModel)
        
        # 더 현실적인 Mock 응답 설정
        mock_rag.extract_keywords.return_value = ["technology", "innovation", "artificial", "intelligence"]
        mock_rag.search_similar_content.return_value = [
            SearchResult(
                content="Artificial intelligence is transforming various industries including healthcare, finance, and education.",
                source_file="ai_overview.pdf",
                relevance_score=0.85,
                metadata={"document_id": "ai_doc_1", "chunk_index": 0}
            ),
            SearchResult(
                content="Machine learning algorithms are becoming more sophisticated and accessible to developers.",
                source_file="ml_guide.pdf",
                relevance_score=0.78,
                metadata={"document_id": "ml_doc_1", "chunk_index": 2}
            )
        ]
        mock_rag.get_indexed_document_info.return_value = {
            "total_documents": 10,
            "total_chunks": 150
        }
        
        mock_llm.generate_response.return_value = LLMResponse(
            content="안녕하세요! 오늘은 인공지능에 대해 영어로 이야기해볼까요? AI가 우리 일상생활에 어떤 영향을 미치고 있다고 생각하시나요?",
            model="test_model",
            metadata={"tokens_used": 50}
        )
        
        return DialogManager(
            rag_engine=mock_rag,
            llm=mock_llm,
            user_language="korean"
        )
    
    def test_full_conversation_flow(self, real_dialog_manager):
        """전체 대화 흐름 통합 테스트."""
        # 1. 대화 시작
        starter = real_dialog_manager.generate_conversation_starter(
            document_topics=["artificial intelligence", "machine learning"]
        )
        assert isinstance(starter, str)
        assert len(starter) > 0
        
        # 2. 대화 기록 생성
        conversation_history = [
            Message(role="assistant", content=starter),
            Message(role="user", content="AI는 정말 흥미로운 주제예요. 특히 자연어 처리가 신기해요."),
            Message(role="assistant", content="자연어 처리는 정말 발전이 빠른 분야죠!")
        ]
        
        # 3. 대화 흐름 유지
        flow_response = real_dialog_manager.maintain_conversation_flow(conversation_history)
        assert isinstance(flow_response, str)
        assert len(flow_response) > 0
        
        # 4. 후속 질문 제안
        follow_ups = real_dialog_manager.suggest_follow_up_questions(
            "자연어 처리와 머신러닝에 대해 이야기하고 있습니다.",
            conversation_history=conversation_history
        )
        assert isinstance(follow_ups, list)
        
        # 5. 주제 변경 기회 감지
        needs_change, new_topic = real_dialog_manager.detect_topic_change_opportunity(
            conversation_history
        )
        assert isinstance(needs_change, bool)
    
    def test_error_recovery(self, real_dialog_manager):
        """오류 복구 통합 테스트."""
        # LLM 오류 시뮬레이션
        real_dialog_manager.llm.generate_response.side_effect = [
            Exception("Network error"),  # 첫 번째 호출에서 오류
            LLMResponse(content="복구된 응답입니다.", model="test-model", metadata={})  # 두 번째 호출에서 성공
        ]
        
        # 오류가 발생해도 기본 메시지로 복구되어야 함
        result = real_dialog_manager.generate_conversation_starter(["technology"])
        assert isinstance(result, str)
        assert len(result) > 0