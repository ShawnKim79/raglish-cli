"""
대화 엔진 통합 모듈.

이 모듈은 RAG 엔진과 LLM을 활용하여 사용자 입력을 분석하고,
학습 피드백을 제공하며, 관심사 기반 대화를 유도하고 유지하는
ConversationEngine 클래스를 포함합니다.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..rag.engine import RAGEngine
from ..llm.base import LanguageModel, LanguageModelError
from ..models.conversation import (
    ConversationSession, Message, Interaction, LearningPoint
)
from ..models.response import ConversationResponse, SearchResult, LearningFeedback
from .dialog_manager import DialogManager, DialogManagerError
from .learning_assistant import LearningAssistant, LearningAssistantError
from .session_tracker import SessionTracker

logger = logging.getLogger(__name__)


class ConversationEngineError(Exception):
    """대화 엔진 관련 오류를 나타내는 예외 클래스."""
    pass


class ConversationEngine:
    """RAG 기반 영어 학습 대화 엔진.
    
    이 클래스는 RAG 엔진과 LLM을 활용하여 사용자와의 대화를 처리하고,
    영어 학습을 위한 피드백과 가이드를 제공합니다. 사용자의 관심사 문서를
    기반으로 자연스러운 대화를 유도하고 유지합니다.
    """
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        llm: LanguageModel,
        user_language: str = "korean",
        sessions_dir: str = "data/sessions"
    ):
        """대화 엔진 초기화.
        
        Args:
            rag_engine: RAG 엔진 인스턴스
            llm: 언어 모델 인스턴스
            user_language: 사용자의 모국어
            sessions_dir: 세션 데이터 저장 디렉토리
        """
        self.rag_engine = rag_engine
        self.llm = llm
        self.user_language = user_language
        
        # 대화 관련 컴포넌트 초기화
        self.dialog_manager = DialogManager(rag_engine, llm, user_language)
        self.learning_assistant = LearningAssistant(llm, user_language)
        self.session_tracker = SessionTracker(sessions_dir)
        
        # 현재 활성 세션
        self._current_session: Optional[ConversationSession] = None
        
        logger.info(f"ConversationEngine 초기화 완료 (언어: {user_language})")
    
    def start_conversation(
        self,
        preferred_topic: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ConversationSession:
        """새로운 대화 세션을 시작합니다.
        
        Args:
            preferred_topic: 선호하는 대화 주제 (선택사항)
            session_id: 기존 세션 ID (재개하는 경우)
            
        Returns:
            ConversationSession: 시작된 대화 세션
            
        Raises:
            ConversationEngineError: 대화 시작 실패 시
        """
        try:
            # 기존 세션 재개 또는 새 세션 생성
            if session_id:
                session = self.session_tracker.load_session(session_id)
                if session and session.is_active():
                    self._current_session = session
                    logger.info(f"기존 세션 재개: {session_id}")
                    return session
                else:
                    logger.warning(f"세션을 찾을 수 없거나 비활성 상태: {session_id}")
            
            # 새 세션 생성
            session = self.session_tracker.create_session(self.user_language)
            self._current_session = session
            
            # 문서 주제 추출
            document_topics = self._extract_available_topics()
            
            # 대화 시작 메시지 생성
            starter_message = self.dialog_manager.generate_conversation_starter(
                document_topics=document_topics,
                preferred_topic=preferred_topic
            )
            
            # 시작 메시지를 세션에 추가
            assistant_message = Message(
                role="assistant",
                content=starter_message,
                metadata={"type": "conversation_starter", "topics": document_topics}
            )
            session.add_message(assistant_message)
            
            logger.info(f"새로운 대화 세션 시작: {session.session_id}")
            return session
            
        except Exception as e:
            logger.error(f"대화 시작 실패: {e}")
            raise ConversationEngineError(f"대화 시작 중 오류가 발생했습니다: {e}")
    
    def process_user_input(
        self,
        user_input: str,
        session: Optional[ConversationSession] = None
    ) -> ConversationResponse:
        """사용자 입력을 처리하고 응답을 생성합니다.
        
        Args:
            user_input: 사용자의 입력 텍스트
            session: 대화 세션 (없으면 현재 활성 세션 사용)
            
        Returns:
            ConversationResponse: 생성된 응답
            
        Raises:
            ConversationEngineError: 입력 처리 실패 시
        """
        try:
            if not user_input.strip():
                raise ValueError("사용자 입력이 비어있습니다.")
            
            # 세션 확인
            if session is None:
                session = self._current_session
            
            if session is None:
                raise ValueError("활성 세션이 없습니다. 먼저 대화를 시작해주세요.")
            
            logger.debug(f"사용자 입력 처리 중: {user_input[:50]}...")
            
            # 사용자 메시지 생성 및 세션에 추가
            user_message = Message(
                role="user",
                content=user_input,
                metadata={"processing_start": datetime.now().isoformat()}
            )
            session.add_message(user_message)
            
            # 1. 영어 학습 분석 수행
            learning_feedback = self._analyze_user_english(user_input)
            
            # 2. RAG 기반 컨텍스트 검색
            context_sources = self._search_relevant_context(user_input, session)
            
            # 3. 대화 응답 생성
            response_text = self._generate_conversation_response(
                user_input, session, context_sources, learning_feedback
            )
            
            # 4. 후속 주제 제안
            suggested_topics = self._suggest_follow_up_topics(
                user_input, session, context_sources
            )
            
            # 5. 학습 포인트 추출 및 세션 업데이트
            learning_points = self._extract_learning_points(
                user_input, learning_feedback, context_sources
            )
            
            # 응답 메시지 생성 및 세션에 추가
            assistant_message = Message(
                role="assistant",
                content=response_text,
                metadata={
                    "has_feedback": learning_feedback.has_feedback() if learning_feedback else False,
                    "context_count": len(context_sources),
                    "suggested_topics_count": len(suggested_topics)
                }
            )
            session.add_message(assistant_message)
            
            # 학습 포인트와 주제를 세션에 직접 추가 (메시지는 이미 추가됨)
            for learning_point in learning_points:
                if learning_point not in session.learning_points:
                    session.learning_points.append(learning_point)
            
            # 주제들을 세션에 추가
            topics = self._extract_topics_from_context(context_sources)
            for topic in topics:
                if topic not in session.topics_covered:
                    session.topics_covered.append(topic)
            
            # 활성 세션 캐시 업데이트
            self.session_tracker._active_sessions[session.session_id] = session
            
            # 응답 객체 생성
            response = ConversationResponse(
                response_text=response_text,
                learning_feedback=learning_feedback,
                suggested_topics=suggested_topics,
                context_sources=context_sources
            )
            
            logger.info(f"사용자 입력 처리 완료 (세션: {session.session_id})")
            return response
            
        except Exception as e:
            logger.error(f"사용자 입력 처리 실패: {e}")
            raise ConversationEngineError(f"입력 처리 중 오류가 발생했습니다: {e}")
    
    def end_conversation(
        self,
        session: Optional[ConversationSession] = None
    ) -> Dict[str, Any]:
        """대화 세션을 종료하고 요약을 생성합니다.
        
        Args:
            session: 종료할 세션 (없으면 현재 활성 세션)
            
        Returns:
            Dict[str, Any]: 세션 요약 정보
            
        Raises:
            ConversationEngineError: 세션 종료 실패 시
        """
        try:
            if session is None:
                session = self._current_session
            
            if session is None:
                raise ValueError("종료할 활성 세션이 없습니다.")
            
            logger.info(f"대화 세션 종료 중: {session.session_id}")
            
            # 세션 종료 및 요약 생성
            summary = self.session_tracker.end_session(session)
            
            # 현재 세션 초기화
            if self._current_session and self._current_session.session_id == session.session_id:
                self._current_session = None
            
            # 요약 정보 반환
            result = {
                "session_id": session.session_id,
                "duration_seconds": summary.duration_seconds,
                "total_messages": summary.total_messages,
                "topics_covered": summary.topics_covered,
                "learning_points_count": len(summary.learning_points),
                "key_vocabulary": summary.key_vocabulary,
                "grammar_points": summary.grammar_points,
                "user_progress": summary.user_progress,
                "recommendations": summary.recommendations
            }
            
            logger.info(f"대화 세션 종료 완료: {session.session_id}")
            return result
            
        except Exception as e:
            logger.error(f"세션 종료 실패: {e}")
            raise ConversationEngineError(f"세션 종료 중 오류가 발생했습니다: {e}")
    
    def get_current_session(self) -> Optional[ConversationSession]:
        """현재 활성 세션을 반환합니다.
        
        Returns:
            ConversationSession: 현재 활성 세션 (없으면 None)
        """
        return self._current_session
    
    def get_session_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """최근 세션 기록을 반환합니다.
        
        Args:
            limit: 반환할 최대 세션 수
            
        Returns:
            List[Dict[str, Any]]: 세션 기록 목록
        """
        try:
            session_ids = self.session_tracker.list_all_sessions()
            recent_sessions = session_ids[-limit:] if len(session_ids) > limit else session_ids
            
            history = []
            for session_id in reversed(recent_sessions):  # 최신순으로 정렬
                summary = self.session_tracker.get_session_summary(session_id)
                if summary:
                    history.append({
                        "session_id": session_id,
                        "duration_seconds": summary.duration_seconds,
                        "total_messages": summary.total_messages,
                        "topics_covered": summary.topics_covered[:3],  # 처음 3개만
                        "created_at": summary.created_at.isoformat()
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"세션 기록 조회 실패: {e}")
            return []
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """사용자의 학습 진행 상황을 반환합니다.
        
        Returns:
            Dict[str, Any]: 학습 진행 통계
        """
        try:
            return self.session_tracker.get_user_progress_stats(self.user_language)
        except Exception as e:
            logger.error(f"학습 진행 상황 조회 실패: {e}")
            return {}
    
    def suggest_conversation_topics(self, count: int = 5) -> List[str]:
        """대화 주제를 제안합니다.
        
        Args:
            count: 제안할 주제 수
            
        Returns:
            List[str]: 제안된 주제 목록
        """
        try:
            # 문서에서 주제 추출
            document_topics = self._extract_available_topics()
            
            # 현재 세션의 기존 주제 확인
            covered_topics = []
            if self._current_session:
                covered_topics = self._current_session.topics_covered
            
            # 아직 다루지 않은 주제들 필터링
            available_topics = [
                topic for topic in document_topics 
                if topic not in covered_topics
            ]
            
            # 요청된 수만큼 반환
            return available_topics[:count]
            
        except Exception as e:
            logger.error(f"주제 제안 실패: {e}")
            return []
    
    def _extract_available_topics(self) -> List[str]:
        """사용가능한 대화 주제를 추출합니다.
        
        Returns:
            List[str]: 추출된 주제 목록
        """
        try:
            # RAG 엔진에서 인덱싱된 문서 정보 가져오기
            doc_info = self.rag_engine.get_indexed_document_info()
            
            if doc_info['total_documents'] == 0:
                logger.warning("인덱싱된 문서가 없습니다.")
                return []
            
            # 샘플 검색을 통해 주요 주제 추출
            sample_queries = ["main topic", "important", "key concept", "subject", "theme"]
            all_topics = set()
            
            for query in sample_queries:
                try:
                    results = self.rag_engine.search_similar_content(
                        query, top_k=3, min_relevance_score=0.1
                    )
                    for result in results:
                        keywords = self.rag_engine.extract_keywords(result.content, max_keywords=3)
                        all_topics.update(keywords)
                except Exception:
                    continue
            
            # 주제를 자연스러운 형태로 변환
            topics = list(all_topics)[:15]  # 최대 15개 주제
            
            logger.debug(f"추출된 주제 수: {len(topics)}")
            return topics
            
        except Exception as e:
            logger.error(f"주제 추출 실패: {e}")
            return []
    
    def _analyze_user_english(self, user_input: str) -> Optional[LearningFeedback]:
        """사용자의 영어 입력을 분석하여 학습 피드백을 생성합니다.
        
        Args:
            user_input: 사용자 입력
            
        Returns:
            LearningFeedback: 학습 피드백 (분석 실패 시 None)
        """
        try:
            # 영어 텍스트인지 간단히 확인 (영어 단어 비율)
            words = user_input.split()
            english_words = sum(1 for word in words if word.isascii() and word.isalpha())
            
            if len(words) == 0 or english_words / len(words) < 0.5:
                # 영어 비율이 낮으면 분석하지 않음
                logger.debug("영어 텍스트 비율이 낮아 분석을 건너뜁니다.")
                return None
            
            # 학습 어시스턴트를 통한 분석
            feedback = self.learning_assistant.create_learning_feedback(user_input)
            
            logger.debug(f"영어 학습 분석 완료: {len(feedback.corrections)}개 교정, "
                        f"{len(feedback.vocabulary_suggestions)}개 어휘 제안")
            
            return feedback
            
        except LearningAssistantError as e:
            logger.warning(f"영어 학습 분석 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 분석 오류: {e}")
            return None
    
    def _search_relevant_context(
        self,
        user_input: str,
        session: ConversationSession
    ) -> List[SearchResult]:
        """사용자 입력과 관련된 컨텍스트를 검색합니다.
        
        Args:
            user_input: 사용자 입력
            session: 현재 세션
            
        Returns:
            List[SearchResult]: 검색된 컨텍스트 목록
        """
        try:
            # 기본 검색
            search_results = self.rag_engine.search_similar_content(
                user_input, top_k=3, min_relevance_score=0.2
            )
            
            # 세션의 기존 주제들도 고려하여 추가 검색
            if session.topics_covered:
                for topic in session.topics_covered[-2:]:  # 최근 2개 주제만
                    topic_results = self.rag_engine.search_similar_content(
                        topic, top_k=2, min_relevance_score=0.3
                    )
                    search_results.extend(topic_results)
            
            # 중복 제거 및 관련성 점수로 정렬
            unique_results = []
            seen_content = set()
            
            for result in sorted(search_results, key=lambda x: x.relevance_score, reverse=True):
                content_hash = hash(result.content[:100])  # 처음 100자로 중복 확인
                if content_hash not in seen_content:
                    unique_results.append(result)
                    seen_content.add(content_hash)
                
                if len(unique_results) >= 5:  # 최대 5개까지
                    break
            
            logger.debug(f"컨텍스트 검색 완료: {len(unique_results)}개 결과")
            return unique_results
            
        except Exception as e:
            logger.error(f"컨텍스트 검색 실패: {e}")
            return []
    
    def _generate_conversation_response(
        self,
        user_input: str,
        session: ConversationSession,
        context_sources: List[SearchResult],
        learning_feedback: Optional[LearningFeedback]
    ) -> str:
        """대화 응답을 생성합니다.
        
        Args:
            user_input: 사용자 입력
            session: 현재 세션
            context_sources: 컨텍스트 소스
            learning_feedback: 학습 피드백
            
        Returns:
            str: 생성된 응답 텍스트
        """
        try:
            # 대화 흐름 유지 확인
            flow_response = self.dialog_manager.maintain_conversation_flow(
                session.messages, 
                session.topics_covered[-1] if session.topics_covered else None
            )
            
            # RAG 기반 답변 생성 (컨텍스트가 있는 경우)
            rag_response = ""
            if context_sources:
                try:
                    rag_response = self.rag_engine.generate_answer(
                        user_input, 
                        context_sources, 
                        self.user_language
                    )
                except Exception as e:
                    logger.warning(f"RAG 답변 생성 실패: {e}")
            
            # 학습 피드백 통합
            feedback_text = ""
            if learning_feedback and learning_feedback.has_feedback():
                feedback_text = self._format_learning_feedback(learning_feedback)
            
            # 최종 응답 구성
            response_parts = []
            
            # RAG 답변이 있으면 우선 사용
            if rag_response:
                response_parts.append(rag_response)
            else:
                # RAG 답변이 없으면 대화 흐름 응답 사용
                response_parts.append(flow_response)
            
            # 학습 피드백 추가
            if feedback_text:
                response_parts.append(feedback_text)
            
            final_response = "\n\n".join(response_parts)
            
            logger.debug(f"대화 응답 생성 완료 (길이: {len(final_response)})")
            return final_response
            
        except Exception as e:
            logger.error(f"대화 응답 생성 실패: {e}")
            # 폴백 응답
            return self._get_fallback_response()
    
    def _suggest_follow_up_topics(
        self,
        user_input: str,
        session: ConversationSession,
        context_sources: List[SearchResult]
    ) -> List[str]:
        """후속 대화 주제를 제안합니다.
        
        Args:
            user_input: 사용자 입력
            session: 현재 세션
            context_sources: 컨텍스트 소스
            
        Returns:
            List[str]: 제안된 주제 목록
        """
        try:
            # 대화 관리자를 통한 후속 질문 생성
            follow_up_questions = self.dialog_manager.suggest_follow_up_questions(
                user_input, session.messages, max_suggestions=3
            )
            
            # 컨텍스트에서 추가 주제 추출
            context_topics = []
            for source in context_sources:
                keywords = self.rag_engine.extract_keywords(source.content, max_keywords=2)
                context_topics.extend(keywords)
            
            # 중복 제거 및 기존 주제와 다른 것들만 선택
            all_suggestions = follow_up_questions + context_topics
            covered_topics_lower = [topic.lower() for topic in session.topics_covered]
            
            unique_suggestions = []
            for suggestion in all_suggestions:
                if (suggestion.lower() not in covered_topics_lower and 
                    suggestion not in unique_suggestions):
                    unique_suggestions.append(suggestion)
                
                if len(unique_suggestions) >= 5:  # 최대 5개
                    break
            
            return unique_suggestions
            
        except Exception as e:
            logger.error(f"후속 주제 제안 실패: {e}")
            return []
    
    def _extract_learning_points(
        self,
        user_input: str,
        learning_feedback: Optional[LearningFeedback],
        context_sources: List[SearchResult]
    ) -> List[LearningPoint]:
        """상호작용에서 학습 포인트를 추출합니다.
        
        Args:
            user_input: 사용자 입력
            learning_feedback: 학습 피드백
            context_sources: 컨텍스트 소스
            
        Returns:
            List[LearningPoint]: 추출된 학습 포인트 목록
        """
        learning_points = []
        
        try:
            # 학습 피드백에서 학습 포인트 추출
            if learning_feedback:
                # 문법 교정에서 학습 포인트 생성
                for correction in learning_feedback.corrections:
                    learning_point = LearningPoint(
                        topic=f"Grammar: {correction.error_type}",
                        description=correction.explanation,
                        example=f"{correction.original_text} → {correction.corrected_text}",
                        difficulty_level="intermediate"
                    )
                    learning_points.append(learning_point)
                
                # 어휘 제안에서 학습 포인트 생성
                for vocab in learning_feedback.vocabulary_suggestions:
                    learning_point = LearningPoint(
                        topic=f"Vocabulary: {vocab.word}",
                        description=vocab.definition,
                        example=vocab.usage_example,
                        difficulty_level=vocab.difficulty_level
                    )
                    learning_points.append(learning_point)
            
            # 컨텍스트에서 주제 관련 학습 포인트 추출
            if context_sources:
                main_topics = set()
                for source in context_sources:
                    keywords = self.rag_engine.extract_keywords(source.content, max_keywords=2)
                    main_topics.update(keywords)
                
                for topic in list(main_topics)[:2]:  # 최대 2개 주제
                    learning_point = LearningPoint(
                        topic=f"Topic: {topic}",
                        description=f"Discussed topic related to {topic}",
                        example=user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        difficulty_level="intermediate"
                    )
                    learning_points.append(learning_point)
            
            # 최대 5개까지만 반환
            return learning_points[:5]
            
        except Exception as e:
            logger.error(f"학습 포인트 추출 실패: {e}")
            return []
    
    def _extract_topics_from_context(self, context_sources: List[SearchResult]) -> List[str]:
        """컨텍스트 소스에서 주제를 추출합니다.
        
        Args:
            context_sources: 컨텍스트 소스 목록
            
        Returns:
            List[str]: 추출된 주제 목록
        """
        topics = set()
        
        try:
            for source in context_sources:
                # 메타데이터에서 주제 정보 확인
                if 'topics' in source.metadata:
                    topics.update(source.metadata['topics'])
                
                # 컨텐츠에서 키워드 추출
                keywords = self.rag_engine.extract_keywords(source.content, max_keywords=3)
                topics.update(keywords)
            
            return list(topics)[:10]  # 최대 10개
            
        except Exception as e:
            logger.error(f"주제 추출 실패: {e}")
            return []
    
    def _format_learning_feedback(self, feedback: LearningFeedback) -> str:
        """학습 피드백을 텍스트로 포맷팅합니다.
        
        Args:
            feedback: 학습 피드백
            
        Returns:
            str: 포맷팅된 피드백 텍스트
        """
        parts = []
        
        if self.user_language == "korean":
            if feedback.corrections:
                parts.append("📝 **문법 교정:**")
                for correction in feedback.corrections[:3]:  # 최대 3개
                    parts.append(f"• {correction.original_text} → {correction.corrected_text}")
                    parts.append(f"  {correction.explanation}")
            
            if feedback.vocabulary_suggestions:
                parts.append("\n📚 **어휘 제안:**")
                for vocab in feedback.vocabulary_suggestions[:2]:  # 최대 2개
                    parts.append(f"• {vocab.word}: {vocab.definition}")
            
            if feedback.encouragement:
                parts.append(f"\n💪 {feedback.encouragement}")
        else:
            if feedback.corrections:
                parts.append("📝 **Grammar Corrections:**")
                for correction in feedback.corrections[:3]:
                    parts.append(f"• {correction.original_text} → {correction.corrected_text}")
                    parts.append(f"  {correction.explanation}")
            
            if feedback.vocabulary_suggestions:
                parts.append("\n📚 **Vocabulary Suggestions:**")
                for vocab in feedback.vocabulary_suggestions[:2]:
                    parts.append(f"• {vocab.word}: {vocab.definition}")
            
            if feedback.encouragement:
                parts.append(f"\n💪 {feedback.encouragement}")
        
        return "\n".join(parts)
    
    def _get_fallback_response(self) -> str:
        """폴백 응답을 반환합니다.
        
        Returns:
            str: 폴백 응답 텍스트
        """
        if self.user_language == "korean":
            return "죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다. 다시 시도해주세요."
        else:
            return "I'm sorry, there was an issue generating a response. Please try again."