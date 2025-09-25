"""
대화 관리자 모듈.

이 모듈은 문서 주제 기반 대화 시작, 대화 흐름 유지, 후속 질문 제안 등의 
기능을 제공하는 DialogManager 클래스를 포함합니다.
"""

import logging
import random
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..models.conversation import Message, ConversationSession
from ..models.response import SearchResult
from ..rag.engine import RAGEngine
from ..llm.base import LanguageModel

logger = logging.getLogger(__name__)


class DialogManagerError(Exception):
    """대화 관리자 관련 오류를 나타내는 예외 클래스."""
    pass


class DialogManager:
    """문서 기반 대화를 관리하는 클래스.
    
    이 클래스는 사용자의 관심사 문서를 기반으로 자연스러운 대화를 시작하고,
    대화 흐름을 유지하며, 적절한 후속 질문을 제안하는 기능을 제공합니다.
    """
    
    def __init__(
        self,
        rag_engine: RAGEngine,
        llm: LanguageModel,
        user_language: str = "korean"
    ):
        """대화 관리자 초기화.
        
        Args:
            rag_engine: RAG 엔진 인스턴스
            llm: 언어 모델 인스턴스
            user_language: 사용자의 모국어
        """
        self.rag_engine = rag_engine
        self.llm = llm
        self.user_language = user_language
        
        # 대화 시작 템플릿들
        self._conversation_starters = self._load_conversation_starters()
        
        # 후속 질문 패턴들
        self._follow_up_patterns = self._load_follow_up_patterns()
        
        logger.info(f"DialogManager 초기화 완료 (언어: {user_language})")
    
    def generate_conversation_starter(
        self,
        document_topics: Optional[List[str]] = None,
        preferred_topic: Optional[str] = None
    ) -> str:
        """문서 주제를 기반으로 대화 시작 메시지를 생성합니다.
        
        Args:
            document_topics: 문서에서 추출된 주제 목록
            preferred_topic: 선호하는 특정 주제
            
        Returns:
            str: 생성된 대화 시작 메시지
            
        Raises:
            DialogManagerError: 대화 시작 메시지 생성 실패 시
        """
        try:
            logger.info("대화 시작 메시지 생성 중...")
            
            # 주제가 제공되지 않은 경우 RAG에서 추출
            if not document_topics:
                document_topics = self._extract_topics_from_documents()
            
            if not document_topics:
                # 기본 대화 시작 메시지 반환
                return self._get_default_conversation_starter()
            
            # 선호 주제가 있으면 우선 사용
            selected_topic = preferred_topic if preferred_topic in document_topics else None
            
            if not selected_topic:
                # 랜덤하게 주제 선택
                selected_topic = random.choice(document_topics)
            
            # 주제 기반 대화 시작 메시지 생성
            starter_message = self._generate_topic_based_starter(selected_topic, document_topics)
            
            logger.info(f"대화 시작 메시지 생성 완료 (주제: {selected_topic})")
            return starter_message
            
        except Exception as e:
            logger.error(f"대화 시작 메시지 생성 실패: {e}")
            raise DialogManagerError(f"대화 시작 메시지 생성 중 오류가 발생했습니다: {e}")
    
    def maintain_conversation_flow(
        self,
        conversation_history: List[Message],
        current_topic: Optional[str] = None
    ) -> str:
        """대화 흐름을 유지하고 자연스러운 전환을 제공합니다.
        
        Args:
            conversation_history: 현재까지의 대화 기록
            current_topic: 현재 대화 주제
            
        Returns:
            str: 대화 흐름 유지를 위한 응답 또는 전환 메시지
            
        Raises:
            DialogManagerError: 대화 흐름 유지 실패 시
        """
        try:
            if not conversation_history:
                return self.generate_conversation_starter()
            
            logger.debug(f"대화 흐름 분석 중 (메시지 수: {len(conversation_history)})")
            
            # 최근 대화 분석
            recent_messages = conversation_history[-5:]  # 최근 5개 메시지만 분석
            
            # 대화 패턴 분석
            conversation_analysis = self._analyze_conversation_pattern(recent_messages)
            
            # 대화 흐름에 따른 응답 생성
            if conversation_analysis['needs_topic_change']:
                return self._suggest_topic_transition(conversation_analysis, current_topic)
            elif conversation_analysis['needs_encouragement']:
                return self._generate_encouragement_message(conversation_analysis)
            elif conversation_analysis['needs_clarification']:
                return self._generate_clarification_request(conversation_analysis)
            else:
                return self._generate_natural_continuation(conversation_analysis)
            
        except Exception as e:
            logger.error(f"대화 흐름 유지 실패: {e}")
            raise DialogManagerError(f"대화 흐름 유지 중 오류가 발생했습니다: {e}")
    
    def suggest_follow_up_questions(
        self,
        context: str,
        conversation_history: Optional[List[Message]] = None,
        max_suggestions: int = 3
    ) -> List[str]:
        """현재 컨텍스트를 기반으로 후속 질문을 제안합니다.
        
        Args:
            context: 현재 대화 컨텍스트
            conversation_history: 대화 기록 (선택사항)
            max_suggestions: 최대 제안 수
            
        Returns:
            List[str]: 제안된 후속 질문 목록
            
        Raises:
            DialogManagerError: 후속 질문 제안 실패 시
        """
        try:
            logger.debug(f"후속 질문 제안 생성 중 (컨텍스트 길이: {len(context)})")
            
            # 컨텍스트에서 키워드 추출
            keywords = self.rag_engine.extract_keywords(context, max_keywords=5)
            
            # 관련 문서 검색
            related_content = self.rag_engine.search_similar_content(
                context, 
                top_k=3,
                min_relevance_score=0.3
            )
            
            # LLM을 통한 후속 질문 생성
            follow_up_questions = self._generate_llm_follow_up_questions(
                context, keywords, related_content, conversation_history
            )
            
            # 패턴 기반 후속 질문 추가
            pattern_questions = self._generate_pattern_based_questions(keywords, context)
            
            # 결합 및 중복 제거
            all_questions = follow_up_questions + pattern_questions
            unique_questions = self._remove_duplicate_questions(all_questions)
            
            # 최대 개수만큼 반환
            result = unique_questions[:max_suggestions]
            
            logger.info(f"후속 질문 {len(result)}개 생성 완료")
            return result
            
        except Exception as e:
            logger.error(f"후속 질문 제안 실패: {e}")
            raise DialogManagerError(f"후속 질문 제안 중 오류가 발생했습니다: {e}")
    
    def detect_topic_change_opportunity(
        self,
        conversation_history: List[Message],
        min_messages_per_topic: int = 5
    ) -> Tuple[bool, Optional[str]]:
        """주제 변경 기회를 감지합니다.
        
        Args:
            conversation_history: 대화 기록
            min_messages_per_topic: 주제당 최소 메시지 수
            
        Returns:
            Tuple[bool, Optional[str]]: (주제 변경 필요 여부, 제안할 새 주제)
        """
        try:
            if len(conversation_history) < min_messages_per_topic:
                return False, None
            
            # 최근 대화의 반복성 분석
            recent_messages = conversation_history[-min_messages_per_topic:]
            
            # 키워드 중복도 계산
            all_keywords = []
            for message in recent_messages:
                keywords = self.rag_engine.extract_keywords(message.content, max_keywords=3)
                all_keywords.extend(keywords)
            
            # 중복도가 높으면 주제 변경 제안
            unique_keywords = set(all_keywords)
            repetition_ratio = 1 - (len(unique_keywords) / len(all_keywords)) if all_keywords else 0
            
            if repetition_ratio > 0.7:  # 70% 이상 중복시 주제 변경
                # 새로운 주제 제안
                current_keywords = list(unique_keywords)
                new_topic = self._suggest_new_topic(current_keywords)
                return True, new_topic
            
            return False, None
            
        except Exception as e:
            logger.error(f"주제 변경 기회 감지 실패: {e}")
            return False, None
    
    def _extract_topics_from_documents(self) -> List[str]:
        """인덱싱된 문서들에서 주요 주제를 추출합니다.
        
        Returns:
            List[str]: 추출된 주제 목록
        """
        try:
            # RAG 엔진에서 문서 정보 가져오기
            doc_info = self.rag_engine.get_indexed_document_info()
            
            if doc_info['total_documents'] == 0:
                return []
            
            # 샘플 검색을 통해 주요 키워드 추출
            sample_queries = ["", "important", "main", "key", "topic"]
            all_keywords = set()
            
            for query in sample_queries:
                try:
                    results = self.rag_engine.search_similar_content(
                        query, top_k=5, min_relevance_score=0.1
                    )
                    for result in results:
                        keywords = self.rag_engine.extract_keywords(result.content, max_keywords=3)
                        all_keywords.update(keywords)
                except:
                    continue
            
            # 주제로 변환 (키워드를 더 자연스러운 주제명으로)
            topics = list(all_keywords)[:10]  # 최대 10개 주제
            
            logger.debug(f"문서에서 {len(topics)}개 주제 추출")
            return topics
            
        except Exception as e:
            logger.error(f"문서 주제 추출 실패: {e}")
            return []
    
    def _get_default_conversation_starter(self) -> str:
        """기본 대화 시작 메시지를 반환합니다.
        
        Returns:
            str: 기본 대화 시작 메시지
        """
        if self.user_language == "korean":
            starters = [
                "안녕하세요! 오늘은 어떤 주제에 대해 영어로 이야기해볼까요?",
                "영어 학습을 시작해볼까요? 관심 있는 주제가 있으시면 말씀해주세요!",
                "좋은 하루입니다! 오늘은 어떤 내용으로 영어 대화를 나눠볼까요?",
                "영어 연습 시간입니다! 어떤 주제로 시작해보고 싶으신가요?"
            ]
        else:
            starters = [
                "Hello! What topic would you like to discuss in English today?",
                "Let's start our English learning session! Do you have any topic in mind?",
                "Good day! What would you like to talk about in English today?",
                "Time for English practice! What topic interests you today?"
            ]
        
        return random.choice(starters)
    
    def _generate_topic_based_starter(self, topic: str, all_topics: List[str]) -> str:
        """주제 기반 대화 시작 메시지를 생성합니다.
        
        Args:
            topic: 선택된 주제
            all_topics: 모든 가능한 주제 목록
            
        Returns:
            str: 주제 기반 대화 시작 메시지
        """
        try:
            # 주제 관련 컨텍스트 검색
            topic_context = self.rag_engine.search_similar_content(
                topic, top_k=2, min_relevance_score=0.3
            )
            
            # LLM을 통한 자연스러운 대화 시작 메시지 생성
            prompt = self._create_starter_prompt(topic, topic_context, all_topics)
            response = self.llm.generate_response(prompt)
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"주제 기반 시작 메시지 생성 실패: {e}")
            # 폴백: 템플릿 기반 메시지
            return self._get_template_based_starter(topic)
    
    def _create_starter_prompt(
        self, 
        topic: str, 
        context: List[SearchResult], 
        all_topics: List[str]
    ) -> str:
        """대화 시작을 위한 프롬프트를 생성합니다.
        
        Args:
            topic: 주제
            context: 관련 컨텍스트
            all_topics: 모든 주제 목록
            
        Returns:
            str: 생성된 프롬프트
        """
        context_text = "\n".join([result.content[:200] + "..." for result in context])
        
        if self.user_language == "korean":
            return f"""다음 주제에 대해 영어 학습자와 자연스러운 대화를 시작하는 메시지를 작성해주세요.

주제: {topic}
관련 내용: {context_text}
다른 가능한 주제들: {', '.join(all_topics[:5])}

요구사항:
1. 친근하고 격려적인 톤으로 작성
2. 영어 학습자가 부담스럽지 않게 대화에 참여할 수 있도록 유도
3. 구체적인 질문이나 의견을 물어보기
4. 한국어로 작성하되, 영어 단어나 표현을 자연스럽게 포함
5. 2-3문장으로 간결하게 작성

대화 시작 메시지:"""
        else:
            return f"""Create a natural conversation starter message for an English learner about the following topic.

Topic: {topic}
Related content: {context_text}
Other possible topics: {', '.join(all_topics[:5])}

Requirements:
1. Use a friendly and encouraging tone
2. Make it easy for the learner to join the conversation
3. Ask specific questions or opinions
4. Write in English with simple, clear language
5. Keep it concise (2-3 sentences)

Conversation starter:"""
    
    def _get_template_based_starter(self, topic: str) -> str:
        """템플릿 기반 대화 시작 메시지를 생성합니다.
        
        Args:
            topic: 주제
            
        Returns:
            str: 템플릿 기반 메시지
        """
        if self.user_language == "korean":
            templates = [
                f"오늘은 '{topic}'에 대해 영어로 이야기해볼까요? 이 주제에 대해 어떻게 생각하시나요?",
                f"'{topic}'라는 흥미로운 주제가 있네요! 이것에 대한 당신의 경험이나 의견을 영어로 들려주세요.",
                f"'{topic}' 관련해서 영어 대화를 시작해보죠. 이 주제에 대해 무엇이 가장 궁금하신가요?",
                f"오늘의 주제는 '{topic}'입니다. 이것에 대해 영어로 자유롭게 이야기해보세요!"
            ]
        else:
            templates = [
                f"Let's talk about '{topic}' in English today! What do you think about this topic?",
                f"Here's an interesting topic: '{topic}'! Please share your experience or opinion about it in English.",
                f"Let's start an English conversation about '{topic}'. What interests you most about this topic?",
                f"Today's topic is '{topic}'. Feel free to share your thoughts about it in English!"
            ]
        
        return random.choice(templates)
    
    def _analyze_conversation_pattern(self, messages: List[Message]) -> Dict[str, Any]:
        """대화 패턴을 분석합니다.
        
        Args:
            messages: 분석할 메시지 목록
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        analysis = {
            'needs_topic_change': False,
            'needs_encouragement': False,
            'needs_clarification': False,
            'user_engagement_level': 'medium',
            'conversation_length': len(messages),
            'last_user_message': None,
            'dominant_keywords': []
        }
        
        if not messages:
            return analysis
        
        # 사용자 메시지만 필터링
        user_messages = [msg for msg in messages if msg.role == 'user']
        
        if user_messages:
            analysis['last_user_message'] = user_messages[-1].content
            
            # 사용자 참여도 분석
            avg_length = sum(len(msg.content.split()) for msg in user_messages) / len(user_messages)
            
            if avg_length < 3:
                analysis['user_engagement_level'] = 'low'
                analysis['needs_encouragement'] = True
            elif avg_length > 15:
                analysis['user_engagement_level'] = 'high'
            
            # 최근 메시지가 너무 짧거나 반복적인 경우
            if len(user_messages) >= 3:
                recent_contents = [msg.content.lower() for msg in user_messages[-3:]]
                if all(len(content.split()) <= 2 for content in recent_contents):
                    analysis['needs_encouragement'] = True
        
        # 주제 변경 필요성 판단
        if len(messages) > 10:
            try:
                # 키워드 추출 및 반복성 확인
                all_content = " ".join([msg.content for msg in messages[-6:]])
                keywords = self.rag_engine.extract_keywords(all_content, max_keywords=5)
                analysis['dominant_keywords'] = keywords
                
                # 키워드가 너무 적으면 주제 변경 필요
                if len(set(keywords)) < 3:
                    analysis['needs_topic_change'] = True
            except Exception as e:
                logger.error(f"키워드 추출 실패: {e}")
                # 키워드 추출 실패 시 기본값 사용
                analysis['dominant_keywords'] = []
        
        return analysis
    
    def _suggest_topic_transition(self, analysis: Dict[str, Any], current_topic: Optional[str]) -> str:
        """주제 전환을 제안합니다.
        
        Args:
            analysis: 대화 분석 결과
            current_topic: 현재 주제
            
        Returns:
            str: 주제 전환 메시지
        """
        # 새로운 주제 제안
        new_topic = self._suggest_new_topic(analysis.get('dominant_keywords', []))
        
        if self.user_language == "korean":
            transitions = [
                f"이 주제에 대해 많이 이야기했네요! 이제 '{new_topic}'에 대해서도 영어로 대화해볼까요?",
                f"새로운 주제로 넘어가볼까요? '{new_topic}'에 대한 당신의 생각이 궁금합니다.",
                f"다른 흥미로운 주제인 '{new_topic}'에 대해서도 이야기해보면 어떨까요?",
                f"주제를 바꿔서 '{new_topic}'에 대해 영어로 대화해보시겠어요?"
            ]
        else:
            transitions = [
                f"We've talked a lot about this topic! How about discussing '{new_topic}' in English now?",
                f"Let's move to a new topic. What do you think about '{new_topic}'?",
                f"How about talking about another interesting topic: '{new_topic}'?",
                f"Shall we switch topics and discuss '{new_topic}' in English?"
            ]
        
        return random.choice(transitions)
    
    def _generate_encouragement_message(self, analysis: Dict[str, Any]) -> str:
        """격려 메시지를 생성합니다.
        
        Args:
            analysis: 대화 분석 결과
            
        Returns:
            str: 격려 메시지
        """
        if self.user_language == "korean":
            encouragements = [
                "좋아요! 더 자세히 설명해주실 수 있나요? 당신의 의견이 궁금합니다.",
                "훌륭합니다! 조금 더 길게 영어로 표현해보세요. 어떤 예시가 있을까요?",
                "잘하고 있어요! 그것에 대해 더 많은 생각이나 경험을 영어로 나눠주세요.",
                "멋져요! 그 주제에 대해 더 구체적으로 영어로 이야기해볼까요?"
            ]
        else:
            encouragements = [
                "Great! Can you tell me more about that? I'd love to hear your thoughts.",
                "Excellent! Try to express that in more detail in English. What examples come to mind?",
                "You're doing well! Please share more of your thoughts or experiences about that in English.",
                "Wonderful! Let's talk more specifically about that topic in English."
            ]
        
        return random.choice(encouragements)
    
    def _generate_clarification_request(self, analysis: Dict[str, Any]) -> str:
        """명확화 요청 메시지를 생성합니다.
        
        Args:
            analysis: 대화 분석 결과
            
        Returns:
            str: 명확화 요청 메시지
        """
        if self.user_language == "korean":
            clarifications = [
                "흥미롭네요! 그것에 대해 좀 더 자세히 설명해주실 수 있나요?",
                "그 부분이 궁금합니다. 영어로 더 구체적으로 말씀해주세요.",
                "좋은 포인트네요! 그것에 대한 예시나 경험이 있으시면 영어로 들려주세요.",
                "그 의견에 대해 더 알고 싶습니다. 왜 그렇게 생각하시는지 영어로 설명해주세요."
            ]
        else:
            clarifications = [
                "That's interesting! Could you explain that in more detail?",
                "I'm curious about that part. Please tell me more specifically in English.",
                "Good point! If you have any examples or experiences about that, please share in English.",
                "I'd like to know more about that opinion. Could you explain why you think so in English?"
            ]
        
        return random.choice(clarifications)
    
    def _generate_natural_continuation(self, analysis: Dict[str, Any]) -> str:
        """자연스러운 대화 지속 메시지를 생성합니다.
        
        Args:
            analysis: 대화 분석 결과
            
        Returns:
            str: 대화 지속 메시지
        """
        if self.user_language == "korean":
            continuations = [
                "그렇군요! 그것과 관련해서 다른 경험도 있으신가요?",
                "흥미로운 관점이네요. 그것에 대해 어떻게 더 발전시킬 수 있을까요?",
                "좋은 생각입니다! 그런 상황에서 어떻게 대처하셨나요?",
                "이해했습니다. 그것이 당신에게 어떤 의미인지 영어로 말씀해주세요."
            ]
        else:
            continuations = [
                "I see! Do you have any other experiences related to that?",
                "That's an interesting perspective. How do you think we could develop that further?",
                "Good thinking! How did you handle that kind of situation?",
                "I understand. Could you tell me what that means to you in English?"
            ]
        
        return random.choice(continuations)
    
    def _suggest_new_topic(self, current_keywords: List[str]) -> str:
        """현재 키워드를 기반으로 새로운 주제를 제안합니다.
        
        Args:
            current_keywords: 현재 대화의 키워드들
            
        Returns:
            str: 제안된 새 주제
        """
        try:
            # 현재 키워드와 다른 주제 검색
            all_topics = self._extract_topics_from_documents()
            
            # 현재 키워드와 겹치지 않는 주제 찾기
            available_topics = []
            for topic in all_topics:
                topic_lower = topic.lower()
                if not any(keyword.lower() in topic_lower for keyword in current_keywords):
                    available_topics.append(topic)
            
            if available_topics:
                return random.choice(available_topics)
            else:
                # 폴백: 일반적인 주제들
                fallback_topics = [
                    "technology", "travel", "food", "culture", "education", 
                    "environment", "health", "sports", "music", "books"
                ]
                return random.choice(fallback_topics)
                
        except Exception as e:
            logger.error(f"새 주제 제안 실패: {e}")
            return "daily life"
    
    def _generate_llm_follow_up_questions(
        self,
        context: str,
        keywords: List[str],
        related_content: List[SearchResult],
        conversation_history: Optional[List[Message]] = None
    ) -> List[str]:
        """LLM을 통해 후속 질문을 생성합니다.
        
        Args:
            context: 현재 컨텍스트
            keywords: 추출된 키워드
            related_content: 관련 컨텐츠
            conversation_history: 대화 기록
            
        Returns:
            List[str]: 생성된 후속 질문 목록
        """
        try:
            # 프롬프트 생성
            prompt = self._create_follow_up_prompt(context, keywords, related_content, conversation_history)
            
            # LLM 응답 생성
            response = self.llm.generate_response(prompt)
            
            # 응답에서 질문들 추출
            questions = self._parse_follow_up_response(response.content)
            
            return questions[:3]  # 최대 3개
            
        except Exception as e:
            logger.error(f"LLM 후속 질문 생성 실패: {e}")
            return []
    
    def _create_follow_up_prompt(
        self,
        context: str,
        keywords: List[str],
        related_content: List[SearchResult],
        conversation_history: Optional[List[Message]] = None
    ) -> str:
        """후속 질문 생성을 위한 프롬프트를 생성합니다.
        
        Args:
            context: 현재 컨텍스트
            keywords: 키워드 목록
            related_content: 관련 컨텐츠
            conversation_history: 대화 기록
            
        Returns:
            str: 생성된 프롬프트
        """
        related_text = "\n".join([result.content[:100] + "..." for result in related_content[:2]])
        
        if self.user_language == "korean":
            return f"""다음 대화 컨텍스트를 바탕으로 영어 학습자에게 적절한 후속 질문 3개를 생성해주세요.

현재 컨텍스트: {context}
주요 키워드: {', '.join(keywords)}
관련 내용: {related_text}

요구사항:
1. 영어 학습자가 답변하기 쉬운 수준의 질문
2. 현재 주제와 연관성이 있는 질문
3. 학습자의 의견이나 경험을 물어보는 질문
4. 각 질문은 한 줄로 작성
5. 질문 앞에 번호를 붙여서 작성 (1. 2. 3.)

후속 질문들:"""
        else:
            return f"""Generate 3 appropriate follow-up questions for an English learner based on the following conversation context.

Current context: {context}
Key keywords: {', '.join(keywords)}
Related content: {related_text}

Requirements:
1. Questions should be at an appropriate level for English learners
2. Questions should be related to the current topic
3. Questions should ask for opinions or experiences
4. Write each question on one line
5. Number the questions (1. 2. 3.)

Follow-up questions:"""
    
    def _parse_follow_up_response(self, response: str) -> List[str]:
        """LLM 응답에서 후속 질문들을 파싱합니다.
        
        Args:
            response: LLM 응답
            
        Returns:
            List[str]: 파싱된 질문 목록
        """
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # 번호나 불릿 포인트가 있는 줄 찾기
            if line and (line[0].isdigit() or line.startswith(('-', '•', '*'))):
                # 번호나 불릿 포인트 제거
                question = line
                prefixes = ['1.', '2.', '3.', '4.', '5.', '-', '•', '*']
                for prefix in prefixes:
                    if question.startswith(prefix):
                        question = question[len(prefix):].strip()
                        break
                
                if question and question.endswith('?'):
                    questions.append(question)
        
        return questions
    
    def _generate_pattern_based_questions(self, keywords: List[str], context: str) -> List[str]:
        """패턴 기반 후속 질문을 생성합니다.
        
        Args:
            keywords: 키워드 목록
            context: 컨텍스트
            
        Returns:
            List[str]: 패턴 기반 질문 목록
        """
        questions = []
        
        if not keywords:
            return questions
        
        # 키워드 기반 질문 패턴
        if self.user_language == "korean":
            patterns = [
                f"'{keywords[0]}'에 대한 당신의 경험은 어떤가요?",
                f"'{keywords[0]}'와 관련해서 어떤 것이 가장 중요하다고 생각하시나요?",
                f"'{keywords[0]}'에 대해 다른 사람들과 어떻게 다른 의견을 가지고 계신가요?"
            ]
        else:
            patterns = [
                f"What's your experience with '{keywords[0]}'?",
                f"What do you think is most important about '{keywords[0]}'?",
                f"How do your opinions about '{keywords[0]}' differ from others?"
            ]
        
        # 키워드가 여러 개인 경우 추가 패턴
        if len(keywords) > 1:
            if self.user_language == "korean":
                patterns.append(f"'{keywords[0]}'와 '{keywords[1]}' 중에서 어느 것이 더 중요하다고 생각하시나요?")
            else:
                patterns.append(f"Between '{keywords[0]}' and '{keywords[1]}', which do you think is more important?")
        
        return patterns[:2]  # 최대 2개
    
    def _remove_duplicate_questions(self, questions: List[str]) -> List[str]:
        """중복된 질문을 제거합니다.
        
        Args:
            questions: 질문 목록
            
        Returns:
            List[str]: 중복이 제거된 질문 목록
        """
        unique_questions = []
        seen_questions = set()
        
        for question in questions:
            # 질문을 정규화 (소문자, 공백 정리)
            normalized = question.lower().strip()
            
            # 유사한 질문 체크 (간단한 키워드 기반)
            is_duplicate = False
            for seen in seen_questions:
                # 공통 단어가 70% 이상이면 중복으로 간주
                question_words = set(normalized.split())
                seen_words = set(seen.split())
                
                if question_words and seen_words:
                    common_ratio = len(question_words & seen_words) / len(question_words | seen_words)
                    if common_ratio > 0.7:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_questions.append(question)
                seen_questions.add(normalized)
        
        return unique_questions
    
    def _load_conversation_starters(self) -> Dict[str, List[str]]:
        """대화 시작 템플릿을 로드합니다.
        
        Returns:
            Dict[str, List[str]]: 언어별 대화 시작 템플릿
        """
        return {
            "korean": [
                "안녕하세요! 오늘은 {topic}에 대해 영어로 이야기해볼까요?",
                "{topic}라는 흥미로운 주제가 있네요! 어떻게 생각하시나요?",
                "오늘의 주제는 {topic}입니다. 이것에 대한 당신의 경험을 들려주세요.",
                "{topic}에 대해 영어로 대화해보시겠어요? 어떤 것이 가장 궁금하신가요?"
            ],
            "english": [
                "Hello! Let's talk about {topic} in English today!",
                "Here's an interesting topic: {topic}! What do you think about it?",
                "Today's topic is {topic}. Please share your experience about it.",
                "Shall we discuss {topic} in English? What interests you most about it?"
            ]
        }
    
    def _load_follow_up_patterns(self) -> Dict[str, List[str]]:
        """후속 질문 패턴을 로드합니다.
        
        Returns:
            Dict[str, List[str]]: 언어별 후속 질문 패턴
        """
        return {
            "korean": [
                "그것에 대해 더 자세히 말씀해주실 수 있나요?",
                "그런 경험이 있으셨군요! 어떤 느낌이었나요?",
                "흥미롭네요. 다른 예시도 있을까요?",
                "그것이 당신에게 어떤 의미인가요?",
                "그 상황에서 어떻게 대처하셨나요?"
            ],
            "english": [
                "Could you tell me more about that?",
                "You had that experience! How did it feel?",
                "That's interesting. Are there other examples?",
                "What does that mean to you?",
                "How did you handle that situation?"
            ]
        }