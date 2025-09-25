"""
대화 세션 추적 및 관리 모듈.

이 모듈은 영어 학습 대화 세션의 생성, 업데이트, 저장 및 요약 기능을 제공합니다.
세션 데이터는 JSON 형식으로 로컬 파일 시스템에 저장됩니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..models import (
    ConversationSession,
    ConversationSummary,
    Interaction,
    LearningPoint,
    Message
)


logger = logging.getLogger(__name__)


class SessionTracker:
    """대화 세션을 추적하고 관리하는 클래스.
    
    이 클래스는 대화 세션의 생성, 업데이트, 저장 및 요약 기능을 제공합니다.
    세션 데이터는 JSON 형식으로 로컬 파일 시스템에 저장되며,
    학습 포인트와 진행 상황을 추적합니다.
    """
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        """SessionTracker를 초기화합니다.
        
        Args:
            sessions_dir: 세션 데이터를 저장할 디렉토리 경로
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # 활성 세션들을 메모리에 캐시
        self._active_sessions: Dict[str, ConversationSession] = {}
        
        logger.info(f"SessionTracker initialized with sessions directory: {self.sessions_dir}")
    
    def create_session(self, user_language: str = "korean") -> ConversationSession:
        """새로운 대화 세션을 생성합니다.
        
        Args:
            user_language: 사용자의 모국어 (기본값: "korean")
            
        Returns:
            ConversationSession: 새로 생성된 대화 세션
        """
        session = ConversationSession(user_language=user_language)
        self._active_sessions[session.session_id] = session
        
        logger.info(f"Created new session: {session.session_id}")
        return session
    
    def update_session(self, session: ConversationSession, interaction: Interaction) -> None:
        """세션에 새로운 상호작용을 추가하여 업데이트합니다.
        
        Args:
            session: 업데이트할 대화 세션
            interaction: 추가할 상호작용 (사용자 입력 + 어시스턴트 응답)
        """
        # 메시지들을 세션에 추가
        session.add_message(interaction.user_message)
        session.add_message(interaction.assistant_message)
        
        # 학습 포인트들을 세션에 추가
        for learning_point in interaction.learning_points:
            if learning_point not in session.learning_points:
                session.learning_points.append(learning_point)
        
        # 주제들을 세션에 추가
        for topic in interaction.topics:
            if topic not in session.topics_covered:
                session.topics_covered.append(topic)
        
        # 활성 세션 캐시 업데이트
        self._active_sessions[session.session_id] = session
        
        logger.debug(f"Updated session {session.session_id} with new interaction")
    
    def save_session(self, session: ConversationSession) -> None:
        """세션을 파일 시스템에 저장합니다.
        
        Args:
            session: 저장할 대화 세션
            
        Raises:
            IOError: 파일 저장 중 오류가 발생한 경우
        """
        try:
            session_file = self.sessions_dir / f"{session.session_id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                f.write(session.to_json())
            
            logger.info(f"Saved session {session.session_id} to {session_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            raise IOError(f"Failed to save session: {e}")
    
    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """저장된 세션을 로드합니다.
        
        Args:
            session_id: 로드할 세션의 ID
            
        Returns:
            ConversationSession: 로드된 세션, 없으면 None
        """
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            
            if not session_file.exists():
                logger.warning(f"Session file not found: {session_file}")
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = f.read()
            
            session = ConversationSession.from_json(session_data)
            logger.info(f"Loaded session {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def end_session(self, session: ConversationSession) -> ConversationSummary:
        """세션을 종료하고 요약을 생성합니다.
        
        Args:
            session: 종료할 대화 세션
            
        Returns:
            ConversationSummary: 생성된 세션 요약
        """
        # 세션 종료 처리
        session.end_session()
        
        # 세션 저장
        self.save_session(session)
        
        # 활성 세션 캐시에서 제거
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        
        # 요약 생성
        summary = self._generate_session_summary(session)
        
        # 요약 저장
        self._save_session_summary(summary)
        
        logger.info(f"Ended session {session.session_id} and generated summary")
        return summary
    
    def get_session_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """저장된 세션 요약을 가져옵니다.
        
        Args:
            session_id: 요약을 가져올 세션의 ID
            
        Returns:
            ConversationSummary: 세션 요약, 없으면 None
        """
        try:
            summary_file = self.sessions_dir / f"{session_id}_summary.json"
            
            if not summary_file.exists():
                logger.warning(f"Session summary not found: {summary_file}")
                return None
            
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = f.read()
            
            summary = ConversationSummary.from_json(summary_data)
            logger.info(f"Retrieved summary for session {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to load session summary {session_id}: {e}")
            return None
    
    def get_active_sessions(self) -> List[ConversationSession]:
        """현재 활성 상태인 세션들을 반환합니다.
        
        Returns:
            List[ConversationSession]: 활성 세션 목록
        """
        return list(self._active_sessions.values())
    
    def list_all_sessions(self) -> List[str]:
        """저장된 모든 세션의 ID 목록을 반환합니다.
        
        Returns:
            List[str]: 세션 ID 목록
        """
        session_files = self.sessions_dir.glob("*.json")
        session_ids = []
        
        for file in session_files:
            if not file.name.endswith("_summary.json"):
                session_ids.append(file.stem)
        
        return sorted(session_ids)
    
    def get_user_progress_stats(self, user_language: str = "korean") -> Dict[str, Any]:
        """사용자의 전체 학습 진행 통계를 반환합니다.
        
        Args:
            user_language: 사용자의 모국어
            
        Returns:
            Dict[str, Any]: 학습 진행 통계
        """
        stats = {
            'total_sessions': 0,
            'total_messages': 0,
            'total_learning_points': 0,
            'topics_covered': set(),
            'average_session_duration': 0.0,
            'recent_sessions': []
        }
        
        session_ids = self.list_all_sessions()
        total_duration = 0.0
        sessions_with_duration = 0
        
        for session_id in session_ids:
            session = self.load_session(session_id)
            if session and session.user_language == user_language:
                stats['total_sessions'] += 1
                stats['total_messages'] += len(session.messages)
                stats['total_learning_points'] += len(session.learning_points)
                stats['topics_covered'].update(session.topics_covered)
                
                if session.get_duration():
                    total_duration += session.get_duration()
                    sessions_with_duration += 1
                
                # 최근 5개 세션 정보
                if len(stats['recent_sessions']) < 5:
                    stats['recent_sessions'].append({
                        'session_id': session_id,
                        'start_time': session.start_time.isoformat(),
                        'message_count': len(session.messages),
                        'topics': session.topics_covered[:3]  # 처음 3개 주제만
                    })
        
        # 평균 세션 지속 시간 계산
        if sessions_with_duration > 0:
            stats['average_session_duration'] = total_duration / sessions_with_duration
        
        # set을 list로 변환
        stats['topics_covered'] = list(stats['topics_covered'])
        
        return stats
    
    def _generate_session_summary(self, session: ConversationSession) -> ConversationSummary:
        """세션 요약을 생성합니다.
        
        Args:
            session: 요약할 대화 세션
            
        Returns:
            ConversationSummary: 생성된 요약
        """
        # 기본 통계 계산
        duration = session.get_duration() or 0.0
        total_messages = len(session.messages)
        
        # 주요 어휘 추출 (학습 포인트에서)
        key_vocabulary = []
        grammar_points = []
        
        for lp in session.learning_points:
            if 'vocabulary' in lp.topic.lower() or 'word' in lp.topic.lower():
                key_vocabulary.append(lp.topic)
            elif 'grammar' in lp.topic.lower() or 'tense' in lp.topic.lower():
                grammar_points.append(lp.topic)
        
        # 사용자 진행 상황 평가
        user_progress = self._assess_user_progress(session)
        
        # 학습 권장사항 생성
        recommendations = self._generate_recommendations(session)
        
        return ConversationSummary(
            session_id=session.session_id,
            duration_seconds=duration,
            total_messages=total_messages,
            topics_covered=session.topics_covered.copy(),
            learning_points=session.learning_points.copy(),
            key_vocabulary=key_vocabulary,
            grammar_points=grammar_points,
            user_progress=user_progress,
            recommendations=recommendations
        )
    
    def _assess_user_progress(self, session: ConversationSession) -> str:
        """세션을 기반으로 사용자 진행 상황을 평가합니다.
        
        Args:
            session: 평가할 대화 세션
            
        Returns:
            str: 진행 상황 평가 텍스트
        """
        message_count = len(session.messages)
        learning_points_count = len(session.learning_points)
        topics_count = len(session.topics_covered)
        
        if message_count < 10:
            return "짧은 대화였지만 기본적인 영어 표현을 연습했습니다."
        elif message_count < 30:
            return f"{topics_count}개 주제에 대해 활발한 대화를 나누며 {learning_points_count}개의 학습 포인트를 익혔습니다."
        else:
            return f"매우 활발한 대화를 통해 {topics_count}개 주제를 깊이 있게 다루고 {learning_points_count}개의 중요한 학습 포인트를 습득했습니다."
    
    def _generate_recommendations(self, session: ConversationSession) -> List[str]:
        """세션을 기반으로 학습 권장사항을 생성합니다.
        
        Args:
            session: 분석할 대화 세션
            
        Returns:
            List[str]: 학습 권장사항 목록
        """
        recommendations = []
        
        # 학습 포인트 기반 권장사항
        if len(session.learning_points) > 5:
            recommendations.append("다양한 학습 포인트를 다뤘으니 복습을 통해 정착시키는 것이 좋겠습니다.")
        elif len(session.learning_points) < 3:
            recommendations.append("더 많은 질문과 표현을 시도해보세요.")
        
        # 주제 다양성 기반 권장사항
        if len(session.topics_covered) > 3:
            recommendations.append("다양한 주제로 대화했으니 각 주제별 핵심 표현을 정리해보세요.")
        else:
            recommendations.append("관심 있는 다른 주제들로도 대화를 시도해보세요.")
        
        # 대화 길이 기반 권장사항
        if len(session.messages) > 20:
            recommendations.append("긴 대화를 잘 유지했습니다. 이런 패턴을 계속 유지해보세요.")
        
        # 기본 권장사항
        if not recommendations:
            recommendations.append("꾸준한 대화 연습을 통해 영어 실력을 향상시켜보세요.")
        
        return recommendations
    
    def _save_session_summary(self, summary: ConversationSummary) -> None:
        """세션 요약을 파일에 저장합니다.
        
        Args:
            summary: 저장할 세션 요약
            
        Raises:
            IOError: 파일 저장 중 오류가 발생한 경우
        """
        try:
            summary_file = self.sessions_dir / f"{summary.session_id}_summary.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary.to_json())
            
            logger.info(f"Saved session summary {summary.session_id} to {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session summary {summary.session_id}: {e}")
            raise IOError(f"Failed to save session summary: {e}")