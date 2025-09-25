"""
학습 어시스턴트 모듈.

이 모듈은 사용자의 영어 학습을 지원하는 LearningAssistant 클래스를 포함합니다.
사용자의 영어 텍스트를 분석하고, 문법 교정, 어휘 향상 제안 등의 기능을 제공합니다.
"""

import logging
import re
from typing import List, Optional, Dict, Any

from ..llm.base import LanguageModel, LanguageModelError
from ..models.llm import EnglishAnalysis, GrammarError, ErrorType, ImprovementSuggestion
from ..models.response import (
    LearningFeedback, Correction, GrammarTip, VocabSuggestion
)

logger = logging.getLogger(__name__)


class LearningAssistantError(Exception):
    """학습 어시스턴트 관련 오류를 나타내는 예외 클래스."""
    pass


class LearningAssistant:
    """사용자의 영어 학습을 지원하는 어시스턴트 클래스.
    
    이 클래스는 사용자가 입력한 영어 텍스트를 분석하여 문법 오류를 식별하고,
    교정 제안과 어휘 향상 제안을 제공합니다.
    """
    
    def __init__(self, llm: LanguageModel, user_language: str = "korean"):
        """학습 어시스턴트 초기화.
        
        Args:
            llm: 사용할 언어 모델
            user_language: 사용자의 모국어 (설명에 사용)
        """
        self.llm = llm
        self.user_language = user_language
        self._common_errors = self._load_common_errors()
        
        logger.info(f"LearningAssistant initialized with {llm.__class__.__name__}")
    
    def analyze_user_english(self, text: str) -> EnglishAnalysis:
        """사용자의 영어 텍스트를 분석합니다.
        
        Args:
            text: 분석할 영어 텍스트
            
        Returns:
            EnglishAnalysis: 분석 결과
            
        Raises:
            LearningAssistantError: 분석 실패 시
        """
        try:
            if not text.strip():
                raise ValueError("분석할 텍스트가 비어있습니다.")
            
            logger.info(f"Analyzing English text: {text[:50]}...")
            
            # LLM을 통한 문법 분석
            analysis = self.llm.analyze_grammar(text, self.user_language)
            
            # 추가적인 로컬 분석 수행
            self._enhance_analysis_with_local_checks(text, analysis)
            
            logger.info(f"Analysis completed: {len(analysis.grammar_errors)} errors, "
                       f"{len(analysis.suggestions)} suggestions")
            
            return analysis
            
        except LanguageModelError as e:
            logger.error(f"LLM analysis failed: {e}")
            raise LearningAssistantError(f"영어 분석 중 오류가 발생했습니다: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in analyze_user_english: {e}")
            raise LearningAssistantError(f"예상치 못한 오류가 발생했습니다: {e}")
    
    def provide_corrections(self, text: str, analysis: Optional[EnglishAnalysis] = None) -> List[Correction]:
        """문법 오류에 대한 교정 제안을 제공합니다.
        
        Args:
            text: 원본 텍스트
            analysis: 기존 분석 결과 (없으면 새로 분석)
            
        Returns:
            List[Correction]: 교정 제안 목록
            
        Raises:
            LearningAssistantError: 교정 제안 생성 실패 시
        """
        try:
            if analysis is None:
                analysis = self.analyze_user_english(text)
            
            corrections = []
            
            for error in analysis.grammar_errors:
                # 오류 유형에 따른 설명 생성
                explanation = self._generate_error_explanation(error)
                
                correction = Correction(
                    original_text=error.text,
                    corrected_text=error.suggestion,
                    explanation=explanation,
                    error_type=error.error_type.value
                )
                corrections.append(correction)
            
            logger.info(f"Generated {len(corrections)} corrections")
            return corrections
            
        except Exception as e:
            logger.error(f"Error in provide_corrections: {e}")
            raise LearningAssistantError(f"교정 제안 생성 중 오류가 발생했습니다: {e}")
    
    def explain_grammar_point(self, text: str, grammar_point: str) -> GrammarTip:
        """특정 문법 포인트에 대한 설명을 제공합니다.
        
        Args:
            text: 관련 텍스트
            grammar_point: 설명할 문법 포인트
            
        Returns:
            GrammarTip: 문법 설명
            
        Raises:
            LearningAssistantError: 설명 생성 실패 시
        """
        try:
            # LLM을 통한 문법 설명 생성
            prompt = self._create_grammar_explanation_prompt(text, grammar_point)
            response = self.llm.generate_response(prompt)
            
            # 응답 파싱하여 GrammarTip 생성
            grammar_tip = self._parse_grammar_explanation(response.content, grammar_point)
            
            logger.info(f"Generated grammar explanation for: {grammar_point}")
            return grammar_tip
            
        except Exception as e:
            logger.error(f"Error in explain_grammar_point: {e}")
            raise LearningAssistantError(f"문법 설명 생성 중 오류가 발생했습니다: {e}")
    
    def suggest_vocabulary_improvements(self, text: str, analysis: Optional[EnglishAnalysis] = None) -> List[VocabSuggestion]:
        """어휘 향상 제안을 제공합니다.
        
        Args:
            text: 원본 텍스트
            analysis: 기존 분석 결과 (없으면 새로 분석)
            
        Returns:
            List[VocabSuggestion]: 어휘 제안 목록
            
        Raises:
            LearningAssistantError: 어휘 제안 생성 실패 시
        """
        try:
            if analysis is None:
                analysis = self.analyze_user_english(text)
            
            vocab_suggestions = []
            
            # 분석 결과에서 어휘 관련 제안 추출
            for suggestion in analysis.suggestions:
                if suggestion.category.lower() in ['vocabulary', 'word choice', 'vocab']:
                    vocab_suggestion = self._create_vocab_suggestion(suggestion, text)
                    vocab_suggestions.append(vocab_suggestion)
            
            # 추가적인 어휘 제안 생성
            additional_suggestions = self._generate_additional_vocab_suggestions(text)
            vocab_suggestions.extend(additional_suggestions)
            
            logger.info(f"Generated {len(vocab_suggestions)} vocabulary suggestions")
            return vocab_suggestions
            
        except Exception as e:
            logger.error(f"Error in suggest_vocabulary_improvements: {e}")
            raise LearningAssistantError(f"어휘 제안 생성 중 오류가 발생했습니다: {e}")
    
    def create_learning_feedback(self, text: str) -> LearningFeedback:
        """종합적인 학습 피드백을 생성합니다.
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            LearningFeedback: 종합 학습 피드백
            
        Raises:
            LearningAssistantError: 피드백 생성 실패 시
        """
        try:
            # 영어 분석 수행
            analysis = self.analyze_user_english(text)
            
            # 각종 제안 생성
            corrections = self.provide_corrections(text, analysis)
            vocab_suggestions = self.suggest_vocabulary_improvements(text, analysis)
            
            # 문법 팁 생성 (주요 오류가 있는 경우)
            grammar_tips = []
            if analysis.grammar_errors:
                main_error_type = self._get_most_common_error_type(analysis.grammar_errors)
                grammar_tip = self._create_grammar_tip_for_error_type(main_error_type)
                grammar_tips.append(grammar_tip)
            
            # 격려 메시지 생성
            encouragement = self._generate_encouragement_message(analysis)
            
            feedback = LearningFeedback(
                corrections=corrections,
                grammar_tips=grammar_tips,
                vocabulary_suggestions=vocab_suggestions,
                encouragement=encouragement
            )
            
            logger.info(f"Created comprehensive learning feedback with "
                       f"{len(corrections)} corrections, {len(grammar_tips)} tips, "
                       f"{len(vocab_suggestions)} vocab suggestions")
            
            return feedback
            
        except Exception as e:
            logger.error(f"Error in create_learning_feedback: {e}")
            raise LearningAssistantError(f"학습 피드백 생성 중 오류가 발생했습니다: {e}")
    
    def _enhance_analysis_with_local_checks(self, text: str, analysis: EnglishAnalysis) -> None:
        """로컬 체크를 통해 분석 결과를 보강합니다.
        
        Args:
            text: 원본 텍스트
            analysis: 보강할 분석 결과
        """
        # 일반적인 오류 패턴 체크
        for pattern, error_info in self._common_errors.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # 이미 발견된 오류와 중복되지 않는 경우에만 추가
                if not self._is_duplicate_error(match, analysis.grammar_errors):
                    error = GrammarError(
                        text=match.group(),
                        error_type=ErrorType(error_info['type']),
                        position=(match.start(), match.end()),
                        suggestion=error_info['suggestion'],
                        explanation=error_info['explanation']
                    )
                    analysis.grammar_errors.append(error)
    
    def _generate_error_explanation(self, error: GrammarError) -> str:
        """문법 오류에 대한 설명을 생성합니다.
        
        Args:
            error: 문법 오류
            
        Returns:
            str: 오류 설명
        """
        base_explanation = error.explanation
        
        # 사용자 언어에 따른 추가 설명
        if self.user_language == "korean":
            error_type_explanations = {
                ErrorType.GRAMMAR: "문법 오류입니다.",
                ErrorType.VOCABULARY: "더 적절한 어휘를 사용할 수 있습니다.",
                ErrorType.SPELLING: "철자 오류입니다.",
                ErrorType.PUNCTUATION: "구두점 사용에 문제가 있습니다.",
                ErrorType.SYNTAX: "문장 구조에 문제가 있습니다."
            }
            
            type_explanation = error_type_explanations.get(error.error_type, "")
            if type_explanation:
                base_explanation = f"{type_explanation} {base_explanation}"
        
        return base_explanation
    
    def _create_grammar_explanation_prompt(self, text: str, grammar_point: str) -> str:
        """문법 설명을 위한 프롬프트를 생성합니다.
        
        Args:
            text: 관련 텍스트
            grammar_point: 문법 포인트
            
        Returns:
            str: 생성된 프롬프트
        """
        if self.user_language == "korean":
            return f"""
다음 영어 텍스트에서 '{grammar_point}'에 대해 한국어로 설명해주세요:

텍스트: "{text}"

다음 형식으로 답변해주세요:
1. 문법 규칙: [규칙 설명]
2. 설명: [자세한 설명]
3. 예시: [2-3개의 예문]
4. 난이도: [beginner/intermediate/advanced]
"""
        else:
            return f"""
Please explain the grammar point '{grammar_point}' in the following English text:

Text: "{text}"

Please provide:
1. Grammar rule: [rule explanation]
2. Explanation: [detailed explanation]
3. Examples: [2-3 example sentences]
4. Difficulty level: [beginner/intermediate/advanced]
"""
    
    def _parse_grammar_explanation(self, response: str, grammar_point: str) -> GrammarTip:
        """LLM 응답을 파싱하여 GrammarTip을 생성합니다.
        
        Args:
            response: LLM 응답
            grammar_point: 문법 포인트
            
        Returns:
            GrammarTip: 파싱된 문법 팁
        """
        # 간단한 파싱 로직 (실제로는 더 정교한 파싱이 필요할 수 있음)
        lines = response.strip().split('\n')
        
        rule = grammar_point
        explanation = response
        examples = []
        difficulty_level = "intermediate"
        
        # 응답에서 정보 추출 시도
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', 'Grammar rule:', '문법 규칙:')):
                rule = line.split(':', 1)[-1].strip()
            elif line.startswith(('2.', 'Explanation:', '설명:')):
                explanation = line.split(':', 1)[-1].strip()
            elif line.startswith(('3.', 'Examples:', '예시:')):
                examples_text = line.split(':', 1)[-1].strip()
                examples = [ex.strip() for ex in examples_text.split(',') if ex.strip()]
            elif line.startswith(('4.', 'Difficulty:', '난이도:')):
                difficulty_text = line.split(':', 1)[-1].strip().lower()
                if difficulty_text in ['beginner', 'intermediate', 'advanced']:
                    difficulty_level = difficulty_text
        
        return GrammarTip(
            rule=rule,
            explanation=explanation,
            examples=examples,
            difficulty_level=difficulty_level
        )
    
    def _create_vocab_suggestion(self, suggestion: ImprovementSuggestion, text: str) -> VocabSuggestion:
        """ImprovementSuggestion에서 VocabSuggestion을 생성합니다.
        
        Args:
            suggestion: 개선 제안
            text: 원본 텍스트
            
        Returns:
            VocabSuggestion: 어휘 제안
        """
        # 동의어 생성 (간단한 예시)
        synonyms = self._get_synonyms(suggestion.improved)
        
        return VocabSuggestion(
            word=suggestion.original,
            definition=f"더 나은 표현: {suggestion.improved}",
            usage_example=f"예시: {text.replace(suggestion.original, suggestion.improved)}",
            synonyms=synonyms,
            difficulty_level="intermediate"
        )
    
    def _generate_additional_vocab_suggestions(self, text: str) -> List[VocabSuggestion]:
        """추가적인 어휘 제안을 생성합니다.
        
        Args:
            text: 원본 텍스트
            
        Returns:
            List[VocabSuggestion]: 추가 어휘 제안 목록
        """
        suggestions = []
        
        # 간단한 단어들에 대한 고급 대안 제안
        simple_to_advanced = {
            'good': ['excellent', 'outstanding', 'remarkable'],
            'bad': ['terrible', 'awful', 'dreadful'],
            'big': ['enormous', 'massive', 'gigantic'],
            'small': ['tiny', 'minuscule', 'compact'],
            'nice': ['pleasant', 'delightful', 'wonderful'],
            'very': ['extremely', 'tremendously', 'exceptionally']
        }
        
        words = text.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in simple_to_advanced:
                alternatives = simple_to_advanced[clean_word]
                suggestion = VocabSuggestion(
                    word=clean_word,
                    definition=f"더 정교한 표현들: {', '.join(alternatives)}",
                    usage_example=f"'{clean_word}' 대신 '{alternatives[0]}'를 사용해보세요.",
                    synonyms=alternatives,
                    difficulty_level="intermediate"
                )
                suggestions.append(suggestion)
        
        return suggestions[:3]  # 최대 3개까지만 반환
    
    def _get_most_common_error_type(self, errors: List[GrammarError]) -> ErrorType:
        """가장 흔한 오류 유형을 찾습니다.
        
        Args:
            errors: 오류 목록
            
        Returns:
            ErrorType: 가장 흔한 오류 유형
        """
        if not errors:
            return ErrorType.GRAMMAR
        
        error_counts = {}
        for error in errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        
        return max(error_counts, key=error_counts.get)
    
    def _create_grammar_tip_for_error_type(self, error_type: ErrorType) -> GrammarTip:
        """오류 유형에 따른 문법 팁을 생성합니다.
        
        Args:
            error_type: 오류 유형
            
        Returns:
            GrammarTip: 문법 팁
        """
        tips = {
            ErrorType.GRAMMAR: GrammarTip(
                rule="기본 문법 규칙",
                explanation="영어 문법의 기본 규칙을 확인해보세요. 주어-동사 일치, 시제 일치 등을 점검하세요.",
                examples=["She goes to school.", "They are playing soccer."],
                difficulty_level="beginner"
            ),
            ErrorType.VOCABULARY: GrammarTip(
                rule="어휘 선택",
                explanation="문맥에 맞는 적절한 어휘를 선택하는 것이 중요합니다.",
                examples=["I'm interested in music.", "The weather is pleasant today."],
                difficulty_level="intermediate"
            ),
            ErrorType.SPELLING: GrammarTip(
                rule="철자 확인",
                explanation="영어 단어의 정확한 철자를 확인하세요.",
                examples=["receive (not recieve)", "definitely (not definately)"],
                difficulty_level="beginner"
            ),
            ErrorType.PUNCTUATION: GrammarTip(
                rule="구두점 사용",
                explanation="적절한 구두점 사용으로 문장의 의미를 명확히 하세요.",
                examples=["Hello, how are you?", "I like apples, oranges, and bananas."],
                difficulty_level="beginner"
            ),
            ErrorType.SYNTAX: GrammarTip(
                rule="문장 구조",
                explanation="영어의 기본 어순(SVO)과 문장 구조를 확인하세요.",
                examples=["I read a book.", "She gave me a present."],
                difficulty_level="intermediate"
            )
        }
        
        return tips.get(error_type, tips[ErrorType.GRAMMAR])
    
    def _generate_encouragement_message(self, analysis: EnglishAnalysis) -> str:
        """격려 메시지를 생성합니다.
        
        Args:
            analysis: 분석 결과
            
        Returns:
            str: 격려 메시지
        """
        if self.user_language == "korean":
            if analysis.fluency_score >= 0.8:
                return "훌륭합니다! 영어 실력이 많이 향상되었네요. 계속 이런 식으로 연습하세요!"
            elif analysis.fluency_score >= 0.6:
                return "잘하고 있습니다! 몇 가지 개선점만 보완하면 더욱 자연스러운 영어가 될 것입니다."
            else:
                return "좋은 시작입니다! 꾸준히 연습하면 분명히 실력이 향상될 것입니다. 포기하지 마세요!"
        else:
            if analysis.fluency_score >= 0.8:
                return "Excellent work! Your English skills have improved significantly. Keep practicing like this!"
            elif analysis.fluency_score >= 0.6:
                return "Good job! With a few improvements, your English will sound even more natural."
            else:
                return "Great start! Keep practicing consistently and your skills will definitely improve. Don't give up!"
    
    def _load_common_errors(self) -> Dict[str, Dict[str, str]]:
        """일반적인 오류 패턴을 로드합니다.
        
        Returns:
            Dict[str, Dict[str, str]]: 오류 패턴과 정보
        """
        return {
            r'\bi am interesting\b': {
                'type': 'vocabulary',
                'suggestion': 'I am interested',
                'explanation': '"interesting"은 "흥미로운"이라는 뜻이고, "interested"는 "관심이 있는"이라는 뜻입니다.'
            },
            r'\bhow about you\?\s*$': {
                'type': 'grammar',
                'suggestion': 'What about you?',
                'explanation': '상대방의 의견을 물을 때는 "What about you?"가 더 자연스럽습니다.'
            },
            r'\bi have a plan to\b': {
                'type': 'vocabulary',
                'suggestion': 'I plan to',
                'explanation': '"I plan to"가 더 자연스러운 표현입니다.'
            }
        }
    
    def _is_duplicate_error(self, match: re.Match, existing_errors: List[GrammarError]) -> bool:
        """중복 오류인지 확인합니다.
        
        Args:
            match: 새로 발견된 매치
            existing_errors: 기존 오류 목록
            
        Returns:
            bool: 중복이면 True
        """
        for error in existing_errors:
            if (error.position[0] <= match.start() <= error.position[1] or
                error.position[0] <= match.end() <= error.position[1]):
                return True
        return False
    
    def _get_synonyms(self, word: str) -> List[str]:
        """단어의 동의어를 반환합니다.
        
        Args:
            word: 단어
            
        Returns:
            List[str]: 동의어 목록
        """
        # 간단한 동의어 사전 (실제로는 더 정교한 사전이나 API를 사용할 수 있음)
        synonyms_dict = {
            'excellent': ['outstanding', 'superb', 'exceptional'],
            'good': ['fine', 'great', 'nice'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['little', 'tiny', 'minute']
        }
        
        return synonyms_dict.get(word.lower(), [])