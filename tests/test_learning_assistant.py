"""
LearningAssistant 클래스에 대한 단위 테스트.

이 모듈은 학습 어시스턴트의 모든 기능을 테스트합니다.
"""

import pytest
from unittest.mock import Mock, MagicMock
import re

from src.document_rag_english_study.conversation.learning_assistant import (
    LearningAssistant, LearningAssistantError
)
from src.document_rag_english_study.llm.base import LanguageModel, LanguageModelError
from src.document_rag_english_study.models.llm import (
    EnglishAnalysis, GrammarError, ErrorType, ImprovementSuggestion, LLMResponse
)
from src.document_rag_english_study.models.response import (
    LearningFeedback, Correction, GrammarTip, VocabSuggestion
)


class TestLearningAssistant:
    """LearningAssistant 클래스 테스트."""
    
    @pytest.fixture
    def mock_llm(self):
        """모의 LLM 객체를 생성합니다."""
        llm = Mock(spec=LanguageModel)
        llm.__class__.__name__ = "MockLLM"
        return llm
    
    @pytest.fixture
    def learning_assistant(self, mock_llm):
        """LearningAssistant 인스턴스를 생성합니다."""
        return LearningAssistant(mock_llm, user_language="korean")
    
    @pytest.fixture
    def sample_analysis(self):
        """샘플 분석 결과를 생성합니다."""
        grammar_error = GrammarError(
            text="I am interesting",
            error_type=ErrorType.VOCABULARY,
            position=(0, 16),
            suggestion="I am interested",
            explanation="Use 'interested' for personal feelings"
        )
        
        improvement = ImprovementSuggestion(
            category="vocabulary",
            original="good",
            improved="excellent",
            reason="More sophisticated vocabulary",
            confidence=0.9
        )
        
        return EnglishAnalysis(
            grammar_errors=[grammar_error],
            vocabulary_level="intermediate",
            fluency_score=0.7,
            suggestions=[improvement],
            complexity_score=0.6
        )
    
    def test_init(self, mock_llm):
        """초기화 테스트."""
        assistant = LearningAssistant(mock_llm, user_language="korean")
        
        assert assistant.llm == mock_llm
        assert assistant.user_language == "korean"
        assert isinstance(assistant._common_errors, dict)
    
    def test_init_default_language(self, mock_llm):
        """기본 언어 설정 테스트."""
        assistant = LearningAssistant(mock_llm)
        assert assistant.user_language == "korean"
    
    def test_analyze_user_english_success(self, learning_assistant, mock_llm, sample_analysis):
        """영어 분석 성공 테스트."""
        text = "I am interesting in music"
        mock_llm.analyze_grammar.return_value = sample_analysis
        
        result = learning_assistant.analyze_user_english(text)
        
        assert isinstance(result, EnglishAnalysis)
        assert len(result.grammar_errors) >= 1
        assert result.vocabulary_level == "intermediate"
        assert result.fluency_score == 0.7
        mock_llm.analyze_grammar.assert_called_once_with(text, "korean")
    
    def test_analyze_user_english_empty_text(self, learning_assistant):
        """빈 텍스트 분석 테스트."""
        with pytest.raises(LearningAssistantError, match="분석할 텍스트가 비어있습니다"):
            learning_assistant.analyze_user_english("")
    
    def test_analyze_user_english_whitespace_only(self, learning_assistant):
        """공백만 있는 텍스트 분석 테스트."""
        with pytest.raises(LearningAssistantError, match="분석할 텍스트가 비어있습니다"):
            learning_assistant.analyze_user_english("   ")
    
    def test_analyze_user_english_llm_error(self, learning_assistant, mock_llm):
        """LLM 오류 처리 테스트."""
        text = "Test text"
        mock_llm.analyze_grammar.side_effect = LanguageModelError("API error")
        
        with pytest.raises(LearningAssistantError, match="영어 분석 중 오류가 발생했습니다"):
            learning_assistant.analyze_user_english(text)
    
    def test_provide_corrections_with_analysis(self, learning_assistant, sample_analysis):
        """기존 분석 결과로 교정 제안 테스트."""
        text = "I am interesting in music"
        
        corrections = learning_assistant.provide_corrections(text, sample_analysis)
        
        assert len(corrections) == 1
        assert isinstance(corrections[0], Correction)
        assert corrections[0].original_text == "I am interesting"
        assert corrections[0].corrected_text == "I am interested"
        assert corrections[0].error_type == "vocabulary"
        assert "interested" in corrections[0].explanation
    
    def test_provide_corrections_without_analysis(self, learning_assistant, mock_llm, sample_analysis):
        """새로운 분석으로 교정 제안 테스트."""
        text = "I am interesting in music"
        mock_llm.analyze_grammar.return_value = sample_analysis
        
        corrections = learning_assistant.provide_corrections(text)
        
        assert len(corrections) == 1
        mock_llm.analyze_grammar.assert_called_once_with(text, "korean")
    
    def test_explain_grammar_point(self, learning_assistant, mock_llm):
        """문법 설명 테스트."""
        text = "I am interesting"
        grammar_point = "adjectives vs participles"
        
        mock_response = LLMResponse(
            content="""1. Grammar rule: Adjectives ending in -ed vs -ing
2. Explanation: Use -ed for feelings, -ing for characteristics
3. Examples: I am interested, The book is interesting
4. Difficulty level: intermediate""",
            model="test-model"
        )
        mock_llm.generate_response.return_value = mock_response
        
        result = learning_assistant.explain_grammar_point(text, grammar_point)
        
        assert isinstance(result, GrammarTip)
        assert "Adjectives ending in -ed vs -ing" in result.rule
        assert "feelings" in result.explanation
        assert len(result.examples) > 0
        assert result.difficulty_level == "intermediate"
    
    def test_suggest_vocabulary_improvements_with_analysis(self, learning_assistant, sample_analysis):
        """기존 분석으로 어휘 제안 테스트."""
        text = "This is good music"
        
        suggestions = learning_assistant.suggest_vocabulary_improvements(text, sample_analysis)
        
        assert len(suggestions) >= 1
        # 분석 결과의 vocabulary 제안이 포함되어야 함
        vocab_suggestion = next((s for s in suggestions if s.word == "good"), None)
        assert vocab_suggestion is not None
        assert "excellent" in vocab_suggestion.definition
    
    def test_suggest_vocabulary_improvements_without_analysis(self, learning_assistant, mock_llm, sample_analysis):
        """새로운 분석으로 어휘 제안 테스트."""
        text = "This is good music"
        mock_llm.analyze_grammar.return_value = sample_analysis
        
        suggestions = learning_assistant.suggest_vocabulary_improvements(text)
        
        assert len(suggestions) >= 1
        mock_llm.analyze_grammar.assert_called_once_with(text, "korean")
    
    def test_create_learning_feedback(self, learning_assistant, mock_llm, sample_analysis):
        """종합 학습 피드백 생성 테스트."""
        text = "I am interesting in good music"
        mock_llm.analyze_grammar.return_value = sample_analysis
        
        # 문법 설명을 위한 모의 응답
        mock_response = LLMResponse(
            content="Grammar explanation for vocabulary errors",
            model="test-model"
        )
        mock_llm.generate_response.return_value = mock_response
        
        feedback = learning_assistant.create_learning_feedback(text)
        
        assert isinstance(feedback, LearningFeedback)
        assert len(feedback.corrections) >= 1
        assert len(feedback.grammar_tips) >= 1
        assert len(feedback.vocabulary_suggestions) >= 1
        assert feedback.encouragement != ""
        assert "잘하고 있습니다" in feedback.encouragement or "좋은 시작" in feedback.encouragement
    
    def test_enhance_analysis_with_local_checks(self, learning_assistant, sample_analysis):
        """로컬 체크로 분석 보강 테스트."""
        text = "I am interesting in music. How about you?"
        
        # 초기 오류 수 확인
        initial_error_count = len(sample_analysis.grammar_errors)
        
        learning_assistant._enhance_analysis_with_local_checks(text, sample_analysis)
        
        # 로컬 체크로 추가 오류가 발견되어야 함
        assert len(sample_analysis.grammar_errors) >= initial_error_count
        
        # "how about you" 패턴이 감지되어야 함
        how_about_error = next(
            (error for error in sample_analysis.grammar_errors 
             if "how about you" in error.text.lower()), 
            None
        )
        assert how_about_error is not None
    
    def test_generate_error_explanation_korean(self, learning_assistant):
        """한국어 오류 설명 생성 테스트."""
        error = GrammarError(
            text="I am interesting",
            error_type=ErrorType.VOCABULARY,
            position=(0, 16),
            suggestion="I am interested",
            explanation="Use interested for personal feelings"
        )
        
        explanation = learning_assistant._generate_error_explanation(error)
        
        assert "더 적절한 어휘를 사용할 수 있습니다" in explanation
        assert "Use interested for personal feelings" in explanation
    
    def test_generate_error_explanation_english(self, mock_llm):
        """영어 오류 설명 생성 테스트."""
        assistant = LearningAssistant(mock_llm, user_language="english")
        
        error = GrammarError(
            text="I am interesting",
            error_type=ErrorType.VOCABULARY,
            position=(0, 16),
            suggestion="I am interested",
            explanation="Use interested for personal feelings"
        )
        
        explanation = assistant._generate_error_explanation(error)
        
        # 영어 사용자의 경우 추가 설명이 없어야 함
        assert explanation == "Use interested for personal feelings"
    
    def test_get_most_common_error_type(self, learning_assistant):
        """가장 흔한 오류 유형 찾기 테스트."""
        errors = [
            GrammarError("error1", ErrorType.GRAMMAR, (0, 5), "fix1", "exp1"),
            GrammarError("error2", ErrorType.VOCABULARY, (6, 10), "fix2", "exp2"),
            GrammarError("error3", ErrorType.GRAMMAR, (11, 15), "fix3", "exp3"),
        ]
        
        most_common = learning_assistant._get_most_common_error_type(errors)
        assert most_common == ErrorType.GRAMMAR
    
    def test_get_most_common_error_type_empty(self, learning_assistant):
        """빈 오류 목록에서 기본 유형 반환 테스트."""
        most_common = learning_assistant._get_most_common_error_type([])
        assert most_common == ErrorType.GRAMMAR
    
    def test_create_grammar_tip_for_error_type(self, learning_assistant):
        """오류 유형별 문법 팁 생성 테스트."""
        tip = learning_assistant._create_grammar_tip_for_error_type(ErrorType.VOCABULARY)
        
        assert isinstance(tip, GrammarTip)
        assert "어휘" in tip.rule
        assert len(tip.examples) > 0
        assert tip.difficulty_level in ["beginner", "intermediate", "advanced"]
    
    def test_generate_encouragement_message_high_score(self, learning_assistant):
        """높은 점수에 대한 격려 메시지 테스트."""
        analysis = EnglishAnalysis(fluency_score=0.9)
        message = learning_assistant._generate_encouragement_message(analysis)
        
        assert "훌륭합니다" in message
    
    def test_generate_encouragement_message_medium_score(self, learning_assistant):
        """중간 점수에 대한 격려 메시지 테스트."""
        analysis = EnglishAnalysis(fluency_score=0.7)
        message = learning_assistant._generate_encouragement_message(analysis)
        
        assert "잘하고 있습니다" in message
    
    def test_generate_encouragement_message_low_score(self, learning_assistant):
        """낮은 점수에 대한 격려 메시지 테스트."""
        analysis = EnglishAnalysis(fluency_score=0.4)
        message = learning_assistant._generate_encouragement_message(analysis)
        
        assert "좋은 시작" in message
    
    def test_generate_encouragement_message_english(self, mock_llm):
        """영어 격려 메시지 테스트."""
        assistant = LearningAssistant(mock_llm, user_language="english")
        analysis = EnglishAnalysis(fluency_score=0.9)
        
        message = assistant._generate_encouragement_message(analysis)
        
        assert "Excellent work" in message
    
    def test_is_duplicate_error(self, learning_assistant):
        """중복 오류 확인 테스트."""
        existing_errors = [
            GrammarError("error1", ErrorType.GRAMMAR, (5, 15), "fix1", "exp1")
        ]
        
        # 겹치는 매치
        overlapping_match = Mock()
        overlapping_match.start.return_value = 10
        overlapping_match.end.return_value = 20
        
        # 겹치지 않는 매치
        non_overlapping_match = Mock()
        non_overlapping_match.start.return_value = 25
        non_overlapping_match.end.return_value = 30
        
        assert learning_assistant._is_duplicate_error(overlapping_match, existing_errors) == True
        assert learning_assistant._is_duplicate_error(non_overlapping_match, existing_errors) == False
    
    def test_get_synonyms(self, learning_assistant):
        """동의어 반환 테스트."""
        synonyms = learning_assistant._get_synonyms("excellent")
        
        assert isinstance(synonyms, list)
        assert "outstanding" in synonyms
        assert "superb" in synonyms
    
    def test_get_synonyms_unknown_word(self, learning_assistant):
        """알 수 없는 단어의 동의어 테스트."""
        synonyms = learning_assistant._get_synonyms("unknownword")
        
        assert synonyms == []
    
    def test_create_vocab_suggestion(self, learning_assistant):
        """어휘 제안 생성 테스트."""
        suggestion = ImprovementSuggestion(
            category="vocabulary",
            original="good",
            improved="excellent",
            reason="Better word choice",
            confidence=0.9
        )
        text = "This is good music"
        
        vocab_suggestion = learning_assistant._create_vocab_suggestion(suggestion, text)
        
        assert isinstance(vocab_suggestion, VocabSuggestion)
        assert vocab_suggestion.word == "good"
        assert "excellent" in vocab_suggestion.definition
        assert "excellent" in vocab_suggestion.usage_example
        assert len(vocab_suggestion.synonyms) > 0
    
    def test_generate_additional_vocab_suggestions(self, learning_assistant):
        """추가 어휘 제안 생성 테스트."""
        text = "This is very good and nice music"
        
        suggestions = learning_assistant._generate_additional_vocab_suggestions(text)
        
        assert len(suggestions) <= 3  # 최대 3개
        assert any(s.word == "good" for s in suggestions)
        assert any(s.word == "nice" for s in suggestions)
        assert any(s.word == "very" for s in suggestions)
    
    def test_load_common_errors(self, learning_assistant):
        """일반적인 오류 패턴 로드 테스트."""
        common_errors = learning_assistant._load_common_errors()
        
        assert isinstance(common_errors, dict)
        assert len(common_errors) > 0
        
        # 특정 패턴이 포함되어 있는지 확인
        patterns = list(common_errors.keys())
        assert any("interesting" in pattern for pattern in patterns)
        assert any("how about you" in pattern for pattern in patterns)
    
    def test_parse_grammar_explanation_structured(self, learning_assistant):
        """구조화된 문법 설명 파싱 테스트."""
        response = """1. Grammar rule: Subject-verb agreement
2. Explanation: The verb must agree with the subject
3. Examples: She goes, They go, He has
4. Difficulty level: beginner"""
        
        tip = learning_assistant._parse_grammar_explanation(response, "subject-verb agreement")
        
        assert "Subject-verb agreement" in tip.rule
        assert "agree with the subject" in tip.explanation
        assert len(tip.examples) == 3
        assert tip.difficulty_level == "beginner"
    
    def test_parse_grammar_explanation_unstructured(self, learning_assistant):
        """비구조화된 문법 설명 파싱 테스트."""
        response = "This is a general explanation about grammar without specific structure."
        
        tip = learning_assistant._parse_grammar_explanation(response, "general grammar")
        
        assert tip.rule == "general grammar"
        assert tip.explanation == response
        assert tip.difficulty_level == "intermediate"  # 기본값
    
    def test_create_grammar_explanation_prompt_korean(self, learning_assistant):
        """한국어 문법 설명 프롬프트 생성 테스트."""
        text = "I am interesting"
        grammar_point = "adjectives"
        
        prompt = learning_assistant._create_grammar_explanation_prompt(text, grammar_point)
        
        assert "한국어로 설명해주세요" in prompt
        assert text in prompt
        assert grammar_point in prompt
        assert "문법 규칙" in prompt
    
    def test_create_grammar_explanation_prompt_english(self, mock_llm):
        """영어 문법 설명 프롬프트 생성 테스트."""
        assistant = LearningAssistant(mock_llm, user_language="english")
        text = "I am interesting"
        grammar_point = "adjectives"
        
        prompt = assistant._create_grammar_explanation_prompt(text, grammar_point)
        
        assert "Please explain" in prompt
        assert text in prompt
        assert grammar_point in prompt
        assert "Grammar rule" in prompt


class TestLearningAssistantIntegration:
    """LearningAssistant 통합 테스트."""
    
    @pytest.fixture
    def mock_llm_with_responses(self):
        """응답이 설정된 모의 LLM을 생성합니다."""
        llm = Mock(spec=LanguageModel)
        llm.__class__.__name__ = "MockLLM"
        
        # 분석 결과 설정
        analysis = EnglishAnalysis(
            grammar_errors=[
                GrammarError(
                    text="I am interesting",
                    error_type=ErrorType.VOCABULARY,
                    position=(0, 16),
                    suggestion="I am interested",
                    explanation="Use 'interested' for personal feelings"
                )
            ],
            vocabulary_level="intermediate",
            fluency_score=0.7,
            suggestions=[
                ImprovementSuggestion(
                    category="vocabulary",
                    original="good",
                    improved="excellent",
                    reason="More sophisticated vocabulary",
                    confidence=0.9
                )
            ]
        )
        llm.analyze_grammar.return_value = analysis
        
        # 문법 설명 응답 설정
        grammar_response = LLMResponse(
            content="""1. Grammar rule: Adjectives vs participles
2. Explanation: Use -ed for feelings, -ing for characteristics
3. Examples: I am interested, The book is interesting
4. Difficulty level: intermediate""",
            model="test-model"
        )
        llm.generate_response.return_value = grammar_response
        
        return llm
    
    def test_full_learning_workflow(self, mock_llm_with_responses):
        """전체 학습 워크플로우 테스트."""
        assistant = LearningAssistant(mock_llm_with_responses, user_language="korean")
        text = "I am interesting in good music"
        
        # 1. 영어 분석
        analysis = assistant.analyze_user_english(text)
        assert len(analysis.grammar_errors) >= 1
        assert analysis.fluency_score == 0.7
        
        # 2. 교정 제안
        corrections = assistant.provide_corrections(text, analysis)
        assert len(corrections) >= 1
        assert corrections[0].original_text == "I am interesting"
        
        # 3. 어휘 제안
        vocab_suggestions = assistant.suggest_vocabulary_improvements(text, analysis)
        assert len(vocab_suggestions) >= 1
        
        # 4. 종합 피드백
        feedback = assistant.create_learning_feedback(text)
        assert feedback.has_feedback()
        assert len(feedback.corrections) >= 1
        assert len(feedback.vocabulary_suggestions) >= 1
        assert feedback.encouragement != ""
    
    def test_error_handling_in_workflow(self, mock_llm_with_responses):
        """워크플로우에서 오류 처리 테스트."""
        assistant = LearningAssistant(mock_llm_with_responses, user_language="korean")
        
        # LLM 오류 시뮬레이션
        mock_llm_with_responses.analyze_grammar.side_effect = LanguageModelError("API error")
        
        with pytest.raises(LearningAssistantError):
            assistant.analyze_user_english("test text")
        
        # 다른 메서드들도 적절히 오류를 처리해야 함
        with pytest.raises(LearningAssistantError):
            assistant.create_learning_feedback("test text")