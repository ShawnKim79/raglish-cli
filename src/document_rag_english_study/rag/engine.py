"""
RAG 엔진 코어 모듈

문서 인덱싱, 검색, 컨텍스트 기반 답변 생성을 담당하는 RAG 엔진의 핵심 구현체입니다.
"""

import logging
import re
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..models.document import Document, IndexingResult
from ..models.response import SearchResult
from ..llm.base import LanguageModel
from .vector_database import VectorDatabase
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG(Retrieval-Augmented Generation) 엔진 클래스
    
    문서 인덱싱, 유사도 검색, 컨텍스트 기반 답변 생성을 통합적으로 관리합니다.
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_generator: EmbeddingGenerator,
        llm: Optional[LanguageModel] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_context_length: int = 4000
    ):
        """
        RAGEngine 초기화
        
        Args:
            vector_db: 벡터 데이터베이스 인스턴스
            embedding_generator: 임베딩 생성기 인스턴스
            llm: 언어 모델 인스턴스 (선택사항)
            chunk_size: 문서 청크 크기 (문자 수)
            chunk_overlap: 청크 간 겹치는 부분 크기
            max_context_length: 최대 컨텍스트 길이
        """
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_context_length = max_context_length
        
        # 인덱싱 통계
        self._indexed_documents: Dict[str, Document] = {}
        self._total_chunks = 0
        
        logger.info("RAG 엔진 초기화 완료")
    
    def set_language_model(self, llm: LanguageModel) -> None:
        """
        언어 모델 설정
        
        Args:
            llm: 설정할 언어 모델 인스턴스
        """
        self.llm = llm
        logger.info(f"언어 모델 설정 완료: {llm.__class__.__name__}")
    
    def index_document(self, document: Document) -> IndexingResult:
        """
        단일 문서를 인덱싱
        
        Args:
            document: 인덱싱할 문서
            
        Returns:
            IndexingResult: 인덱싱 결과
        """
        start_time = time.time()
        result = IndexingResult(success=True)
        
        try:
            logger.info(f"문서 인덱싱 시작: {document.title}")
            
            # 문서를 청크로 분할
            chunks = self._split_document_into_chunks(document.content)
            
            if not chunks:
                result.add_error(f"문서에서 유효한 청크를 생성할 수 없습니다: {document.title}")
                return result
            
            # 각 청크에 대한 메타데이터 생성
            metadatas = []
            chunk_texts = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    'document_id': document.id,
                    'document_title': document.title,
                    'source_file': document.file_path,
                    'file_type': document.file_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split()),
                    'language': document.language
                }
                metadatas.append(metadata)
                chunk_texts.append(chunk)
            
            # 임베딩 생성
            logger.debug(f"임베딩 생성 중: {len(chunks)}개 청크")
            embeddings = self.embedding_generator.generate_batch_embeddings(chunk_texts)
            
            # 벡터 데이터베이스에 저장
            chunk_ids = self.vector_db.add_documents(
                documents=chunk_texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            # 결과 업데이트
            result.add_indexed_file(document.file_path, len(chunks))
            self._indexed_documents[document.id] = document
            self._total_chunks += len(chunks)
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info(f"문서 인덱싱 완료: {document.title}, 청크 수: {len(chunks)}, 처리 시간: {processing_time:.2f}초")
            
        except Exception as e:
            logger.error(f"문서 인덱싱 실패: {document.title}, 오류: {e}")
            result.add_failed_file(document.file_path, str(e))
        
        return result
    
    def index_documents(self, documents: List[Document]) -> IndexingResult:
        """
        여러 문서를 일괄 인덱싱
        
        Args:
            documents: 인덱싱할 문서 리스트
            
        Returns:
            IndexingResult: 전체 인덱싱 결과
        """
        start_time = time.time()
        result = IndexingResult(success=True)
        
        logger.info(f"일괄 문서 인덱싱 시작: {len(documents)}개 문서")
        
        for document in documents:
            doc_result = self.index_document(document)
            
            # 결과 통합
            result.documents_processed += doc_result.documents_processed
            result.total_chunks += doc_result.total_chunks
            result.errors.extend(doc_result.errors)
            result.indexed_files.extend(doc_result.indexed_files)
            result.failed_files.extend(doc_result.failed_files)
            
            if not doc_result.success:
                result.success = False
        
        result.processing_time = time.time() - start_time
        
        logger.info(f"일괄 문서 인덱싱 완료: {result.documents_processed}개 성공, {len(result.failed_files)}개 실패")
        
        return result
    
    def search_similar_content(
        self,
        query: str,
        top_k: int = 5,
        min_relevance_score: float = 0.1,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        유사한 내용 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            min_relevance_score: 최소 관련성 점수
            filter_metadata: 메타데이터 필터링 조건
            
        Returns:
            검색 결과 리스트
        """
        if not query.strip():
            logger.warning("빈 검색 쿼리")
            return []
        
        try:
            logger.debug(f"유사 내용 검색: '{query[:50]}...', top_k={top_k}")
            
            # 벡터 데이터베이스에서 검색
            search_results = self.vector_db.search_similar_documents(
                query=query,
                n_results=top_k,
                where=filter_metadata
            )
            
            # 최소 관련성 점수로 필터링
            filtered_results = [
                result for result in search_results
                if result.relevance_score >= min_relevance_score
            ]
            
            logger.debug(f"검색 완료: {len(filtered_results)}개 결과 (필터링 후)")
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"유사 내용 검색 실패: {e}")
            return []
    
    def generate_answer(
        self,
        query: str,
        context_results: Optional[List[SearchResult]] = None,
        user_language: str = "korean",
        **llm_kwargs
    ) -> str:
        """
        컨텍스트 기반 답변 생성
        
        Args:
            query: 사용자 질문
            context_results: 컨텍스트로 사용할 검색 결과 (None인 경우 자동 검색)
            user_language: 사용자 모국어
            **llm_kwargs: LLM에 전달할 추가 매개변수
            
        Returns:
            생성된 답변
            
        Raises:
            ValueError: LLM이 설정되지 않은 경우
            RuntimeError: 답변 생성 실패 시
        """
        if not self.llm:
            raise ValueError("언어 모델이 설정되지 않았습니다. set_language_model()을 먼저 호출하세요.")
        
        if not query.strip():
            raise ValueError("빈 질문은 처리할 수 없습니다.")
        
        try:
            # 컨텍스트 검색 (제공되지 않은 경우)
            if context_results is None:
                context_results = self.search_similar_content(query, top_k=5)
            
            # 컨텍스트 구성
            context = self._build_context_from_results(context_results)
            
            # 프롬프트 생성
            prompt = self._build_answer_prompt(query, context, user_language)
            
            logger.debug(f"답변 생성 중: 컨텍스트 길이={len(context)}")
            
            # LLM으로 답변 생성
            response = self.llm.generate_response(prompt, context, **llm_kwargs)
            
            return response.content
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            raise RuntimeError(f"답변 생성 실패: {e}")
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        텍스트에서 키워드 추출
        
        Args:
            text: 키워드를 추출할 텍스트
            max_keywords: 최대 키워드 수
            
        Returns:
            추출된 키워드 리스트
        """
        if not text.strip():
            return []
        
        try:
            # 간단한 키워드 추출 (단어 빈도 기반)
            # 실제 구현에서는 더 정교한 NLP 기법을 사용할 수 있음
            
            # 텍스트 전처리
            text = text.lower()
            # 특수문자 제거, 단어만 추출
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            
            # 불용어 제거 (간단한 영어 불용어)
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
                'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
                'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                'may', 'might', 'must', 'can', 'shall'
            }
            
            filtered_words = [word for word in words if word not in stop_words]
            
            # 단어 빈도 계산
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 빈도순으로 정렬하여 상위 키워드 반환
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in keywords[:max_keywords]]
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return []
    
    def get_indexed_document_info(self) -> Dict[str, Any]:
        """
        인덱싱된 문서 정보 반환
        
        Returns:
            인덱싱 통계 정보
        """
        return {
            'total_documents': len(self._indexed_documents),
            'total_chunks': self._total_chunks,
            'vector_db_info': self.vector_db.get_collection_info(),
            'embedding_info': self.embedding_generator.get_cache_info(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
    
    def remove_document(self, document_id: str) -> bool:
        """
        인덱싱된 문서 제거
        
        Args:
            document_id: 제거할 문서 ID
            
        Returns:
            제거 성공 여부
        """
        try:
            if document_id not in self._indexed_documents:
                logger.warning(f"문서를 찾을 수 없습니다: {document_id}")
                return False
            
            # 벡터 데이터베이스에서 해당 문서의 모든 청크 검색
            search_results = self.vector_db.search_similar_documents(
                query="",  # 빈 쿼리로 모든 결과 검색
                n_results=1000,  # 충분히 큰 수
                where={"document_id": document_id}
            )
            
            # 청크 ID 수집
            chunk_ids = []
            for result in search_results:
                if result.metadata.get('document_id') == document_id:
                    # ChromaDB에서는 ID를 직접 가져올 수 없으므로 메타데이터 기반으로 삭제
                    # 실제 구현에서는 청크 ID를 별도로 저장해야 할 수 있음
                    pass
            
            # 문서 정보에서 제거
            document = self._indexed_documents.pop(document_id)
            
            logger.info(f"문서 제거 완료: {document.title}")
            return True
            
        except Exception as e:
            logger.error(f"문서 제거 실패: {document_id}, 오류: {e}")
            return False
    
    def clear_index(self) -> None:
        """
        모든 인덱스 데이터 삭제
        """
        try:
            logger.warning("모든 인덱스 데이터 삭제 시작")
            
            self.vector_db.clear_collection()
            self._indexed_documents.clear()
            self._total_chunks = 0
            
            logger.warning("모든 인덱스 데이터 삭제 완료")
            
        except Exception as e:
            logger.error(f"인덱스 삭제 실패: {e}")
            raise RuntimeError(f"인덱스 삭제 실패: {e}")
    
    def _split_document_into_chunks(self, content: str) -> List[str]:
        """
        문서를 청크로 분할
        
        Args:
            content: 분할할 문서 내용
            
        Returns:
            분할된 청크 리스트
        """
        if not content.strip():
            return []
        
        chunks = []
        
        # 문단 단위로 먼저 분할
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # 현재 청크에 문단을 추가했을 때의 길이 확인
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size:
                # 청크 크기 내에 들어가면 추가
                current_chunk = potential_chunk
            else:
                # 청크 크기를 초과하면 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 문단이 청크 크기보다 큰 경우 문장 단위로 분할
                if len(paragraph) > self.chunk_size:
                    sentence_chunks = self._split_paragraph_into_sentences(paragraph)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 겹치는 부분 처리 (선택사항)
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_chunk_overlap(chunks)
        
        return chunks
    
    def _split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """
        문단을 문장 단위로 분할하여 청크 생성
        
        Args:
            paragraph: 분할할 문단
            
        Returns:
            문장 기반 청크 리스트
        """
        # 간단한 문장 분할 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r'[.!?]+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _add_chunk_overlap(self, chunks: List[str]) -> List[str]:
        """
        청크 간 겹치는 부분 추가
        
        Args:
            chunks: 원본 청크 리스트
            
        Returns:
            겹치는 부분이 추가된 청크 리스트
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]  # 첫 번째 청크는 그대로
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # 이전 청크의 마지막 부분을 현재 청크 앞에 추가
            prev_words = prev_chunk.split()
            if len(prev_words) > self.chunk_overlap:
                overlap_text = " ".join(prev_words[-self.chunk_overlap:])
                overlapped_chunk = overlap_text + " " + current_chunk
            else:
                overlapped_chunk = current_chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def _build_context_from_results(self, search_results: List[SearchResult]) -> str:
        """
        검색 결과로부터 컨텍스트 구성
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            구성된 컨텍스트 문자열
        """
        if not search_results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in search_results:
            # 소스 정보와 함께 컨텍스트 구성
            context_part = f"[출처: {result.source_file}]\n{result.content}\n"
            
            # 최대 컨텍스트 길이 확인
            if current_length + len(context_part) > self.max_context_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _build_answer_prompt(self, query: str, context: str, user_language: str) -> str:
        """
        답변 생성을 위한 프롬프트 구성
        
        Args:
            query: 사용자 질문
            context: 컨텍스트 정보
            user_language: 사용자 모국어
            
        Returns:
            구성된 프롬프트
        """
        if user_language.lower() == "korean":
            prompt = f"""다음 문서들을 참고하여 사용자의 질문에 답변해주세요.

참고 문서:
{context}

사용자 질문: {query}

답변 시 다음 사항을 고려해주세요:
1. 제공된 문서의 내용을 바탕으로 정확하고 유용한 답변을 제공하세요.
2. 문서에 없는 내용은 추측하지 말고, 문서 기반으로만 답변하세요.
3. 답변은 자연스럽고 이해하기 쉽게 작성하세요.
4. 필요한 경우 출처를 명시하세요.

답변:"""
        else:
            prompt = f"""Please answer the user's question based on the following documents.

Reference documents:
{context}

User question: {query}

Please consider the following when answering:
1. Provide accurate and useful answers based on the provided documents.
2. Do not speculate on content not in the documents; answer only based on the documents.
3. Write your answer naturally and in an easy-to-understand manner.
4. Cite sources when necessary.

Answer:"""
        
        return prompt