"""
벡터 데이터베이스 모듈

ChromaDB를 활용한 문서 청크 저장 및 유사도 검색 기능을 제공합니다.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..models.response import SearchResult

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    ChromaDB를 사용한 벡터 데이터베이스 클래스
    
    문서 청크를 벡터로 저장하고 유사도 기반 검색을 제공합니다.
    """
    
    def __init__(
        self,
        collection_name: str = "document_chunks",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        VectorDatabase 초기화
        
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 데이터베이스 저장 디렉토리 경로
            embedding_function: 임베딩 함수 (None인 경우 기본 함수 사용)
        """
        self.collection_name = collection_name
        
        # 저장 디렉토리 설정
        if persist_directory:
            self.persist_directory = Path(persist_directory)
        else:
            self.persist_directory = Path.home() / ".cache" / "document_rag_english_study" / "chroma_db"
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB 클라이언트 초기화
        try:
            logger.info(f"ChromaDB 클라이언트 초기화 중: {self.persist_directory}")
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB 클라이언트 초기화 완료")
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 실패: {e}")
            raise
        
        # 임베딩 함수 설정 (기본값으로 sentence-transformers 사용)
        if embedding_function is None:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            self.embedding_function = embedding_function
        
        # 컬렉션 생성 또는 가져오기
        self._initialize_collection()
    
    def _initialize_collection(self) -> None:
        """
        ChromaDB 컬렉션 초기화
        """
        try:
            # 기존 컬렉션이 있는지 확인
            existing_collections = [col.name for col in self.client.list_collections()]
            
            if self.collection_name in existing_collections:
                logger.info(f"기존 컬렉션 사용: {self.collection_name}")
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function
                )
            else:
                logger.info(f"새 컬렉션 생성: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
                )
            
            logger.info(f"컬렉션 초기화 완료: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        문서 청크들을 벡터 데이터베이스에 추가
        
        Args:
            documents: 추가할 문서 텍스트 리스트
            metadatas: 각 문서의 메타데이터 리스트
            ids: 문서 ID 리스트 (None인 경우 자동 생성)
            embeddings: 미리 계산된 임베딩 벡터 리스트 (선택사항)
            
        Returns:
            추가된 문서들의 ID 리스트
            
        Raises:
            ValueError: 입력 데이터가 유효하지 않은 경우
            RuntimeError: 문서 추가 실패 시
        """
        if not documents:
            raise ValueError("추가할 문서가 없습니다")
        
        if len(documents) != len(metadatas):
            raise ValueError("문서 수와 메타데이터 수가 일치하지 않습니다")
        
        # ID 자동 생성
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("문서 수와 ID 수가 일치하지 않습니다")
        
        # 임베딩이 제공된 경우 길이 확인
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError("문서 수와 임베딩 수가 일치하지 않습니다")
        
        try:
            logger.info(f"문서 추가 시작: {len(documents)}개")
            
            # ChromaDB에 문서 추가
            if embeddings is not None:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"문서 추가 완료: {len(documents)}개")
            return ids
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise RuntimeError(f"문서 추가 실패: {e}")
    
    def search_similar_documents(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[SearchResult]:
        """
        유사한 문서 검색
        
        Args:
            query: 검색 쿼리 텍스트
            n_results: 반환할 결과 수
            where: 메타데이터 필터링 조건
            query_embedding: 미리 계산된 쿼리 임베딩 (선택사항)
            
        Returns:
            검색 결과 리스트 (SearchResult 객체들)
            
        Raises:
            ValueError: 쿼리가 유효하지 않은 경우
            RuntimeError: 검색 실패 시
        """
        if not query.strip() and query_embedding is None:
            raise ValueError("검색 쿼리 또는 임베딩이 필요합니다")
        
        if n_results <= 0:
            raise ValueError("결과 수는 1 이상이어야 합니다")
        
        try:
            logger.debug(f"유사도 검색 시작: '{query[:50]}...', 결과 수: {n_results}")
            
            # ChromaDB에서 검색 수행
            if query_embedding is not None:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where
                )
            else:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=where
                )
            
            # SearchResult 객체로 변환
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    # 거리를 유사도 점수로 변환 (코사인 거리 -> 유사도)
                    # 코사인 거리는 0(완전 유사)~2(완전 반대) 범위
                    # 유사도 점수는 1(완전 유사)~0(완전 반대) 범위로 변환
                    relevance_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                    
                    search_result = SearchResult(
                        content=doc,
                        source_file=metadata.get('source_file', 'unknown'),
                        relevance_score=relevance_score,
                        metadata=metadata
                    )
                    search_results.append(search_result)
            
            logger.debug(f"유사도 검색 완료: {len(search_results)}개 결과")
            return search_results
            
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            raise RuntimeError(f"유사도 검색 실패: {e}")
    
    def update_document(
        self,
        document_id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """
        기존 문서 업데이트
        
        Args:
            document_id: 업데이트할 문서 ID
            document: 새로운 문서 텍스트 (선택사항)
            metadata: 새로운 메타데이터 (선택사항)
            embedding: 새로운 임베딩 벡터 (선택사항)
            
        Raises:
            ValueError: 문서 ID가 유효하지 않은 경우
            RuntimeError: 업데이트 실패 시
        """
        if not document_id.strip():
            raise ValueError("문서 ID가 필요합니다")
        
        try:
            logger.debug(f"문서 업데이트: {document_id}")
            
            # 업데이트할 데이터 준비
            update_data = {"ids": [document_id]}
            
            if document is not None:
                update_data["documents"] = [document]
            
            if metadata is not None:
                update_data["metadatas"] = [metadata]
            
            if embedding is not None:
                update_data["embeddings"] = [embedding]
            
            # ChromaDB에서 업데이트
            self.collection.update(**update_data)
            
            logger.debug(f"문서 업데이트 완료: {document_id}")
            
        except Exception as e:
            logger.error(f"문서 업데이트 실패: {document_id}, 오류: {e}")
            raise RuntimeError(f"문서 업데이트 실패: {e}")
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        문서들을 데이터베이스에서 삭제
        
        Args:
            document_ids: 삭제할 문서 ID 리스트
            
        Raises:
            ValueError: 문서 ID가 유효하지 않은 경우
            RuntimeError: 삭제 실패 시
        """
        if not document_ids:
            raise ValueError("삭제할 문서 ID가 없습니다")
        
        try:
            logger.info(f"문서 삭제 시작: {len(document_ids)}개")
            
            self.collection.delete(ids=document_ids)
            
            logger.info(f"문서 삭제 완료: {len(document_ids)}개")
            
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            raise RuntimeError(f"문서 삭제 실패: {e}")
    
    def get_document_count(self) -> int:
        """
        저장된 문서 수 반환
        
        Returns:
            저장된 문서 수
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"문서 수 조회 실패: {e}")
            return 0
    
    def clear_collection(self) -> None:
        """
        컬렉션의 모든 문서 삭제
        
        Raises:
            RuntimeError: 삭제 실패 시
        """
        try:
            logger.warning("컬렉션 전체 삭제 시작")
            
            # 컬렉션 삭제 후 재생성
            self.client.delete_collection(name=self.collection_name)
            self._initialize_collection()
            
            logger.warning("컬렉션 전체 삭제 완료")
            
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            raise RuntimeError(f"컬렉션 삭제 실패: {e}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 반환
        
        Returns:
            컬렉션 통계 정보
        """
        try:
            document_count = self.get_document_count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": document_count,
                "persist_directory": str(self.persist_directory),
                "embedding_function": str(type(self.embedding_function)) if self.embedding_function else "default"
            }
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "persist_directory": str(self.persist_directory),
                "embedding_function": "unknown",
                "error": str(e)
            }