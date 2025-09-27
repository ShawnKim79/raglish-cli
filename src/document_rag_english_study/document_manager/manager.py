"""
Document manager for handling document indexing and management operations.

This module provides the DocumentManager class that orchestrates document
scanning, parsing, and indexing operations for the RAG English Study system.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .parser import DocumentParser, DocumentParsingError
from ..models.document import (
    Document, 
    IndexingResult, 
    IndexingStatus, 
    DocumentSummary
)
from ..utils import (
    get_logger, DocumentError, error_handler_decorator,
    retry_on_error
)


logger = get_logger(__name__)


# DocumentManagerError는 DocumentError로 대체됨


class DocumentManager:
    """문서 인덱싱 및 관리를 담당하는 클래스.
    
    주요 기능:
    - 디렉토리 스캔 및 문서 발견
    - 문서 일괄 인덱싱
    - 인덱싱 상태 추적 및 진행률 표시
    - 문서 요약 정보 제공
    - 인덱싱된 문서 메타데이터 관리
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """DocumentManager 인스턴스를 초기화합니다.
        
        Args:
            storage_path: 인덱싱된 문서 메타데이터를 저장할 경로
        """
        self.parser = DocumentParser()
        self.storage_path = Path(storage_path) if storage_path else Path("data/documents")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 인덱싱 상태 관리
        self._indexing_status = IndexingStatus()
        self._status_lock = threading.Lock()
        
        # 인덱싱된 문서 메타데이터 캐시
        self._documents_cache: Dict[str, Document] = {}
        self._cache_lock = threading.Lock()
        
        # 진행률 콜백 함수
        self._progress_callback: Optional[Callable[[IndexingStatus], None]] = None
        
        # 메타데이터 파일 경로
        self.metadata_file = self.storage_path / "documents_metadata.json"
        
        # 기존 메타데이터 로드
        self._load_metadata()
    
    def set_progress_callback(self, callback: Callable[[IndexingStatus], None]) -> None:
        """인덱싱 진행률 콜백 함수를 설정합니다.
        
        Args:
            callback: 진행률 업데이트 시 호출될 콜백 함수
        """
        self._progress_callback = callback
    
    @error_handler_decorator(context={"operation": "set_document_directory"})
    def set_document_directory(self, directory_path: str) -> IndexingResult:
        """문서 디렉토리를 설정하고 인덱싱을 수행합니다.
        
        Args:
            directory_path: 인덱싱할 문서들이 있는 디렉토리 경로
            
        Returns:
            인덱싱 결과 정보
            
        Raises:
            DocumentError: 디렉토리 설정 또는 인덱싱 실패 시
        """
        logger = get_logger(__name__)
        try:
            logger.info(f"Setting document directory: {directory_path}")
            
            # 디렉토리 유효성 검사
            directory = Path(directory_path)
            if not directory.exists():
                raise DocumentError(f"Directory does not exist: {directory_path}", file_path=directory_path)
            
            if not directory.is_dir():
                raise DocumentError(f"Path is not a directory: {directory_path}", file_path=directory_path)
            
            # 인덱싱 수행
            result = self.index_documents(directory_path)
            
            if result.success:
                logger.info(f"Successfully set document directory: {directory_path}")
            else:
                logger.error(f"Failed to index documents in directory: {directory_path}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to set document directory {directory_path}: {str(e)}"
            logger.error(error_msg)
            raise DocumentError(error_msg, file_path=directory_path) from e
    
    @error_handler_decorator(context={"operation": "index_documents"})
    @retry_on_error(max_retries=2, delay=1.0)
    def index_documents(self, directory_path: str, max_workers: int = 4) -> IndexingResult:
        """디렉토리 내의 모든 지원되는 문서를 인덱싱합니다.
        
        Args:
            directory_path: 인덱싱할 디렉토리 경로
            max_workers: 병렬 처리에 사용할 최대 워커 수
            
        Returns:
            인덱싱 결과 정보
            
        Raises:
            DocumentError: 인덱싱 실패 시
        """
        start_time = time.time()
        result = IndexingResult(success=True)
        
        try:
            logger.info(f"Starting document indexing for directory: {directory_path}")
            
            # 지원되는 파일들 스캔
            supported_files = self._scan_directory(directory_path)
            
            if not supported_files:
                logger.warning(f"No supported files found in directory: {directory_path}")
                result.processing_time = time.time() - start_time
                return result
            
            # 인덱싱 상태 초기화
            with self._status_lock:
                self._indexing_status = IndexingStatus(
                    is_indexing=True,
                    total_documents=len(supported_files),
                    processed_documents=0,
                    start_time=datetime.now()
                )
            
            self._notify_progress()
            
            # 병렬 처리로 문서 인덱싱
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 각 파일에 대한 Future 생성
                future_to_file = {
                    executor.submit(self._index_single_document, file_path): file_path
                    for file_path in supported_files
                }
                
                # 완료된 작업들 처리
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        document = future.result()
                        if document:
                            # 성공적으로 인덱싱된 문서 처리
                            self._add_document_to_cache(document)
                            result.add_indexed_file(file_path, 1)  # 청크 수는 추후 구현
                            logger.info(f"Successfully indexed: {file_path}")
                        else:
                            result.add_failed_file(file_path, "Failed to parse document")
                            
                    except Exception as e:
                        error_msg = str(e)
                        result.add_failed_file(file_path, error_msg)
                        logger.error(f"Failed to index {file_path}: {error_msg}")
                    
                    # 진행률 업데이트
                    with self._status_lock:
                        self._indexing_status.processed_documents += 1
                        self._indexing_status.current_file = file_path
                    
                    self._notify_progress()
            
            # 인덱싱 완료 처리
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # 메타데이터 저장
            self._save_metadata()
            
            # 인덱싱 상태 완료로 변경
            with self._status_lock:
                self._indexing_status.is_indexing = False
                self._indexing_status.current_file = None
            
            self._notify_progress()
            
            logger.info(f"Document indexing completed. "
                       f"Processed: {result.documents_processed}, "
                       f"Failed: {len(result.failed_files)}, "
                       f"Time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            # 인덱싱 상태 리셋
            with self._status_lock:
                self._indexing_status.is_indexing = False
                self._indexing_status.current_file = None
            
            error_msg = f"Document indexing failed: {str(e)}"
            logger.error(error_msg)
            result.add_error(error_msg)
            result.processing_time = time.time() - start_time
            
            raise DocumentError(error_msg, file_path=directory_path) from e
    
    def _scan_directory(self, directory_path: str) -> List[str]:
        """디렉토리를 스캔하여 지원되는 파일들을 찾습니다.
        
        Args:
            directory_path: 스캔할 디렉토리 경로
            
        Returns:
            지원되는 파일 경로들의 리스트
        """
        supported_files = []
        directory = Path(directory_path)
        
        try:
            # 재귀적으로 모든 파일 스캔
            for file_path in directory.rglob("*"):
                if file_path.is_file() and self.parser.is_supported_file(str(file_path)):
                    supported_files.append(str(file_path))
            
            logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
            return supported_files
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory_path}: {e}")
            return []
    
    def _index_single_document(self, file_path: str) -> Optional[Document]:
        """단일 문서를 인덱싱합니다.
        
        Args:
            file_path: 인덱싱할 파일 경로
            
        Returns:
            인덱싱된 Document 객체 또는 None (실패 시)
        """
        try:
            # 이미 인덱싱된 문서인지 확인 (파일 해시 기반)
            existing_doc = self._get_existing_document(file_path)
            if existing_doc:
                logger.debug(f"Document already indexed: {file_path}")
                return existing_doc
            
            # 문서 파싱
            document = self.parser.parse_file(file_path)
            return document
            
        except DocumentParsingError as e:
            logger.error(f"Parsing error for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error indexing {file_path}: {e}")
            return None
    
    def _get_existing_document(self, file_path: str) -> Optional[Document]:
        """기존에 인덱싱된 문서가 있는지 확인합니다.
        
        Args:
            file_path: 확인할 파일 경로
            
        Returns:
            기존 Document 객체 또는 None
        """
        try:
            # 파일 해시 생성
            current_hash = self.parser._generate_file_hash(file_path)
            
            # 캐시에서 검색
            with self._cache_lock:
                for doc in self._documents_cache.values():
                    if doc.file_path == str(Path(file_path).absolute()) and doc.file_hash == current_hash:
                        return doc
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking existing document {file_path}: {e}")
            return None
    
    def _add_document_to_cache(self, document: Document) -> None:
        """문서를 캐시에 추가합니다.
        
        Args:
            document: 추가할 Document 객체
        """
        with self._cache_lock:
            self._documents_cache[document.id] = document
    
    def _notify_progress(self) -> None:
        """진행률 콜백 함수를 호출합니다."""
        if self._progress_callback:
            try:
                with self._status_lock:
                    status_copy = IndexingStatus(
                        is_indexing=self._indexing_status.is_indexing,
                        total_documents=self._indexing_status.total_documents,
                        processed_documents=self._indexing_status.processed_documents,
                        current_file=self._indexing_status.current_file,
                        start_time=self._indexing_status.start_time,
                        estimated_completion=self._indexing_status.estimated_completion
                    )
                
                self._progress_callback(status_copy)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    def get_indexing_status(self) -> IndexingStatus:
        """현재 인덱싱 상태를 반환합니다.
        
        Returns:
            현재 인덱싱 상태 정보
        """
        with self._status_lock:
            return IndexingStatus(
                is_indexing=self._indexing_status.is_indexing,
                total_documents=self._indexing_status.total_documents,
                processed_documents=self._indexing_status.processed_documents,
                current_file=self._indexing_status.current_file,
                start_time=self._indexing_status.start_time,
                estimated_completion=self._indexing_status.estimated_completion
            )
    
    def get_document_summary(self) -> DocumentSummary:
        """인덱싱된 문서들의 요약 정보를 반환합니다.
        
        Returns:
            문서 요약 정보
        """
        summary = DocumentSummary()
        
        with self._cache_lock:
            for document in self._documents_cache.values():
                summary.add_document(document)
        
        return summary
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """문서 ID로 문서를 조회합니다.
        
        Args:
            document_id: 조회할 문서 ID
            
        Returns:
            Document 객체 또는 None
        """
        with self._cache_lock:
            return self._documents_cache.get(document_id)
    
    def get_all_documents(self) -> List[Document]:
        """모든 인덱싱된 문서를 반환합니다.
        
        Returns:
            인덱싱된 모든 Document 객체들의 리스트
        """
        with self._cache_lock:
            return list(self._documents_cache.values())
    
    def search_documents(self, query: str, limit: int = 10) -> List[Document]:
        """문서 제목이나 내용에서 키워드를 검색합니다.
        
        Args:
            query: 검색할 키워드
            limit: 반환할 최대 문서 수
            
        Returns:
            검색 결과 Document 리스트
        """
        results = []
        query_lower = query.lower()
        
        with self._cache_lock:
            for document in self._documents_cache.values():
                # 제목이나 내용에서 검색
                if (query_lower in document.title.lower() or 
                    query_lower in document.content.lower()):
                    results.append(document)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def remove_document(self, document_id: str) -> bool:
        """문서를 인덱스에서 제거합니다.
        
        Args:
            document_id: 제거할 문서 ID
            
        Returns:
            제거 성공 여부
        """
        try:
            removed = False
            with self._cache_lock:
                if document_id in self._documents_cache:
                    del self._documents_cache[document_id]
                    logger.info(f"Removed document: {document_id}")
                    removed = True
                else:
                    logger.warning(f"Document not found for removal: {document_id}")
                    return False
            
            # 락 해제 후 메타데이터 저장
            if removed:
                self._save_metadata()
            
            return removed
                    
        except Exception as e:
            logger.error(f"Error removing document {document_id}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """모든 인덱싱된 문서를 제거합니다.
        
        Returns:
            제거 성공 여부
        """
        try:
            with self._cache_lock:
                self._documents_cache.clear()
            
            # 메타데이터 파일 삭제
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            
            logger.info("Cleared all indexed documents")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
    
    def _save_metadata(self) -> None:
        """문서 메타데이터를 파일에 저장합니다."""
        try:
            # 락 범위를 최소화
            with self._cache_lock:
                documents_data = {
                    doc_id: doc.to_dict() 
                    for doc_id, doc in self._documents_cache.items()
                }
                total_documents = len(self._documents_cache)
            
            # 락 해제 후 파일 I/O 수행
            metadata = {
                'documents': documents_data,
                'last_updated': datetime.now().isoformat(),
                'total_documents': total_documents
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved metadata for {total_documents} documents")
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _load_metadata(self) -> None:
        """저장된 문서 메타데이터를 로드합니다."""
        try:
            if not self.metadata_file.exists():
                logger.debug("No existing metadata file found")
                return
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 문서 데이터 로드
            documents_data = metadata.get('documents', {})
            
            with self._cache_lock:
                self._documents_cache.clear()
                for doc_id, doc_data in documents_data.items():
                    try:
                        document = Document.from_dict(doc_data)
                        self._documents_cache[doc_id] = document
                    except Exception as e:
                        logger.error(f"Error loading document {doc_id}: {e}")
                        continue
            
            logger.info(f"Loaded metadata for {len(self._documents_cache)} documents")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
    
    def get_manager_info(self) -> Dict[str, Any]:
        """문서 관리자의 현재 상태 정보를 반환합니다.
        
        Returns:
            관리자 상태 정보 딕셔너리
        """
        with self._cache_lock:
            document_count = len(self._documents_cache)
        
        with self._status_lock:
            is_indexing = self._indexing_status.is_indexing
        
        return {
            'storage_path': str(self.storage_path),
            'metadata_file': str(self.metadata_file),
            'document_count': document_count,
            'is_indexing': is_indexing,
            'parser_info': self.parser.get_parser_info(),
            'last_metadata_update': self.metadata_file.stat().st_mtime if self.metadata_file.exists() else None
        }