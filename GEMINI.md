# Gemini Project Context: Document RAG English Study

## 1. Project Overview

This project is a CLI program for English learning that builds a RAG (Retrieval-Augmented Generation) system based on documents the user is interested in. The core goal is to provide a natural and engaging English learning experience by leveraging content from the user's own areas of interest. Users can naturally improve their English skills by having conversations in English about topics they care about.

## 2. Core Functionality (User Requirements)

- **Personalized Learning Environment**: Users can specify a directory containing their documents (txt, pdf, docx, md) to build a personalized RAG-based learning environment.
- **Flexible LLM Integration**: Users can connect their preferred LLM provider, with support for OpenAI, Google Gemini, and local models via Ollama.
- **Native Language Support**: Users can set their native language to receive translations, explanations, and feedback in a language they easily understand.
- **Interest-Driven Conversation**: The system initiates conversations based on the content of the user's documents. It continues the dialogue naturally using RAG and provides real-time corrections, explanations, and vocabulary suggestions.
- **Intuitive CLI Interface**: The program offers a user-friendly CLI with a setup guide, clear commands (`setup`, `set-docs`, `set-llm`, `chat`, `status`, `help`), and helpful error messages.

## 3. Architecture and Technology

### 3.1. Technology Stack

- **Language**: Python 3.8+
- **CLI Framework**: Click
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers
- **Language Models**: OpenAI API, Google Gemini API, Ollama
- **Document Parsers**: PyPDF2 (PDF), python-docx (DOCX)
- **Configuration**: YAML

### 3.2. High-Level Architecture

The system is composed of several key layers:
- **CLI Interface**: Handles user commands and interaction.
- **Core Application**: The central orchestrator that connects all components.
- **Document Manager**: Manages document parsing and indexing.
- **RAG Engine**: Handles embedding generation, vector storage (ChromaDB), and content retrieval. It uses an LLM to generate answers based on retrieved context.
- **Conversation Engine**: Manages the dialogue flow, provides learning assistance (grammar correction, vocabulary), and tracks session progress.
- **Configuration Manager**: Manages user settings from a YAML file.

## 4. Project Status

Based on the implementation plan (`tasks.md`), the project is in a relatively mature state.
- **Completed**: Most core functionalities are implemented, including project setup, data modeling, configuration management, document processing, LLM integration (OpenAI, Gemini, Ollama), RAG engine, conversation engine, and the CLI. Initial unit tests have been written.
- **In Progress / To-Do**:
  - **Integration Testing**: Requires more comprehensive testing of the full RAG and conversation pipelines.
  - **Performance Optimization**: Needs final review for performance improvements in indexing and response time.
  - **Final Documentation**: README and guides are in place but may need updates.

## 5. Development Rules & Conventions

**These are critical rules to follow when modifying the codebase.**

### 5.1. Git Workflow
- Create a new branch for each task: `task/<task-number>-<description>`.
- Commit frequently with meaningful messages, using prefixes like `feat:` or `fix:`.
- Merge back to `main` only when a task is fully complete.

### 5.2. Package and Dependency Management
- **Use `uv` for all Python package management.** This includes creating virtual environments and installing dependencies.
- **Do not install new packages directly.** Request new dependencies from the developer, specifying the package name and its purpose.

### 5.3. Code Quality & Style
- **Type hints are mandatory** for all functions and classes.
- **All code comments must be written in Korean (한글).** This is a strict project rule for readability.
- Write Google-style docstrings.
- Implement proper error handling and logging.

### 5.4. Testing
- Use the `pytest` framework.
- Write unit tests for new features to maintain high test coverage.
- **Crucial Check before running tests**:
  - If a new model class is added, ensure it is exported in `src/document_rag_english_study/models/__init__.py`.
  - If a new module is added, ensure it is properly exposed in the corresponding package's `__init__.py`.

### 5.5. Project Structure
- The project follows a `src/` directory layout.
- Maintain clear separation of concerns between modules.
