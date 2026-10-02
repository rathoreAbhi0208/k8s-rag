#!/bin/bash

set -e

echo "Creating Local RAG project structure..."

# Knowledge base
mkdir -p knowledge-base

# Source code
mkdir -p src/ingestion/loaders
mkdir -p src/vectorstore
mkdir -p src/retrieval
mkdir -p src/generation
mkdir -p src/rag
mkdir -p src/api/routes
mkdir -p src/database
mkdir -p src/evaluation

# Tests
mkdir -p tests/unit
mkdir -p tests/integration

# Scripts
mkdir -p scripts

# UI
mkdir -p ui

# Data
mkdir -p data/sqlite
mkdir -p data/pdf

# Docker
mkdir -p docker

# Python files
touch src/__init__.py
touch src/config.py

touch src/ingestion/__init__.py
touch src/ingestion/scanner.py
touch src/ingestion/chunker.py
touch src/ingestion/metadata.py
touch src/ingestion/hasher.py
touch src/ingestion/embedder.py
touch src/ingestion/pipeline.py

touch src/ingestion/loaders/__init__.py
touch src/ingestion/loaders/markdown_loader.py
touch src/ingestion/loaders/pdf_loader.py
touch src/ingestion/loaders/yaml_loader.py
touch src/ingestion/loaders/text_loader.py

touch src/vectorstore/__init__.py
touch src/vectorstore/milvus_client.py
touch src/vectorstore/collections.py
touch src/vectorstore/repository.py

touch src/retrieval/__init__.py
touch src/retrieval/semantic_search.py
touch src/retrieval/hybrid_search.py
touch src/retrieval/reranker.py
touch src/retrieval/retriever.py

touch src/generation/__init__.py
touch src/generation/ollama_client.py
touch src/generation/prompts.py
touch src/generation/generator.py

touch src/rag/__init__.py
touch src/rag/pipeline.py

touch src/api/__init__.py
touch src/api/main.py
touch src/api/schemas.py

touch src/api/routes/__init__.py
touch src/api/routes/chat.py
touch src/api/routes/documents.py

touch src/database/__init__.py
touch src/database/sqlite.py
touch src/database/models.py

touch src/evaluation/__init__.py
touch src/evaluation/evaluate.py
touch src/evaluation/metrics.py
touch src/evaluation/dataset.json

# Tests
touch tests/unit/test_chunker.py
touch tests/unit/test_loaders.py
touch tests/unit/test_hasher.py

touch tests/integration/test_milvus.py
touch tests/integration/test_rag.py

# Scripts
touch scripts/ingest.py
touch scripts/query.py
touch scripts/evaluate.py

# UI
touch ui/streamlit_app.py

# Project files
touch docker-compose.yml
touch requirements.txt
touch .env
touch .gitignore
touch README.md

echo ""
echo "Project structure created successfully!"
echo ""

tree . 2>/dev/null || find . -print