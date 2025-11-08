# Makefile para Cogitura

.PHONY: help install dev-install test lint format clean docker-up docker-down setup

help:
	@echo "Comandos disponíveis:"
	@echo "  make install       - Instala o projeto"
	@echo "  make dev-install   - Instala com dependências de desenvolvimento"
	@echo "  make test          - Executa testes"
	@echo "  make test-cov      - Executa testes com cobertura"
	@echo "  make lint          - Verifica código com flake8"
	@echo "  make format        - Formata código com black"
	@echo "  make clean         - Remove arquivos temporários"
	@echo "  make docker-up     - Inicia ElasticSearch"
	@echo "  make docker-down   - Para ElasticSearch"
	@echo "  make setup         - Setup completo do projeto"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=cogitura --cov-report=html --cov-report=term

lint:
	flake8 src/cogitura tests/
	mypy src/cogitura

format:
	black src/cogitura tests/ scripts/
	isort src/cogitura tests/ scripts/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

docker-up:
	docker-compose up -d
	@echo "Aguardando ElasticSearch iniciar..."
	@sleep 10
	@curl -s http://localhost:9200 > /dev/null && echo "ElasticSearch está rodando!" || echo "Erro ao iniciar ElasticSearch"

docker-down:
	docker-compose down

setup: clean install docker-up
	@echo "Setup completo!"
	@echo "Configure o arquivo .env com suas chaves de API"
	@echo "Execute: cogitura config-check"

run-example:
	@echo "Gerando 100 sentenças de exemplo..."
	cogitura generate --count 100 --save-db --generate-tts
	cogitura db-stats

analyze:
	python scripts/analyze_data.py

full-pipeline:
	python scripts/run_full_pipeline.py
