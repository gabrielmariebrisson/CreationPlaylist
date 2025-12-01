.PHONY: help build up down logs shell test clean dev lint format lint-check install-dev

# Variables
IMAGE_NAME = music-playlist-generator
CONTAINER_NAME = music-playlist-generator
PORT = 8501

help: ## Affiche cette aide
	@echo "Commandes disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Construit l'image Docker
	docker build -t $(IMAGE_NAME) .

build-no-cache: ## Construit l'image Docker sans cache
	docker build --no-cache -t $(IMAGE_NAME) .

up: ## Lance le conteneur en production
	docker-compose up -d

down: ## Arrête le conteneur
	docker-compose down

logs: ## Affiche les logs du conteneur
	docker-compose logs -f

shell: ## Ouvre un shell dans le conteneur
	docker-compose exec playlist-generator bash

dev: ## Lance le conteneur en mode développement (hot reload)
	docker-compose -f docker-compose.dev.yml up

dev-build: ## Construit l'image pour le développement
	docker-compose -f docker-compose.dev.yml build

dev-down: ## Arrête le conteneur de développement
	docker-compose -f docker-compose.dev.yml down

restart: ## Redémarre le conteneur
	docker-compose restart

status: ## Affiche le statut des conteneurs
	docker-compose ps

health: ## Vérifie la santé du conteneur
	docker-compose ps
	@curl -f http://localhost:$(PORT)/_stcore/health || echo "❌ Health check failed"

clean: ## Nettoie les conteneurs et images
	docker-compose down -v
	docker rmi $(IMAGE_NAME) || true

clean-all: ## Nettoie tout (conteneurs, images, volumes)
	docker-compose down -v --rmi all
	docker system prune -f

test: ## Lance les tests dans le conteneur
	docker-compose exec playlist-generator pytest tests/ -v

install: ## Installation locale (sans Docker)
	pip install -r requirements.txt
	pip install -r tests/requirements-test.txt

run: ## Lance l'application localement
	streamlit run CreationPlaylist.py

# Commandes de linting et formatage
install-dev: ## Installe les dépendances de développement (black, ruff, mypy)
	pip install black ruff mypy types-requests types-python-dateutil

format: ## Formate le code avec black
	black src/ tests/ CreationPlaylist.py

lint-check: ## Vérifie le formatage et le linting sans modifier (pour CI)
	@echo "🔍 Vérification du formatage avec Black..."
	black --check --diff src/ tests/ CreationPlaylist.py
	@echo "🔍 Vérification du linting avec Ruff..."
	ruff check src/ tests/ CreationPlaylist.py --output-format=github
	@echo "🔍 Vérification des types avec MyPy..."
	mypy src/
	@echo "✅ Toutes les vérifications sont passées !"

lint: ## Lance toutes les vérifications de linting (format, lint, type)
	@echo "🔍 Vérification du formatage avec Black..."
	black --check --diff src/ tests/ CreationPlaylist.py || (echo "❌ Black: exécutez 'make format' pour corriger" && exit 1)
	@echo "🔍 Vérification du linting avec Ruff..."
	ruff check src/ tests/ CreationPlaylist.py --output-format=github || (echo "❌ Ruff: exécutez 'make lint-fix' pour corriger" && exit 1)
	@echo "🔍 Vérification des types avec MyPy..."
	mypy src/ || (echo "❌ MyPy: corrigez les erreurs de typage" && exit 1)
	@echo "✅ Toutes les vérifications sont passées !"

lint-fix: ## Corrige automatiquement les problèmes de formatage et de linting
	@echo "🔧 Formatage du code avec Black..."
	black src/ tests/ CreationPlaylist.py
	@echo "🔧 Correction automatique avec Ruff..."
	ruff check --fix src/ tests/ CreationPlaylist.py
	@echo "✅ Corrections appliquées ! (Vérifiez MyPy manuellement avec 'mypy src/')"
