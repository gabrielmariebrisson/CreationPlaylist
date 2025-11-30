.PHONY: help build up down logs shell test clean dev

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

