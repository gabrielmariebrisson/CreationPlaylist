# Dockerfile pour Music Playlist Generator
# Multi-stage build pour optimiser la taille de l'image

# Stage 1: Build dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python depuis le builder
COPY --from=builder /root/.local /root/.local

# Ajouter les dépendances au PATH
ENV PATH=/root/.local/bin:$PATH

# Copier le code source
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p logs templates/assets/music

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Exposer le port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande par défaut
CMD ["streamlit", "run", "CreationPlaylist.py", "--server.port=8501", "--server.address=0.0.0.0"]

