# Docker Configuration

Ce dossier contient la configuration Docker pour l'application Music Playlist Generator.

## Fichiers

- `Dockerfile` : Image Docker pour la production
- `docker-compose.yml` : Configuration pour la production
- `docker-compose.dev.yml` : Configuration pour le développement avec hot reload
- `.dockerignore` : Fichiers exclus du build Docker

## Utilisation

### Production

```bash
# Build l'image
docker build -t music-playlist-generator .

# Lancer le conteneur
docker run -d \
  -p 8501:8501 \
  -e CLIENT_ID_SPOTIFY=your_client_id \
  -e CLIENT_SECRET_SPOTIFY=your_client_secret \
  -e REFRESH_TOKEN_SPOTIFY=your_refresh_token \
  --name playlist-generator \
  music-playlist-generator

# Ou utiliser docker-compose
docker-compose up -d
```

### Développement

```bash
# Lancer avec hot reload
docker-compose -f docker-compose.dev.yml up

# Rebuild après changement de dépendances
docker-compose -f docker-compose.dev.yml build --no-cache
```

### Commandes utiles

```bash
# Voir les logs
docker-compose logs -f

# Arrêter
docker-compose down

# Rebuild
docker-compose build --no-cache

# Accéder au shell du conteneur
docker-compose exec playlist-generator bash

# Vérifier la santé
docker-compose ps
```

## Variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
CLIENT_ID_SPOTIFY=your_client_id
CLIENT_SECRET_SPOTIFY=your_client_secret
REFRESH_TOKEN_SPOTIFY=your_refresh_token
REDIRECT_URI_SPOTIFY=http://localhost:8501
```

## Optimisations

- **Multi-stage build** : Réduit la taille de l'image finale
- **Layer caching** : Accélère les rebuilds
- **Healthcheck** : Vérifie automatiquement la santé du conteneur
- **Volumes** : Persiste les logs et cache

## Troubleshooting

### Le conteneur ne démarre pas
```bash
docker-compose logs playlist-generator
```

### Port déjà utilisé
Changez le port dans `docker-compose.yml` :
```yaml
ports:
  - "8502:8501"  # Utilise le port 8502 au lieu de 8501
```

### Problèmes de permissions
```bash
sudo chown -R $USER:$USER logs/
```

