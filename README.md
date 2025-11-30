# 🎵 Music Playlist Generator

Application web intelligente de génération de playlists musicales basée sur l'Intelligence Artificielle. Le système analyse les caractéristiques audio des morceaux à l'aide d'un réseau de neurones convolutif (CNN) et génère des playlists cohérentes en utilisant la similarité cosinus dans l'espace des embeddings.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Méthodologie](#méthodologie)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Tests](#tests)
- [Auteurs](#auteurs)

## 🎯 Vue d'ensemble

Ce projet propose une solution complète pour la génération automatique de playlists musicales en utilisant :

- **Deep Learning** : Classification des genres musicaux avec un CNN pré-entraîné
- **Feature Extraction** : Extraction d'embeddings audio (1536 dimensions)
- **Pathfinding Intelligent** : Génération de playlists progressives via interpolation dans l'espace des features et recherche par similarité cosinus
- **Intégration Spotify** : Export direct des playlists générées vers Spotify

## 🏗️ Architecture

Le projet suit une architecture modulaire respectant le principe de responsabilité unique (SRP) :

### Frontend : Streamlit

- **Interface utilisateur** : Application web interactive avec onglets multiples
- **Visualisation** : Graphiques Plotly pour la visualisation 2D (PCA/t-SNE)
- **Multilingue** : Support de 10 langues via traduction automatique

### Backend : PyTorch

- **Modèle CNN** : Architecture `SimpleCNN` avec modules CBAM (Convolutional Block Attention Module)
- **Classification** : Prédiction de 10 genres musicaux (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock)
- **Feature Extraction** : Extraction d'embeddings de 1536 dimensions avant la couche de classification

### Services

- **SpotifyService** : Gestion de l'authentification OAuth et des opérations API Spotify
- **AudioClassifier** : Chargement du modèle PyTorch et prédiction de genres
- **PlaylistPathfinder** : Génération de playlists basée sur la similarité cosinus

### Structure modulaire

```
src/
├── config.py              # Configuration centralisée
├── models/
│   └── audio_classifier.py    # Classification audio avec CNN
├── services/
│   └── spotify_service.py     # Service API Spotify
└── logic/
    └── playlist_generator.py  # Génération de playlist
```

## 🔬 Méthodologie

### 1. Pré-traitement Audio

Les fichiers audio sont convertis en **Spectrogrammes de Mel** :
- Durée d'extraction : 30 secondes
- Taille du spectrogramme : 599 × 128
- Normalisation pour optimiser l'entraînement

### 2. Classification par CNN

Le modèle CNN analyse les spectrogrammes et :
- **Extrait des features** : 1536 dimensions avant la couche de classification
- **Prédit le genre** : Probabilités pour 10 genres musicaux
- **Calcule la confiance** : Score de confiance de la prédiction

**Architecture du modèle** :
- 4 couches convolutionnelles avec BatchNorm et ReLU
- Modules CBAM pour l'attention
- Couches fully-connected pour la classification finale

### 3. Génération de Playlist

#### Approche innovante : Embeddings bruts + Cosine Similarity

Contrairement aux approches classiques qui utilisent la PCA 2D (perte d'information), notre système :

1. **Interpolation dans l'espace des features brutes** (1536 dimensions)
   - Préservation de toute l'information musicale
   - Interpolation linéaire entre deux morceaux de référence

2. **Recherche par similarité cosinus**
   - Plus adapté aux embeddings normalisés que la distance euclidienne
   - Capture mieux les similarités directionnelles entre morceaux

3. **PCA uniquement pour la visualisation**
   - Réduction à 2D uniquement pour l'affichage graphique
   - Les calculs de similarité utilisent les features complètes

#### Algorithme de Pathfinding

```
Pour chaque point interpolé dans l'espace 1536D :
    1. Calculer la similarité cosinus avec tous les morceaux disponibles
    2. Sélectionner le morceau le plus similaire (non utilisé)
    3. Ajouter à la playlist
    4. Répéter jusqu'à obtenir num_tracks morceaux
```

### 4. Visualisation

- **PCA 2D** : Projection pour visualisation interactive
- **t-SNE** : Alternative pour une meilleure séparation des clusters
- **Graphiques Plotly** : Visualisation interactive des playlists générées

## 📦 Installation

### Prérequis

- Python 3.10 ou supérieur
- pip ou conda
- Compte Spotify Developer (pour l'export de playlists)

### Installation des dépendances

#### Avec pip

```bash
# Cloner le repository (si applicable)
git clone <repository-url>
cd CreationPlaylist

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

#### Avec conda

```bash
# Créer un environnement conda
conda create -n Creationplaylist python=3.10
conda activate Creationplaylist

# Installer les dépendances
pip install -r requirements.txt
# ou
conda install --file requirements.txt -c conda-forge
```

### Dépendances principales

- **Streamlit** : Interface web
- **PyTorch** : Deep learning
- **librosa** : Traitement audio
- **scikit-learn** : PCA, t-SNE, preprocessing
- **spotipy** : API Spotify
- **pandas** : Manipulation de données
- **plotly** : Visualisation interactive

## ⚙️ Configuration

### Variables d'environnement

Créez un fichier `.env` à la racine du projet avec les variables suivantes :

```bash
# Spotify API Credentials
CLIENT_ID_SPOTIFY=votre_client_id_spotify
CLIENT_SECRET_SPOTIFY=votre_client_secret_spotify
REFRESH_TOKEN_SPOTIFY=votre_refresh_token_spotify
REDIRECT_URI_SPOTIFY=http://localhost:8501
```

### Obtenir les credentials Spotify

1. **Créer une application Spotify** :
   - Allez sur [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Créez une nouvelle application
   - Récupérez le `CLIENT_ID` et `CLIENT_SECRET`

2. **Configurer les Redirect URIs** :
   - Dans les paramètres de l'application, ajoutez `http://localhost:8501` aux Redirect URIs

3. **Obtenir un Refresh Token** :
   - Utilisez le script d'authentification Spotify OAuth
   - Ou suivez le [guide officiel Spotify](https://developer.spotify.com/documentation/web-api/tutorials/getting-started)

### Exemple de fichier `.env`

```env
# Spotify Configuration
CLIENT_ID_SPOTIFY=abc123def456ghi789jkl012mno345pq
CLIENT_SECRET_SPOTIFY=xyz789uvw456rst123opq012klm345
REFRESH_TOKEN_SPOTIFY=AQA...votre_refresh_token...xyz
REDIRECT_URI_SPOTIFY=http://localhost:8501
```

**⚠️ Important** : Ne commitez jamais le fichier `.env` dans le repository. Il est déjà dans `.gitignore`.

## 🚀 Utilisation

### Lancer l'application

```bash
# Activer l'environnement (si vous utilisez venv/conda)
conda activate Creationplaylist  # ou source venv/bin/activate

# Lancer Streamlit
streamlit run CreationPlaylist.py
```

L'application sera accessible sur `http://localhost:8501`

### Workflow d'utilisation

1. **Recherche de morceaux** :
   - Utilisez l'onglet "🔍 Recherche" pour trouver des morceaux sur Deezer
   - Analysez les morceaux pour extraire leurs features audio

2. **Analyse des genres** :
   - Consultez l'onglet "📊 Analyse" pour voir la distribution des genres
   - Visualisez les morceaux dans l'espace 2D (PCA ou t-SNE)

3. **Génération de playlist** :
   - Allez dans l'onglet "🎨 Playlist"
   - Choisissez le mode :
     - **Transition progressive** : Sélectionnez deux morceaux et générez une playlist entre eux
     - **Par genre** : Créez une playlist basée sur un ou plusieurs genres

4. **Export vers Spotify** :
   - Nommez votre playlist
   - Cliquez sur "🎵 Créer la playlist sur Spotify"
   - La playlist sera créée dans votre compte Spotify

## 📁 Structure du projet

```
CreationPlaylist/
├── CreationPlaylist.py          # Application Streamlit principale
├── requirements.txt              # Dépendances Python
├── pytest.ini                   # Configuration pytest
├── .env                         # Variables d'environnement (à créer)
│
├── src/                         # Code source modulaire
│   ├── config.py                # Configuration centralisée
│   ├── models/
│   │   └── audio_classifier.py  # Classification audio
│   ├── services/
│   │   └── spotify_service.py   # Service Spotify
│   └── logic/
│       └── playlist_generator.py # Génération de playlist
│
├── templates/                   # Assets statiques
│   └── assets/
│       ├── images/              # Images de documentation
│       └── music/
│           ├── architecture.py  # Architecture du modèle CNN
│           └── best_model_original_loss.pth  # Modèle pré-entraîné
│
└── tests/                       # Tests unitaires
    ├── test_playlist_generator.py
    ├── conftest.py
    └── requirements-test.txt
```

## 🧪 Tests

### Installation des dépendances de test

```bash
pip install -r tests/requirements-test.txt
```

### Exécution des tests

```bash
# Tous les tests
pytest tests/ -v

# Avec couverture de code
pytest tests/ --cov=src --cov-report=html

# Un fichier spécifique
pytest tests/test_playlist_generator.py -v
```

### Couverture des tests

Les tests couvrent :
- ✅ Initialisation et configuration
- ✅ Calcul de similarité cosinus
- ✅ Réduction de dimensionnalité (PCA, t-SNE)
- ✅ Génération de playlist avec différentes longueurs
- ✅ Gestion des erreurs (indices invalides, colonnes manquantes)
- ✅ Analyse de qualité de playlist

## 🎓 Méthodologie technique détaillée

### Pipeline complet

```
Audio (30s) 
  → Spectrogramme de Mel (599×128)
  → CNN (SimpleCNN avec CBAM)
  → Embeddings (1536 dimensions)
  → Interpolation linéaire dans l'espace 1536D
  → Recherche par similarité cosinus
  → Playlist générée
  → Export Spotify (optionnel)
```

### Avantages de l'approche

1. **Préservation de l'information** : Utilisation des embeddings bruts (1536D) au lieu de PCA 2D
2. **Similarité adaptée** : Cosine similarity plus appropriée pour les embeddings normalisés
3. **Interpolation précise** : Interpolation dans l'espace complet des features
4. **Visualisation séparée** : PCA uniquement pour l'affichage, pas pour les calculs

### Performance du modèle

- **Précision globale** : ~73% sur le dataset GTZAN
- **Meilleure performance** : Blues (100%), Metal (90%)
- **Entraînement** : 30 minutes sur MacBook M1

## 👥 Auteurs

- **Gabriel Marie-Brisson** - [Portfolio](https://gabriel.mariebrisson.fr)
- Clément Delmas
- Thibault Pottier
- Aurélien Gauthier

**Enseignant référent** : Charles Brazier

## 📚 Références

- [Spotify CNNs - Sander Dieleman](https://sander.ai/2014/08/05/spotify-cnns.html)
- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html)
- [Spotify Web API Documentation](https://developer.spotify.com/documentation/web-api)

## 📄 License

Ce projet est développé dans le cadre d'un projet universitaire.

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue.

---

**Note** : Ce projet utilise un modèle pré-entraîné sur le dataset GTZAN (10 genres, 100 morceaux). Pour de meilleures performances, il serait recommandé d'entraîner sur un dataset plus large et diversifié.

