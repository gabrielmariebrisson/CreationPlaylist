import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from io import BytesIO
import tempfile
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime
import matplotlib.pyplot as plt
load_dotenv()

# Imports des nouvelles classes modulaires
from src.services.spotify_service import SpotifyService
from src.models.audio_classifier import AudioClassifier
from src.logic.playlist_generator import PlaylistPathfinder
from src.config import FEATURE_VIEW_SIZE

from deep_translator import GoogleTranslator

# --- Configuration de la traduction automatique ---
LANGUAGES = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ja": "🇯🇵 日本語",
    "zh-CN": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский"
}

# Initialisation de la langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# Sélecteur de langue
lang = st.sidebar.selectbox(
    "🌐 Language / Langue", 
    options=list(LANGUAGES.keys()),
    format_func=lambda x: LANGUAGES[x],
    index=list(LANGUAGES.keys()).index(st.session_state.language)
)

st.session_state.language = lang

# Cache pour les traductions (évite de retranduire à chaque fois)
if 'translations_cache' not in st.session_state:
    st.session_state.translations_cache = {}

def _(text):
    """Fonction de traduction automatique avec cache"""
    if lang == 'fr':
        return text
    
    # Vérifier le cache
    cache_key = f"{lang}_{text}"
    if cache_key in st.session_state.translations_cache:
        return st.session_state.translations_cache[cache_key]
    
    # Traduire
    try:
        translated = GoogleTranslator(source='fr', target=lang).translate(text)
        st.session_state.translations_cache[cache_key] = translated
        return translated
    except:
        return text
    

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Music Playlist Generator",
    page_icon="🎵",
    layout="wide"
)

# --- CONSTANTES ET CONFIGURATION ---
LABEL_MAPPING = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
    5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
}

GENRE_COLORS = {
    'blues': '#1f77b4', 'classical': '#ff7f0e', 'country': '#2ca02c',
    'disco': '#d62728', 'hiphop': '#9467bd', 'jazz': '#8c564b',
    'metal': '#e377c2', 'pop': '#7f7f7f', 'reggae': '#bcbd22', 'rock': '#17becf'
}

SPOTIFY_SCOPE = "user-library-read user-top-read playlist-modify-public playlist-modify-private user-read-recently-played"
DEEZER_BASE_URL = "https://api.deezer.com"


# --- INITIALISATION DES SERVICES ---
@st.cache_resource
def init_audio_classifier(model_path):
    """Initialise le classifieur audio avec cache Streamlit."""
    try:
        return AudioClassifier(model_path)
    except Exception as e:
        st.error(f"Erreur chargement modèle: {str(e)}")
        return None

@st.cache_resource
def init_spotify_service():
    """Initialise le service Spotify avec cache Streamlit."""
    return SpotifyService(session_state=st.session_state)

@st.cache_resource
def init_playlist_pathfinder():
    """Initialise le générateur de playlist avec cache Streamlit."""
    return PlaylistPathfinder()


# --- FONCTIONS DEEZER ---
def search_deezer_tracks(query, limit=10):
    """Recherche des tracks sur Deezer"""
    try:
        url = f"{DEEZER_BASE_URL}/search"
        params = {
            'q': query,
            'limit': limit
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            tracks = []
            
            for item in data.get('data', []):
                track = {
                    'id': item.get('id'),
                    'name': item.get('title'),
                    'artists': item.get('artist', {}).get('name', ''),
                    'preview_url': item.get('preview'),
                    'album': item.get('album', {}).get('title', ''),
                    'duration': item.get('duration'),
                    'deezer_id': item.get('id')
                }
                tracks.append(track)
            
            return tracks
        else:
            st.error(f"Erreur API Deezer: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erreur recherche Deezer: {str(e)}")
        return []

def download_deezer_preview(preview_url, output_path):
    """Télécharge l'extrait audio de 30s depuis Deezer"""
    try:
        if not preview_url:
            return False
            
        response = requests.get(preview_url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Erreur téléchargement Deezer: {str(e)}")
        return False


# --- ARCHITECTURE DU MODÈLE ---
# Les fonctions suivantes sont maintenant gérées par AudioClassifier
# convert_song_to_matrix, load_model, extract_features, analyze_audio_genre

def analyze_audio_genre(audio_path, classifier):
    """
    Wrapper pour compatibilité avec l'ancien code.
    Analyse un fichier audio et prédit son genre.
    
    Args:
        audio_path: Chemin vers le fichier audio
        classifier: Instance d'AudioClassifier (ou None pour compatibilité)
    
    Returns:
        Tuple (genre, confidence, features, probabilities) ou (None, None, None, None)
    """
    if classifier is None:
        genre_id = np.random.randint(0, 10)
        confidence = np.random.uniform(0.6, 0.95)
        return LABEL_MAPPING[genre_id], confidence, None, None
    
    try:
        result = classifier.predict(audio_path, return_features=True, return_probabilities=True)
        return result['genre'], result['confidence'], result.get('features'), result.get('probabilities')
    except Exception as e:
        st.error(f"Erreur prédiction: {str(e)}")
        return None, None, None, None

# Les fonctions suivantes sont maintenant gérées par PlaylistPathfinder
# perform_pca, perform_tsne, perform_dimensionality_reduction, generate_playlist_line

def perform_pca(features_list, pathfinder):
    """Wrapper pour compatibilité - Effectue une PCA sur les features extraites"""
    return pathfinder.perform_pca(features_list)

def perform_tsne(features_list, pathfinder, random_state=42, perplexity=30):
    """Wrapper pour compatibilité - Effectue une t-SNE sur les features extraites"""
    return pathfinder.perform_tsne(features_list, random_state=random_state, perplexity=perplexity)

def perform_dimensionality_reduction(features_list, pathfinder, method='pca', **kwargs):
    """Wrapper pour compatibilité - Effectue une réduction de dimensionnalité (PCA ou t-SNE)"""
    return pathfinder.perform_dimensionality_reduction(features_list, method=method, **kwargs)

def generate_playlist_line(pca_df, track1_idx, track2_idx, pathfinder, num_tracks=10):
    """
    Wrapper pour compatibilité - Génère une playlist progressive (linéaire) entre deux morceaux.
    Utilise maintenant les features brutes avec cosine similarity au lieu de PCA.
    """
    try:
        # Extraire les features brutes depuis analyzed_tracks
        valid_tracks = [t for t in st.session_state.analyzed_tracks if t.get('features') is not None]
        
        if len(valid_tracks) < 2:
            st.error("Pas assez de tracks avec features pour générer une playlist")
            return None, None, None, None
        
        # Créer un mapping multi-clés pour trouver les features
        # On utilise plusieurs clés car les IDs peuvent varier
        features_dict = {}
        for t in valid_tracks:
            feature = t.get('features')
            if feature is not None:
                # Créer des entrées pour différentes clés possibles
                keys = [
                    str(t.get('deezer_id', '')),
                    str(t.get('spotify_id', '')),
                    str(t.get('name', '')).lower(),
                    str(t.get('track_id', ''))
                ]
                for key in keys:
                    if key:
                        features_dict[key] = feature
        
        # Extraire les features brutes dans le même ordre que pca_df
        raw_features_list = []
        missing_features = []
        
        for idx, row in pca_df.iterrows():
            # Chercher la feature correspondante avec plusieurs stratégies
            feature = None
            
            # Stratégie 1: Par deezer_id
            deezer_id = row.get('deezer_id')
            if deezer_id and str(deezer_id) in features_dict:
                feature = features_dict[str(deezer_id)]
            
            # Stratégie 2: Par spotify_id
            if feature is None:
                spotify_id = row.get('spotify_id')
                if spotify_id and str(spotify_id) in features_dict:
                    feature = features_dict[str(spotify_id)]
            
            # Stratégie 3: Par nom (normalisé)
            if feature is None:
                name = str(row.get('name', '')).lower()
                if name in features_dict:
                    feature = features_dict[name]
            
            # Stratégie 4: Recherche directe dans analyzed_tracks
            if feature is None:
                name = row.get('name', '')
                for t in valid_tracks:
                    if (t.get('name') == name or 
                        t.get('deezer_id') == deezer_id or 
                        t.get('spotify_id') == spotify_id):
                        feature = t.get('features')
                        if feature is not None:
                            break
            
            if feature is None:
                missing_features.append(row.get('name', 'Unknown'))
                # Utiliser un vecteur zéro comme fallback (ne devrait pas arriver normalement)
                feature = np.zeros(FEATURE_VIEW_SIZE)
            
            # S'assurer que la feature est un array numpy
            if not isinstance(feature, np.ndarray):
                feature = np.array(feature)
            
            raw_features_list.append(feature)
        
        if missing_features:
            st.warning(f"Features non trouvées pour {len(missing_features)} track(s): {missing_features[:3]}")
        
        # Convertir en array numpy
        raw_features = np.array(raw_features_list)
        
        # Vérifier la dimension
        if len(raw_features) == 0:
            st.error("Aucune feature trouvée")
            return None, None, None, None
        
        # S'assurer que le pathfinder a le modèle PCA chargé pour la visualisation
        if pathfinder.pca_model is None:
            # Recalculer la PCA pour la visualisation
            features_list = [t['features'] for t in valid_tracks if t.get('features') is not None]
            if len(features_list) >= 2:
                pathfinder.perform_pca(features_list)
        
        # Utiliser la nouvelle méthode avec features brutes
        return pathfinder.generate_playlist_line_from_pca_df(
            pca_df=pca_df,
            raw_features=raw_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks
        )
    except Exception as e:
        st.error(f"Erreur génération playlist: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None

def visualize_playlist_transition(pca_df, playlist, line_points, p1, p2, track1_id, track2_id, genre_names):
    plt.figure(figsize=(14, 10))

    colors = plt.cm.Set3(np.linspace(0, 1, len(genre_names)))
    for i, genre in enumerate(genre_names):
        mask = pca_df["genre"] == genre
        genre_label = LABEL_MAPPING.get(genre)
        plt.scatter(
            pca_df[mask]["PC1"], 
            pca_df[mask]["PC2"], 
            c=[colors[genre]], 
            label=genre_label, 
            alpha=0.3, 
            s=30
        )

    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=2, alpha=0.7, label="Playlist Line")

    plt.scatter(
        line_points[:, 0], line_points[:, 1], 
        c="red", s=50, alpha=0.5, marker="x", label="Target Points"
    )

    plt.scatter([p1[0]], [p1[1]], c="blue", s=200, marker="*", 
                edgecolors="black", linewidth=2, label=f"Start: {track1_id}")
    plt.scatter([p2[0]], [p2[1]], c="green", s=200, marker="*", 
                edgecolors="black", linewidth=2, label=f"End: {track2_id}")

    playlist_points = np.array([[track["PC1"], track["PC2"]] for track in playlist])
    plt.scatter(
        playlist_points[:, 0], playlist_points[:, 1], 
        c="red", s=100, alpha=0.8, edgecolors="black", linewidth=1, label="Playlist Tracks"
    )

    for track in playlist:
        genre_label = LABEL_MAPPING.get(track["genre"], track.get("genre", "Unknown"))
        plt.annotate(
            genre_label,  # afficher le label au lieu du numéro
            (track["PC1"], track["PC2"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold"
        )

    for track in playlist:
        target = track["target_point"]
        actual = track["actual_point"]
        plt.plot([target[0], actual[0]], [target[1], actual[1]], "r-", alpha=0.3, linewidth=1)

    plt.xlabel(f"PC1 (variance)") # PCA model not directly available here, so removing variance percentage
    plt.ylabel(f"PC2 (variance)")
    plt.title(f"Playlist Generation: {track1_id} -> {track2_id}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()



def analyze_playlist_quality(playlist, pathfinder):
    """
    Wrapper pour compatibilité - Analyse la qualité d'une playlist générée
    Compatible avec mode transition ET mode genre
    """
    try:
        return pathfinder.analyze_playlist_quality(playlist)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")
        return None

# --- FONCTIONS SPOTIFY ---
# Les fonctions suivantes sont maintenant gérées par SpotifyService
# get_spotify_oauth, get_spotify_client, export_playlist_to_spotify

def get_spotify_oauth():
    """Wrapper pour compatibilité - OAuth pour Spotify"""
    spotify_service = init_spotify_service()
    return spotify_service.auth_manager

def get_spotify_client():
    """Wrapper pour compatibilité - Client Spotify avec refresh token permanent depuis env"""
    spotify_service = init_spotify_service()
    client = spotify_service.get_client()
    if not client:
        st.error(_("❌ REFRESH_TOKEN_SPOTIFY manquant dans .env"))
    return client

def export_playlist_to_spotify(spotify_service, playlist_tracks, playlist_name, playlist_description=""):
    """Wrapper pour compatibilité - Exporte une playlist vers Spotify avec recherche automatique des URIs manquants"""
    if not spotify_service:
        st.error(_("❌ Service Spotify non disponible"))
        return None
    
    def callback_info(msg):
        st.info(_(msg))
    
    def callback_warning(msg):
        st.warning(_(msg))
    
    def callback_success(msg):
        st.success(_(msg))
    
    def callback_error(msg):
        st.error(_(msg))
    
    result = spotify_service.export_playlist(
        playlist_tracks,
        playlist_name,
        playlist_description,
        callback_info=callback_info,
        callback_warning=callback_warning,
        callback_success=callback_success,
        callback_error=callback_error
    )
    
    if result:
        st.markdown(f"[🎵 Ouvrir dans Spotify]({result['external_urls']['spotify']})")
    
    return result


# --- FONCTIONS D'ANALYSE ---
def find_deezer_track_from_spotify(track_name, artist_name):
    """
    Recherche un morceau sur à partir des infos Spotify
    Retourne le preview_url si trouvé
    """
    try:
        # Construire la requête de recherche
        query = f"{artist_name} {track_name}"
        
        url = f"{DEEZER_BASE_URL}/search"
        params = {
            'q': query,
            'limit': 5  # Prendre les 5 premiers résultats
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            
            # Parcourir les résultats pour trouver la meilleure correspondance
            for item in data.get('data', []):
                deezer_title = item.get('title', '').lower()
                deezer_artist = item.get('artist', {}).get('name', '').lower()
                
                # Vérification simple de correspondance
                if (track_name.lower() in deezer_title or deezer_title in track_name.lower()) and \
                   (artist_name.lower() in deezer_artist or deezer_artist in artist_name.lower()):
                    
                    preview_url = item.get('preview')
                    if preview_url:
                        return preview_url, item.get('id')
            
            # Si aucune correspondance exacte, prendre le premier résultat avec preview
            for item in data.get('data', []):
                preview_url = item.get('preview')
                if preview_url:
                    return preview_url, item.get('id')
        
        return None, None
        
    except Exception as e:
        st.warning(f"Erreur recherche: {str(e)}")
        return None, None
    
def match_deezer_to_spotify(track_name, artist_name, spotify_service):
    """Wrapper pour compatibilité - Trouve le track Spotify correspondant à un track Deezer"""
    if not spotify_service:
        return None
    return spotify_service.match_deezer_to_spotify(track_name, artist_name)

def process_track_analysis(track, track_data):
    """Traite l'analyse d'une track (recherche automatique sur Deezer si Spotify)"""
    try:
        classifier = st.session_state.audio_classifier
        if classifier:
            with st.spinner(_("Analyse en cours...")):
                temp_dir = tempfile.mkdtemp()
                audio_path = os.path.join(temp_dir, f"{track_data['type']}_{track_data['index']}.mp3")
                
                preview_url = None
                deezer_id = None
                
                # ✅ Si c'est une track Deezer, utiliser directement son preview
                if track_data['type'] in ['deezer', 'search_deezer']:
                    preview_url = track.get('preview_url')
                    deezer_id = track.get('deezer_id')
                    
                    if not preview_url:
                        st.warning(_("⚠️ Aucun aperçu audio disponible sur Deezer"))
                        return False
                
                # ✅ Si c'est une track Spotify, rechercher sur Deezer
                else:
                    st.info(_("🔍 Recherche de l'extrait audio sur Deezer..."))
                    
                    # Extraire le nom de l'artiste (peut être une string ou une liste)
                    artists = track.get('artists', '')
                    if isinstance(artists, list):
                        artist_name = ', '.join([artist['name'] for artist in artists])
                    else:
                        artist_name = artists
                    
                    # Rechercher sur Deezer
                    preview_url, deezer_id = find_deezer_track_from_spotify(
                        track['name'], 
                        artist_name
                    )
                    
                    if not preview_url:
                        st.error(_("❌ Impossible de trouver cet extrait"))
                        st.info(_("💡 Astuce : Essayez de rechercher directement dans l'onglet dédié"))
                        return False
                    
                    st.success(_("✅ Extrait trouvé !"))
                
                # Télécharger l'extrait Deezer
                download_success = download_deezer_preview(preview_url, audio_path)
                
                if download_success:
                    # Analyser le fichier audio
                    genre, confidence, features, probs = analyze_audio_genre(
                        audio_path, classifier
                    )
                    
                    if genre:
                        artists = track.get('artists', '')
                        if isinstance(artists, list):
                            artists = ', '.join([artist['name'] for artist in artists])
                        
                        # Matcher avec Spotify si c'est une track Deezer
                        spotify_match = None
                        if track_data['type'] in ['deezer', 'search_deezer']:
                            spotify_service = st.session_state.spotify_service
                            if spotify_service:
                                spotify_match = match_deezer_to_spotify(
                                    track['name'], 
                                    artists, 
                                    spotify_service
                                )
                        
                        # Créer l'objet track analysé
                        analyzed_track = {
                            'name': track['name'],
                            'artists': artists,
                            'spotify_id': spotify_match['spotify_id'] if spotify_match else track.get('id'),
                            'deezer_id': deezer_id,
                            'uri': spotify_match['uri'] if spotify_match else track.get('uri'),
                            'preview_url': preview_url,
                            'source': 'deezer' if track_data['type'] in ['deezer', 'search_deezer'] else 'spotify',
                            'genre': genre,
                            'confidence': confidence,
                            'features': features,
                            'probabilities': probs
                        }
                        
                        if spotify_match:
                            st.info(_(f"🎵 Trouvé sur Spotify: {spotify_match['name']}"))
                        
                        st.session_state.analyzed_tracks.append(analyzed_track)
                        st.success(_(f"✅ {track['name']} - {genre} ({confidence:.1%})"))
                        return True
                    else:
                        st.error(_("❌ Erreur lors de la prédiction du genre"))
                        return False
                else:
                    st.error(_("❌ Impossible de télécharger l'aperçu audio"))
                    return False
        else:
            st.warning(_("⚠️ Modèle non chargé"))
            return False
            
    except Exception as e:
        st.error(_(f"❌ Erreur lors de l'analyse: {e}"))
        import traceback
        st.error(traceback.format_exc())
        return False

def process_track_addition(track, source='spotify'):
    """
    Ajoute une track à la playlist, en réutilisant l'analyse existante si disponible.
    - st.session_state.analyzed_tracks : contient les analyses complètes (genre, confidence, etc.)
    - st.session_state.playlist_tracks : contient les morceaux ajoutés à la playlist finale
    """
    try:
        # --- Vérifier si le morceau est déjà dans la playlist ---
        already_in_playlist = any(
            t.get('deezer_id') == track.get('id') or t['name'].lower() == track['name'].lower()
            for t in st.session_state.playlist_tracks
        )
        if already_in_playlist:
            st.info(_(f"✅ '{track['name']}' est déjà dans la playlist."))
            return False

        # --- Vérifier si le morceau a déjà été analysé ---
        existing_analysis = next(
            (t for t in st.session_state.analyzed_tracks
             if t.get('deezer_id') == track.get('id')
             or t.get('spotify_id') == track.get('id')
             or t['name'].lower() == track['name'].lower()),
            None
        )

        artists = track.get('artists', '')
        if isinstance(artists, list):
            artists = ', '.join([artist['name'] for artist in artists])

        is_deezer = track.get('deezer_id') is not None or source == 'deezer'

        # --- Si on a déjà une analyse : réutiliser les infos ---
        if existing_analysis:
            playlist_entry = {
                'name': existing_analysis['name'],
                'artists': existing_analysis['artists'],
                'spotify_id': existing_analysis.get('spotify_id'),
                'deezer_id': existing_analysis.get('deezer_id'),
                'uri': existing_analysis.get('uri'),
                'preview_url': existing_analysis.get('preview_url'),
                'source': existing_analysis.get('source'),
                'genre': existing_analysis.get('genre'),
                'confidence': existing_analysis.get('confidence'),
                'features': existing_analysis.get('features', {}),
                'probabilities': existing_analysis.get('probabilities', {})
            }

            st.session_state.playlist_tracks.append(playlist_entry)
            st.success(_(f"✅ '{track['name']}' ajouté à la playlist depuis les analyses existantes."))
            return True

        # --- Sinon, ajout brut sans analyse ---
        track_data = {
            'name': track['name'],
            'artists': artists,
            'spotify_id': track.get('id') if not is_deezer else None,
            'deezer_id': track.get('deezer_id'),
            'uri': track.get('uri'),
            'preview_url': track.get('preview_url'),
            'source': 'deezer' if is_deezer else 'spotify',
            'genre': 'Non analysé',
            'confidence': 0.0,
            'features': {},
            'probabilities': {}
        }

        st.session_state.playlist_tracks.append(track_data)
        st.success(_(f"✅ '{track['name']}' ajouté à la playlist (non analysé)."))
        return True

    except Exception as e:
        st.error(_(f"❌ Erreur lors de l'ajout du morceau: {e}"))
        import traceback
        st.error(traceback.format_exc())
        return False


# Bouton de redirection
st.markdown(
    f"""
    <a href="https://gabriel.mariebrisson.fr" target="_blank" style="text-decoration:none;">
    <div style="
    display: inline-block;
    background: linear-gradient(135deg, #6A11CB 0%, #2575FC 100%);
    color: white;
    padding: 12px 25px;
    border-radius: 30px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(37, 117, 252, 0.3);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
    ">
    {_("Retour")}
    <span style="
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.2);
    transform: scaleX(0);
    transform-origin: right;
    transition: transform 0.3s ease;
    z-index: 1;
    "></span>
    </div>
    </a>
    """,
    unsafe_allow_html=True
)


# --- INITIALISATION SESSION STATE ---
if 'audio_classifier' not in st.session_state:
    st.session_state.audio_classifier = None
if 'spotify_service' not in st.session_state:
    st.session_state.spotify_service = None
if 'playlist_pathfinder' not in st.session_state:
    st.session_state.playlist_pathfinder = None
# Compatibilité avec ancien code
if 'model' not in st.session_state:
    st.session_state.model = None
if 'analyzed_tracks' not in st.session_state:
    st.session_state.analyzed_tracks = []
if 'playlist_tracks' not in st.session_state:
    st.session_state.playlist_tracks = []

if 'deezer_search_results' not in st.session_state:
    st.session_state.deezer_search_results = []


if 'generated_playlist' not in st.session_state:
    st.session_state.generated_playlist = None
if 'line_points' not in st.session_state:
    st.session_state.line_points = None
if 'p1' not in st.session_state:
    st.session_state.p1 = None
if 'p2' not in st.session_state:
    st.session_state.p2 = None
if 'pca_df' not in st.session_state:
    st.session_state.pca_df = None
if 'playlist_analysis' not in st.session_state:
    st.session_state.playlist_analysis = None


# --- INTERFACE STREAMLIT PRINCIPALE ---
st.title("🎵 Music Playlist Generator")
st.markdown("Créez des playlists personnalisées avec l'IA - Analyse de genres musicaux par CNN")

# Barre latérale
with st.sidebar:
    # Essayer plusieurs chemins possibles
    model_path = "templates/assets/music/best_model_original_loss.pth"
    
    with st.spinner(_("Chargement...")):
        # Initialiser les services
        st.session_state.audio_classifier = init_audio_classifier(model_path)
        st.session_state.spotify_service = init_spotify_service()
        st.session_state.playlist_pathfinder = init_playlist_pathfinder()
        
        # Compatibilité avec ancien code
        st.session_state.model = st.session_state.audio_classifier
    
    spotify_client = st.session_state.spotify_service.get_client() if st.session_state.spotify_service else None
    
    if spotify_client:
        try:
            user_info = spotify_client.current_user()
            user_name = user_info.get('display_name', user_info.get('id', 'Service'))
        except:
            st.error(_("❌ Erreur connexion"))
    else:
        st.error(_("❌ REFRESH_TOKEN_SPOTIFY manquant"))

# Tabs principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([_("🏠 Accueil"), _("🔍 Recherche"), _("📊 Analyse"), _("🎨 Playlist"), _("🤔 Explications")])

# Tab 1: Accueil
with tab1:
    st.header(_("🎵 Analyseur de Musique"))
    st.markdown(_("""
    Bienvenue dans votre analyseur de musique personnel !
    
    **Fonctionnalités :**
    - 🔍 Recherche et analyse depuis
    - 🔍 Classification automatique par IA
    - 🎨 Création de playlists intelligentes
    """))


with tab2:
    st.header(_("🔍 Recherche"))
    st.markdown(_("Recherchez et analysez des morceaux directement (extraits de 30 secondes)"))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        deezer_query = st.text_input(_("Rechercher un titre:"), placeholder="Nom de la chanson ou artiste...")
    with col2:
        deezer_limit = st.slider(_("Nombre de résultats"), 5, 20, 10, key="deezer_slider")
    
    # Afficher les résultats Deezer
    if deezer_query:
        if st.button(_("🔍 Rechercher"), type="primary"):
            with st.spinner(_("Recherche en cours...")):
                deezer_results = search_deezer_tracks(deezer_query, deezer_limit)
                st.session_state.deezer_search_results = deezer_results

                if deezer_results:
                    st.success(_(f"✅ {len(deezer_results)} titres trouvés!"))
                else:
                    st.warning(_("Aucun titre trouvé."))

        # --- Affichage des résultats Deezer ---
        if st.session_state.deezer_search_results:
            st.subheader(_("Résultats"))

            for i, track in enumerate(st.session_state.deezer_search_results):
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    st.markdown(f"**{track['name']}**")
                    st.caption(_(f"Artiste: {track['artists']}"))
                    if track.get('album'):
                        st.caption(_(f"Album: {track['album']}"))

                with col2:
                    if track.get('preview_url'):
                        st.markdown(_("🎵 **30s**"))
                    else:
                        st.markdown(_("❌ **No preview**"))

                with col3:
                    if st.button(_("🎵 Écouter"), key=f"preview_deezer_{i}"):
                        if track.get('preview_url'):
                            st.audio(track['preview_url'], format="audio/mp3")
                        else:
                            st.warning("Aucun extrait disponible")

                # --- Colonne Analyser ---
                with col4:
                    already_analyzed = any(
                        t['deezer_id'] == track.get('id')
                        for t in st.session_state.analyzed_tracks
                    )
                    if already_analyzed:
                        st.button(_("✅ Analysé"), key=f"analyzed_deezer_{i}", disabled=True)
                    else:
                        if st.button(_("🔍 Analyser"), key=f"analyze_deezer_{i}"):
                            if process_track_analysis(track, {"type": "deezer", "index": i}):
                                st.rerun()

                # --- Colonne Ajouter ---
                with col5:
                    already_in_playlist = any(
                        t['deezer_id'] == track.get('id')
                        for t in st.session_state.playlist_tracks
                    )
                    if already_in_playlist:
                        st.button(_("✅ Ajouté"), key=f"added_playlist_{i}", disabled=True)
                    else:
                        already_analyzed = any(
                            t['deezer_id'] == track.get('id')
                            for t in st.session_state.analyzed_tracks
                        )
                        if already_analyzed:
                            if st.button(_("➕ Ajouter"), key=f"add_playlist_{i}"):
                                if process_track_addition(track, source='deezer'):
                                    st.rerun()
                        else:
                            if st.button(_("➕ Ajouter"), key=f"add_playlist_{i}"):
                                if process_track_analysis(track, {"type": "top", "index": i}) and process_track_addition(track):
                                    st.rerun()


# Tab 4: Analyse
with tab3:
    st.header(_("📊 Analyse des genres"))
    
    if st.session_state.analyzed_tracks:
        valid_tracks = [t for t in st.session_state.analyzed_tracks if t.get('genre') != 'Non analysé']
        
        if valid_tracks:
            df = pd.DataFrame(valid_tracks)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total tracks", len(df))
            col2.metric("Genres uniques", df['genre'].nunique())
            col3.metric("confidence moy.", f"{df['confidence'].mean():.1%}")

            # Distribution des genres
            st.subheader(_("Distribution des genres"))
            genre_counts = df['genre'].value_counts()
            
            fig = px.bar(
                x=genre_counts.index, 
                y=genre_counts.values,
                color=genre_counts.index,
                color_discrete_map=GENRE_COLORS,
                labels={'x': 'Genre', 'y': 'Nombre de tracks'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualisation avec choix de méthode
            if len(valid_tracks) >= 2:
                st.subheader(_("Visualisation 2D"))
                
                # Sélecteur de méthode
                col1, col2 = st.columns([1, 3])
                with col1:
                    reduction_method = st.selectbox(
                        _("Méthode:"),
                        options=['pca', 'tsne'],
                        format_func=lambda x: 'PCA' if x == 'pca' else 't-SNE',
                        key='reduction_method'
                    )
                with col2:
                    if reduction_method == 'tsne':
                        perplexity = st.slider(_("Perplexity (t-SNE):"), 5, 50, 30, key='tsne_perplexity')
                
                features_list = [t['features'] for t in valid_tracks if t.get('features') is not None]
                
                if len(features_list) >= 2:
                    # Calcul de la réduction de dimensionnalité
                    pathfinder = st.session_state.playlist_pathfinder
                    if reduction_method == 'tsne':
                        perplexity_val = st.session_state.get('tsne_perplexity', 30)
                        result, model, scaler = perform_dimensionality_reduction(
                            features_list, 
                            pathfinder,
                            method='tsne',
                            perplexity=perplexity_val
                        )
                    else:
                        result, model, scaler = perform_dimensionality_reduction(
                            features_list, 
                            pathfinder,
                            method='pca'
                        )
                    
                    if result is not None:
                        track_ids = [f"track_{i}" for i in range(len(result))]
                        # Créer le DataFrame avec les résultats
                        pca_df = pd.DataFrame({
                            'PC1': result[:, 0],
                            'PC2': result[:, 1],
                            'name': [t['name'] for t in valid_tracks if t.get('features') is not None],
                            'genre': [t['genre'] for t in valid_tracks if t.get('features') is not None],
                            'artists': [t['artists'] for t in valid_tracks if t.get('features') is not None],
                            'uri': [t.get('uri') for t in valid_tracks if t.get('features') is not None],
                            'spotify_id': [t.get('spotify_id') for t in valid_tracks if t.get('features') is not None],
                            'deezer_id': [t.get('deezer_id') for t in valid_tracks if t.get('features') is not None],
                            'preview_url': [t.get('preview_url') for t in valid_tracks if t.get('features') is not None],
                            'confidence': [t.get('confidence', 0) for t in valid_tracks if t.get('features') is not None],
                            'track_id': track_ids
                        })
                        
                        # Sauvegarder dans session_state
                        st.session_state.pca_df = pca_df
                        
                        # Visualisation
                        method_label = 't-SNE' if reduction_method == 'tsne' else 'PCA'
                        fig = px.scatter(
                            pca_df, 
                            x='PC1', 
                            y='PC2',
                            color='genre',
                            color_discrete_map=GENRE_COLORS,
                            hover_data=['name', 'artists'],
                            title=_(f'Espace {method_label} des tracks')
                        )
                        fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
                        fig.update_layout(
                            height=600,
                            plot_bgcolor='white',
                            xaxis=dict(gridcolor='lightgray', showgrid=True, title=f'{method_label}1'),
                            yaxis=dict(gridcolor='lightgray', showgrid=True, title=f'{method_label}2')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Message de succès
                        st.success(_(f"✅ {len(pca_df)} tracks prêtes pour la création de playlist"))
                        st.info(_("👉 Rendez-vous dans l'onglet '🎨 Playlist' pour créer votre playlist personnalisée"))
                    else:
                        st.error(_(f"❌ Erreur lors du calcul {reduction_method.upper()}"))
                else:
                    st.warning(_("⚠️ Pas assez de features valides"))
            
            st.subheader(_("Détails des tracks"))
            
            for idx, track in enumerate(valid_tracks[:20]):
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{track['name']}**")
                    st.caption(f"{track['artists']} - {track['genre']} ({track['confidence']:.1%})")
                
                with col3:
                    if track.get('preview_url'):
                        if st.button("🎵", key=f"preview_{idx}"):
                            st.audio(track['preview_url'], format="audio/mp3")
                
                with col4:
                    st.markdown(f"🎯 {track['confidence']:.1%}")
                
                with col5:
                    st.markdown(f"**{track['genre']}**")
        else:
            st.info(_("👆 Analysez des tracks pour voir les statistiques"))
    else:
        st.info(_("👆 Ajoutez des tracks pour voir l'analyse"))

with tab4:
    st.header(_("🎨 Générateur de Playlist"))
    
    # ========================================
    # VÉRIFICATION DES PRÉREQUIS
    # ========================================
    prerequisites_ok = True
    
    if len(st.session_state.playlist_tracks) < 2:
        st.warning(_("⚠️ **Prérequis**: Ajoutez au moins 2 morceaux"))
        prerequisites_ok = False
    
    if 'pca_df' not in st.session_state or st.session_state.pca_df is None:
        st.warning(_("⚠️ **Prérequis**: Effectuez d'abord l'analyse dans l'onglet '📊 Analyse'"))
        prerequisites_ok = False
    
    # Afficher l'état actuel si prérequis non remplis
    if not prerequisites_ok:
        st.info(_("""
        **Étapes pour créer une playlist:**
        1. 🎧 **Mes Musiques** ou 🔍 **Recherche**: Importez et analysez vos morceaux
        2. 📊 **Analyse**: Laissez le système analyser les caractéristiques audio  
        3. 🎨 **Playlist**: Créez votre playlist personnalisée ici
        """))
        
        st.subheader(_("État actuel"))
        col1, col2 = st.columns(2)
        with col1:
            st.metric(_("Morceaux analysés"), len(st.session_state.playlist_tracks))
        with col2:
            valid_tracks = len([t for t in st.session_state.playlist_tracks if t.get('features') is not None])
            st.metric(_("Avec features"), valid_tracks)
    
    # ========================================
    # GÉNÉRATION DE PLAYLIST
    # ========================================
    elif len(st.session_state.pca_df) >= 2:
        # Choix du mode de génération
        st.markdown("---")
        playlist_mode = st.radio(
            _("Mode de génération:"),
            options=['transition', 'genre'],
            format_func=lambda x: _("🎯 Transition progressive") if x == 'transition' else _("🎸 Par genre"),
            horizontal=True,
            key='playlist_mode'
        )
        
        # MODE TRANSITION
        if playlist_mode == 'transition':
            st.markdown(_("Sélectionnez deux tracks pour créer une playlist progressive entre elles"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                track1_idx = st.selectbox(
                    _("🚀 Track de départ:"),
                    range(len(st.session_state.pca_df)),
                    format_func=lambda x: f"{st.session_state.pca_df.iloc[x]['name'][:40]} ({st.session_state.pca_df.iloc[x]['genre']})",
                    key="track1_selector"
                )
            
            with col2:
                track2_idx = st.selectbox(
                    _("🎯 Track d'arrivée:"),
                    range(len(st.session_state.pca_df)),
                    index=min(1, len(st.session_state.pca_df)-1),
                    format_func=lambda x: f"{st.session_state.pca_df.iloc[x]['name'][:40]} ({st.session_state.pca_df.iloc[x]['genre']})",
                    key="track2_selector"
                )
            
            num_tracks = st.slider(_("Nombre de tracks dans la playlist:"), 2, 100, 10, key="num_tracks_slider")
            
            # BOUTON DE GÉNÉRATION
            if st.button(_("🎯 Générer la playlist"), type="primary", key="generate_playlist_btn"):
                if track1_idx != track2_idx:
                    with st.spinner(_("Génération de la playlist...")):
                        pathfinder = st.session_state.playlist_pathfinder
                        playlist, line_points, p1, p2 = generate_playlist_line(
                            st.session_state.pca_df, 
                            track1_idx, 
                            track2_idx, 
                            pathfinder,
                            num_tracks
                        )
                        
                        if playlist is not None:
                            st.session_state.generated_playlist = playlist
                            st.session_state.line_points = line_points
                            st.session_state.p1 = p1
                            st.session_state.p2 = p2
                            st.session_state.track1_idx = track1_idx
                            st.session_state.track2_idx = track2_idx
                            pathfinder = st.session_state.playlist_pathfinder
                            st.session_state.playlist_analysis = analyze_playlist_quality(playlist, pathfinder)
                            st.success(_(f"✅ Playlist de {len(playlist)} tracks générée!"))
                        else:
                            st.error(_("❌ Échec de la génération de la playlist"))
                        visualize_playlist_transition(st.session_state.pca_df, playlist, line_points, p1, p2, track1_idx, track2_idx, list(LABEL_MAPPING.keys()))

                else:
                    st.warning(_("⚠️ Sélectionnez deux tracks différentes"))
        
        # MODE GENRE
        else:
            st.markdown(_("Créez une playlist basée sur un ou plusieurs genres"))
            
            # Liste des genres disponibles
            available_genres = sorted(st.session_state.pca_df['genre'].unique())
            
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_genres = st.multiselect(
                    _("Sélectionnez les genres:"),
                    options=available_genres,
                    default=[available_genres[0]] if available_genres else [],
                    key='genre_selector'
                )
            
            with col2:
                sort_by = st.selectbox(
                    _("Trier par:"),
                    options=['confidence', 'name', 'random'],
                    format_func=lambda x: {
                        'confidence': _('confidence'),
                        'name': _('Nom'),
                        'random': _('Aléatoire')
                    }[x],
                    key='genre_sort'
                )
            
            max_tracks = st.slider(
                _("Nombre max de tracks par genre:"),
                2, 100, 20,
                key='genre_max_tracks'
            )
            
            if st.button(_("🎸 Générer playlist par genre"), type="primary", key="generate_genre_playlist_btn"):
                if selected_genres:
                    with st.spinner(_("Génération de la playlist...")):
                        # Filtrer par genres
                        genre_tracks = st.session_state.pca_df[
                            st.session_state.pca_df['genre'].isin(selected_genres)
                        ].copy()
                        
                        # Trier
                        if sort_by == 'confidence':
                            genre_tracks = genre_tracks.sort_values('confidence', ascending=False)
                        elif sort_by == 'name':
                            genre_tracks = genre_tracks.sort_values('name')
                        else:  # random
                            genre_tracks = genre_tracks.sample(frac=1)
                        
                        # Limiter le nombre
                        genre_tracks = genre_tracks.head(max_tracks * len(selected_genres))
                        
                        # Créer la playlist
                        playlist = []
                        for idx, row in genre_tracks.iterrows():
                            playlist.append({
                                'position': len(playlist) + 1,
                                'track_id': idx,
                                'name': row['name'],
                                'artists': row['artists'],
                                'genre': row['genre'],
                                'confidence': row['confidence'],
                                'uri': row['uri'],
                                'spotify_id': row['spotify_id'],
                                'deezer_id': row['deezer_id'],
                                'preview_url': row['preview_url'],
                                'distance_to_line': 0,
                                'PC1': row['PC1'],
                                'PC2': row['PC2']
                            })
                        
                        if playlist:
                            st.session_state.generated_playlist = playlist
                            st.session_state.line_points = None
                            st.session_state.p1 = None
                            st.session_state.p2 = None
                            pathfinder = st.session_state.playlist_pathfinder
                            st.session_state.playlist_analysis = analyze_playlist_quality(playlist, pathfinder)
                            st.success(_(f"✅ Playlist de {len(playlist)} tracks générée!"))
                        else:
                            st.error(_("❌ Aucune track trouvée pour ces genres"))
                else:
                    st.warning(_("⚠️ Sélectionnez au moins un genre"))
        
        # ========================================
        # AFFICHAGE DE LA PLAYLIST GÉNÉRÉE
        # ========================================
        if st.session_state.generated_playlist:
            st.markdown("---")
            st.subheader(_("📋 Playlist Générée"))
            
            # ========================================
            # MÉTRIQUES DE QUALITÉ
            # ========================================
            if st.session_state.playlist_analysis:
                analysis = st.session_state.playlist_analysis
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(_("Nombre de tracks"), analysis['num_tracks'])
                col2.metric(_("Genres uniques"), analysis['unique_genres'])
                col3.metric(_("Distance moy."), f"{analysis['avg_distance_to_line']:.3f}")
                col4.metric(_("Fluidité moy."), f"{analysis['avg_smoothness']:.3f}")
                
                with st.expander(_("📊 Analyse détaillée")):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(_("**Distribution des genres:**"))
                        genre_dist = pd.DataFrame(
                            list(analysis['genre_distribution'].items()),
                            columns=[_('Genre'), _('Count')]
                        )
                        st.dataframe(genre_dist, use_container_width=True)
                    
                    with col2:
                        st.write(_("**Statistiques de distance:**"))
                        st.write(f"• {_('Distance max')}: {analysis['max_distance_to_line']:.3f}")
                        st.write(f"• {_('Écart-type')}: {analysis['std_distance_to_line']:.3f}")
                        st.write(f"• {_('Ratio diversité')}: {analysis['genre_diversity_ratio']:.1%}")
            
            st.markdown("---")
            
            
            # ========================================
            # LISTE DES TRACKS
            # ========================================
            st.subheader(_("🎵 Ordre de lecture"))

            spotify_tracks_count = 0

            # Trier la playlist par position pour garantir l'ordre correct
            sorted_playlist = sorted(st.session_state.generated_playlist, key=lambda x: x['position'])

            for track_info in sorted_playlist:
                col1, col2, col3, col4, col5 = st.columns([0.5, 3, 1, 1, 1])

                with col1:
                    if track_info['position'] == 1:
                        st.markdown("🚀")
                    elif track_info['position'] == len(sorted_playlist):
                        st.markdown("🎯")
                    else:
                        st.markdown(f"{track_info['position']}.")

                with col2:
                    st.markdown(f"**{track_info['name']}**")
                    st.caption(f"{track_info.get('artists', 'Unknown')} • {track_info['genre']}")

                with col3:
                    if track_info.get('uri') or track_info.get('spotify_id'):
                        spotify_tracks_count += 1
                        st.markdown("🟢 Spotify")
                    elif track_info.get('deezer_id'):
                        st.markdown("🔵 Deezer")
                    else:
                        st.markdown("⚪ Local")

                with col4:
                    if playlist_mode == 'transition':
                        st.markdown(f"📏 {track_info['distance_to_line']:.3f}")
                    else:
                        st.markdown(f"🎯 {track_info['confidence']:.3f}")

                with col5:
                    if track_info.get('preview_url'):
                        if st.button("🎵", key=f"play_{track_info['position']}"):
                            st.audio(track_info['preview_url'], format="audio/mp3")

            
            # ========================================
            # EXPORT VERS SPOTIFY
            # ========================================
            st.markdown("---")
            st.subheader(_("📤 Exporter vers Spotify"))
            
            deezer_tracks = len(st.session_state.generated_playlist) - spotify_tracks_count
            st.info(_(f"ℹ️ {spotify_tracks_count} tracks Spotify exportables • {deezer_tracks} tracks Deezer non exportables"))
            
            if spotify_tracks_count > 0:
                col1, col2 = st.columns(2)
                with col1:
                    playlist_name = st.text_input(
                        _("Nom de la playlist:"), 
                        value="Ma Playlist IA",
                        key="playlist_name_input"
                    )
                with col2:
                    playlist_desc = st.text_input(
                        _("Description:"), 
                        value=_("Générée par IA - Transition musicale progressive"),
                        key="playlist_desc_input"
                    )
                
                if st.button(_("🎵 Créer la playlist sur Spotify"), type="primary", key="export_spotify_btn"):
                    spotify_service = st.session_state.spotify_service
                    
                    if spotify_service:
                        # Filtrer uniquement les tracks Spotify
                        spotify_only_tracks = [
                            track for track in st.session_state.generated_playlist 
                            if track.get('uri') or track.get('spotify_id')
                        ]
                        
                        if spotify_only_tracks:
                            result = export_playlist_to_spotify(
                                spotify_service, 
                                spotify_only_tracks, 
                                playlist_name, 
                                playlist_desc
                            )
                            
                            if result:
                                st.balloons()
                        else:
                            st.warning(_("Aucune track Spotify dans la playlist"))
                    else:
                        st.error(_("❌ Connectez-vous à Spotify pour exporter"))
            else:
                st.warning(_("⚠️ Aucune track Spotify dans cette playlist"))
            
            # Bouton pour réinitialiser
            st.markdown("---")
            if st.button(_("🔄 Générer une nouvelle playlist"), type="secondary", key="reset_playlist_btn"):
                st.session_state.generated_playlist = None
                st.session_state.line_points = None
                st.session_state.p1 = None
                st.session_state.p2 = None
                st.session_state.playlist_analysis = None
                st.rerun()

with tab5:
    st.header(_("🤔 Explications"))
    IMAGE_DIR = "templates/assets/images/"
    PCA_IMAGE = IMAGE_DIR + "pca.png"
    TSNE_IMAGE = IMAGE_DIR + "t-sne.png"
    PLAYLIST_IMAGE = IMAGE_DIR + "creation_playlist.png"
    SPECT_IMAGE = IMAGE_DIR +"ConvolutionSize.png"

    st.markdown(_("""
    **Auteurs :** Gabriel Marie–Brisson, Clément Delmas, Thibault Pottier, Aurélien Gauthier
                    """))
    st.markdown(_("""
    **Enseignant référent :** Charles Brazier
    """))

    st.header(_("1. Présentation"))

    st.subheader(_("Contexte et Objectif"))
    st.markdown(_("""
    Ce projet a pour objectif principal de développer une **Intelligence Artificielle capable de classifier des musiques par genre** afin de générer des **playlists cohérentes et ordonnées par similarité**. L'approche repose sur un algorithme de *Machine Learning* qui analyse les propriétés sonores des morceaux.

    Le processus de développement a été scindé en trois phases principales :
    1.  Le **pré-traitement des données**, notamment la transformation des fichiers audio en Spectrogrammes de Mel.
    2.  La **réalisation du classifieur** basé sur un réseau neuronal convolutionnel (CNN).
    3.  L'**implémentation de l'algorithme de suggestion musicale** (basé sur la projection des résultats via PCA ou t-SNE pour déterminer le chemin de lecture le plus cohérent).

    Pour ce faire nous nous sommes appuyés sur ce blog post de [**@Sander Dieleman**](https://sander.ai/2014/08/05/spotify-cnns.html) qui explique comment le modèle de suggestion musicale peut être implémenté avec des techniques de deep Learning.
    """))

    st.subheader(_("Données"))
    st.markdown(_("""
    Le modèle a été entraîné sur le **GTZAN Dataset** pour la classification des genres musicaux. Ce jeu de données est composé de 100 fichiers audio de 30 secondes chacun, répartis équitablement entre 10 genres musicaux distincts : Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, et Rock.
    Comme vous pouvez l'imaginer, le fait qu'il y est que 100 fichiers audio de 30 secondes ne suffisent pas à couvrir tous les genres musicaux est un point important à étudier. Ainsi dans le cas d'une musique qui ne correspondrait pas à un des 10 genres, le modèle pourrait avoir des difficultés à la classifier correctement.
    Pour avoir de meilleures performances, il serait pertinent d'augmenter la taille du dataset en ajoutant plus de musiques et plus de genres.
    Le pré-traitement essentiel consiste à convertir les segments audio de 30 secondes en **Spectrogrammes de Mel**, une représentation graphique du spectre de fréquences adaptée à la perception auditive humaine.
    """))

    # --- 2. Architecture du Modèle ---
    st.header(_("2. Architecture du Modèle"))

    st.markdown(_("""
    Le classifieur est basé sur une architecture de **Réseau Neuronal Convolutionnel (CNN)** nommée `CNN_music`. Cette architecture est conçue pour extraire des caractéristiques pertinentes directement à partir des Spectrogrammes de Mel.
    """))

    st.subheader(_("Structure du `CNN_music`"))
    st.markdown(_("""
    Le modèle utilise une succession de couches de convolution, de normalisation par lots (`BatchNorm2d`), de fonctions d'activation (`ReLU`) et de couches de regroupement (`MaxPool2d`), suivies de couches entièrement connectées (`Linear`) pour la classification finale.

    Une caractéristique notable de cette architecture est l'intégration de modules d'attention spécifiques : les **CBAM** (*Convolutional Block Attention Module*).
    """))
    architecture_data = {
        _("Couche"): [
            _("Couches Convolutionnelles (Conv)"),
            _("Modules CBAM"),
            _("Couches Entièrement Connectées (FC)")
        ],
        _("Type"): [
            _("`Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`"),
            _("`CBAM`"),
            _("`Linear`, `BatchNorm1d`, `Dropout2d`")
        ],
        _("Rôle Principal"): [
            _("Extraction hiérarchique des caractéristiques spectrales et temporelles du spectrogramme de Mel."),
            _("Renforcement des caractéristiques importantes via des mécanismes d'attention."),
            _("Transformation en prédictions de probabilités pour les 10 genres musicaux.")
        ]
    }
    st.table(pd.DataFrame(architecture_data))


    st.markdown(_("""
    L'intégration des modules **CBAM** vise à améliorer la performance du modèle en lui permettant de se concentrer dynamiquement sur les régions et les canaux (filtres) les plus informatifs du Spectrogramme de Mel pour chaque musique.
    """))

    # --- 3. Résultats ---
    st.header(_("3. Résultats"))

    st.markdown(_(f"Le modèle a été entraîné pendant **30 minutes** et ses performances ont été évaluées sur des ensembles de validation et de test."))

    st.subheader(_("Performances Globales (Validation)"))
    validation_data = {
        _("Métrique"): [_("Loss du Modèle"), _("Précision (Accuracy)")],
        _("Valeur"): ["0.782", "73.0 %"]
    }
    st.table(pd.DataFrame(validation_data))

    st.subheader(_("Performances Détaillées (Test)"))
    test_data = {
        _("Classe"): [_("Blues"), _("Classical"), _("Country"), _("Disco"), _("Hip-hop"), _("Jazz"), _("Metal"), _("Pop"), _("Reggae"), _("Rock")],
        _("Précision"): ["100.0 %", "80.0 %", "60.0 %", "40.0 %", "80.0 %", "70.0 %", "90.0 %", "80.0 %", "70.0 %", "60.0 %"]
    }
    st.table(pd.DataFrame(test_data))

    st.markdown(_("""
    **Analyse :** Les résultats montrent une excellente performance pour le genre **Blues** (100 %) et une très bonne performance pour le **Metal** (90 %). Cependant, le modèle rencontre des difficultés significatives avec le genre **Disco** (40 %), suggérant un chevauchement des caractéristiques sonores de ce genre avec d'autres, ou un besoin d'ajustement des hyperparamètres pour cette classe.
    """))

    st.subheader(_("Visualisation des Résultats"))

    st.markdown(_("#### Projection 2D (PCA)"))
    st.image(PCA_IMAGE, caption=_("Projection 2D des données via PCA"))
    st.markdown(_("""
    La figure montre la projection des données sur les deux premières composantes principales (PC1 et PC2). On observe une certaine agrégation des points par genre, mais aussi un chevauchement important, indiquant que la simple PCA ne suffit pas à isoler clairement tous les genres.
    """))

    st.markdown(_("#### Visualisation t-SNE"))
    st.image(TSNE_IMAGE, caption=_("Visualisation t-SNE des genres musicaux"))
    st.markdown(_("""
    La visualisation t-SNE, plus apte à révéler la structure locale des données, montre une **séparation beaucoup plus nette** des 10 clusters de genres musicaux, confirmant que le modèle a réussi à apprendre des représentations distinctes pour chaque catégorie.
    """))

    st.markdown(_("#### Suggestion de Playlist"))
    st.image(PLAYLIST_IMAGE, caption=_("Génération de playlist par chemin de similarité"))
    st.markdown(_("""
    La phase de suggestion musicale utilise cette projection pour créer un "chemin" cohérent entre deux morceaux (début et fin), représentant la playlist ordonnée par similarité. L'image illustre un exemple de ce chemin dans l'espace de projection.
    """))

    st.markdown(_("#### Spectrogramme"))
    st.image(SPECT_IMAGE, caption=_("Spectrogramme de Mel d'un extrait audio"))
    st.markdown(_("""
    Le spectrogramme de Mel est la représentation visuelle des caractéristiques fréquentielles d'un extrait audio, utilisée comme entrée pour le modèle CNN."""))

    # --- 4. Coût de Développement ---
    st.header(_("4. Coût de Développement"))

    st.markdown(_("Le projet a été mené sur la durée d'un **semestre universitaire**, représentant le temps de développement et de recherche principal."))

    cost_data = {
        _("Catégorie de Coût"): [_("Temps de Développement"), _("Coûts Matériels"), _("Coûts Logiciels"), _("Coûts d'Infrastructure")],
        _("Détail"): [
            _("Un semestre (recherche, codage, tests, documentation)."),
            _("**Nuls**. L'entraînement du modèle a été réalisé localement sur un **MacBook M1**."),
            _("**Nuls**. Utilisation exclusive de bibliothèques et d'outils *open source* (ex: PyTorch, librosa)."),
            _("**Nuls**. Aucune utilisation de serveurs *cloud* ou de GPU dédiés n'a été nécessaire.")
        ]
    }
    st.table(pd.DataFrame(cost_data))
# Footer
st.markdown(_(
    """
    ---
    Développé par [Gabriel Marie-Brisson](https://gabriel.mariebrisson.fr)
    """
))