import streamlit as st
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
load_dotenv()

# Configuration du logging structuré
from src.utils.logger import StructuredLogger
import logging

# Configurer le logging au démarrage de l'application
StructuredLogger.configure(
    level=logging.INFO,
    use_json=True  # Format JSON pour la production
)

# Imports des nouvelles classes modulaires
from src.services.spotify_service import SpotifyService
from src.services.deezer_service import DeezerService
from src.services.track_processor import TrackProcessor
from src.models.audio_classifier import AudioClassifier
from src.logic.playlist_generator import PlaylistPathfinder
from src.utils.visualization import visualize_playlist_transition
from src.config import GENRE_LABEL_MAPPING, GENRE_COLORS, LANGUAGES

from deep_translator import GoogleTranslator

# Logger pour le fichier principal
logger = logging.getLogger(__name__)

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
# Toutes les constantes sont maintenant dans src/config.py


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

@st.cache_resource
def init_deezer_service():
    """Initialise le service Deezer avec cache Streamlit."""
    return DeezerService()


# --- FONCTIONS D'ANALYSE ---
# Toutes les fonctions métier sont maintenant dans les services src/

# --- FONCTIONS SPOTIFY ---
# Toutes les fonctions Spotify sont maintenant dans SpotifyService


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
if 'deezer_service' not in st.session_state:
    st.session_state.deezer_service = None
if 'track_processor' not in st.session_state:
    st.session_state.track_processor = None
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
        st.session_state.deezer_service = init_deezer_service()
        st.session_state.track_processor = TrackProcessor()
        
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
                deezer_service = st.session_state.deezer_service
                deezer_results = deezer_service.search_tracks(deezer_query, deezer_limit)
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
                            track_processor = st.session_state.track_processor
                            if track_processor.process_track_analysis(
                                track,
                                {"type": "deezer", "index": i},
                                st.session_state.audio_classifier,
                                st.session_state.deezer_service,
                                st.session_state.spotify_service,
                                st.session_state.analyzed_tracks,
                                callback_info=lambda msg: st.info(_(msg)),
                                callback_warning=lambda msg: st.warning(_(msg)),
                                callback_success=lambda msg: st.success(_(msg)),
                                callback_error=lambda msg: st.error(_(msg)),
                                callback_spinner=lambda msg: st.spinner(_(msg))
                            ):
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
                                track_processor = st.session_state.track_processor
                                if track_processor.process_track_addition(
                                    track,
                                    'deezer',
                                    st.session_state.analyzed_tracks,
                                    st.session_state.playlist_tracks,
                                    callback_info=lambda msg: st.info(_(msg)),
                                    callback_success=lambda msg: st.success(_(msg)),
                                    callback_error=lambda msg: st.error(_(msg))
                                ):
                                    st.rerun()
                        else:
                            if st.button(_("➕ Ajouter"), key=f"add_playlist_{i}"):
                                track_processor = st.session_state.track_processor
                                if (track_processor.process_track_analysis(
                                    track,
                                    {"type": "top", "index": i},
                                    st.session_state.audio_classifier,
                                    st.session_state.deezer_service,
                                    st.session_state.spotify_service,
                                    st.session_state.analyzed_tracks,
                                    callback_info=lambda msg: st.info(_(msg)),
                                    callback_warning=lambda msg: st.warning(_(msg)),
                                    callback_success=lambda msg: st.success(_(msg)),
                                    callback_error=lambda msg: st.error(_(msg)),
                                    callback_spinner=lambda msg: st.spinner(_(msg))
                                ) and track_processor.process_track_addition(
                                    track,
                                    'deezer',
                                    st.session_state.analyzed_tracks,
                                    st.session_state.playlist_tracks,
                                    callback_info=lambda msg: st.info(_(msg)),
                                    callback_success=lambda msg: st.success(_(msg)),
                                    callback_error=lambda msg: st.error(_(msg))
                                )):
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
                        result, model, scaler = pathfinder.perform_dimensionality_reduction(
                            features_list, 
                            method='tsne',
                            perplexity=perplexity_val
                        )
                    else:
                        result, model, scaler = pathfinder.perform_dimensionality_reduction(
                            features_list, 
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
                        
                        # Extraire les features brutes depuis analyzed_tracks
                        try:
                            raw_features, missing_tracks = pathfinder.extract_raw_features_from_analyzed_tracks(
                                st.session_state.pca_df,
                                st.session_state.analyzed_tracks
                            )
                            
                            if missing_tracks:
                                st.warning(_(f"Features non trouvées pour {len(missing_tracks)} track(s): {missing_tracks[:3]}"))
                            
                            # S'assurer que le pathfinder a le modèle PCA chargé pour la visualisation
                            if pathfinder.pca_model is None:
                                features_list = [t['features'] for t in st.session_state.analyzed_tracks if t.get('features') is not None]
                                if len(features_list) >= 2:
                                    pathfinder.perform_pca(features_list)
                            
                            # Générer la playlist
                            playlist, line_points, p1, p2 = pathfinder.generate_playlist_line_from_pca_df(
                                pca_df=st.session_state.pca_df,
                                raw_features=raw_features,
                                track1_idx=track1_idx,
                                track2_idx=track2_idx,
                                num_tracks=num_tracks
                            )
                            
                            if playlist is not None:
                                st.session_state.generated_playlist = playlist
                                st.session_state.line_points = line_points
                                st.session_state.p1 = p1
                                st.session_state.p2 = p2
                                st.session_state.track1_idx = track1_idx
                                st.session_state.track2_idx = track2_idx
                                st.session_state.playlist_analysis = pathfinder.analyze_playlist_quality(playlist)
                                st.success(_(f"✅ Playlist de {len(playlist)} tracks générée!"))
                            else:
                                st.error(_("❌ Échec de la génération de la playlist"))
                            
                            # Visualiser la transition
                            visualize_playlist_transition(
                                st.session_state.pca_df,
                                playlist,
                                line_points,
                                p1,
                                p2,
                                track1_idx,
                                track2_idx,
                                list(GENRE_LABEL_MAPPING.keys()),
                                label_mapping=GENRE_LABEL_MAPPING
                            )
                        except Exception as e:
                            st.error(_(f"❌ Erreur génération playlist: {e}"))
                            import traceback
                            st.error(traceback.format_exc())

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
                            st.session_state.playlist_analysis = pathfinder.analyze_playlist_quality(playlist)
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
                        st.dataframe(genre_dist, width='stretch')
                    
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
                            def callback_info(msg):
                                st.info(_(msg))
                            
                            def callback_warning(msg):
                                st.warning(_(msg))
                            
                            def callback_success(msg):
                                st.success(_(msg))
                            
                            def callback_error(msg):
                                st.error(_(msg))
                            
                            result = spotify_service.export_playlist(
                                spotify_only_tracks,
                                playlist_name,
                                playlist_desc,
                                callback_info=callback_info,
                                callback_warning=callback_warning,
                                callback_success=callback_success,
                                callback_error=callback_error
                            )
                            
                            if result:
                                st.markdown(f"[🎵 Ouvrir dans Spotify]({result['external_urls']['spotify']})")
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