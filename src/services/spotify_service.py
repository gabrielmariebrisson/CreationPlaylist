"""Service pour gérer les interactions avec l'API Spotify."""

import os
from typing import Optional, Dict, List, Any, Callable

import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

from src.config import (
    SPOTIFY_SCOPE,
    SPOTIFY_DEFAULT_REDIRECT_URI,
    SPOTIFY_TRACK_ID_LENGTH,
    SPOTIFY_SEARCH_LIMIT_DEFAULT,
    SPOTIFY_SEARCH_LIMIT_MATCH,
    SPOTIFY_PLAYLIST_BATCH_SIZE,
)

load_dotenv()


class SpotifyService:
    """Service pour gérer l'authentification et les opérations Spotify."""
    
    def __init__(
        self, 
        auth_manager: Optional[SpotifyOAuth] = None, 
        session_state: Optional[Any] = None
    ) -> None:
        """
        Initialise le service Spotify.
        
        Args:
            auth_manager: Gestionnaire d'authentification Spotify (injection de dépendance).
            session_state: État de session Streamlit (pour le cache du token).
        """
        self.auth_manager = auth_manager or self._create_auth_manager()
        self.session_state = session_state
        self._client: Optional[spotipy.Spotify] = None
    
    def _create_auth_manager(self) -> Optional[SpotifyOAuth]:
        """
        Crée un gestionnaire d'authentification Spotify depuis les variables d'environnement.
        
        Returns:
            Instance de SpotifyOAuth ou None si les credentials sont manquants.
        """
        client_id = os.getenv('CLIENT_ID_SPOTIFY')
        client_secret = os.getenv('CLIENT_SECRET_SPOTIFY')
        redirect_uri = os.getenv('REDIRECT_URI_SPOTIFY', SPOTIFY_DEFAULT_REDIRECT_URI)
        
        if not client_id or not client_secret:
            return None
        
        return SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SPOTIFY_SCOPE,
            cache_path=None,
            show_dialog=False,
            open_browser=False
        )
    
    def get_client(self) -> Optional[spotipy.Spotify]:
        """
        Obtient un client Spotify authentifié avec refresh token permanent.
        
        Returns:
            Client Spotify authentifié ou None en cas d'erreur.
        
        Raises:
            SpotifyException: Si l'authentification échoue.
        """
        if not self.auth_manager:
            return None
        
        refresh_token = os.getenv('REFRESH_TOKEN_SPOTIFY')
        
        if not refresh_token:
            return None
        
        # Utiliser le cache de session si disponible
        if self.session_state is not None:
            if 'spotify_service_token' not in self.session_state:
                self.session_state.spotify_service_token = {
                    'refresh_token': refresh_token,
                    'expires_at': 0
                }
            
            token_info = self.session_state.spotify_service_token
            
            # Vérifier expiration
            if self.auth_manager.is_token_expired(token_info):
                try:
                    new_token_info = self.auth_manager.refresh_access_token(refresh_token)
                    
                    if 'refresh_token' not in new_token_info:
                        new_token_info['refresh_token'] = refresh_token
                    
                    self.session_state.spotify_service_token = new_token_info
                    token_info = new_token_info
                except SpotifyException as e:
                    # Éviter les dépendances circulaires avec streamlit
                    return None
                except Exception as e:
                    return None
            
            return spotipy.Spotify(auth=token_info['access_token'])
        
        # Fallback sans session_state
        try:
            token_info = self.auth_manager.refresh_access_token(refresh_token)
            return spotipy.Spotify(auth=token_info['access_token'])
        except (SpotifyException, KeyError) as e:
            return None
    
    def search_track(
        self, 
        query: str, 
        limit: int = SPOTIFY_SEARCH_LIMIT_DEFAULT
    ) -> Optional[Dict[str, Any]]:
        """
        Recherche un morceau sur Spotify.
        
        Args:
            query: Requête de recherche (nom du morceau + artiste).
            limit: Nombre maximum de résultats. Par défaut: SPOTIFY_SEARCH_LIMIT_DEFAULT.
        
        Returns:
            Résultat de recherche ou None en cas d'erreur.
        
        Raises:
            SpotifyException: Si la recherche échoue.
        """
        client = self.get_client()
        if not client:
            return None
        
        try:
            results = client.search(q=query, type='track', limit=limit)
            return results
        except SpotifyException:
            return None
    
    def match_deezer_to_spotify(
        self, 
        track_name: str, 
        artist_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Trouve le track Spotify correspondant à un track Deezer.
        
        Args:
            track_name: Nom du morceau.
            artist_name: Nom de l'artiste.
        
        Returns:
            Dictionnaire avec les infos Spotify ou None si non trouvé.
        
        Raises:
            SpotifyException: Si la recherche échoue.
        """
        client = self.get_client()
        if not client:
            return None
        
        try:
            query = f"{track_name} {artist_name}"
            results = client.search(q=query, type='track', limit=SPOTIFY_SEARCH_LIMIT_MATCH)
            
            if results['tracks']['items']:
                best_match = results['tracks']['items'][0]
                return {
                    'spotify_id': best_match['id'],
                    'uri': best_match['uri'],
                    'name': best_match['name'],
                    'artists': ', '.join([a['name'] for a in best_match['artists']])
                }
            return None
        except (SpotifyException, KeyError, IndexError):
            return None
    
    def export_playlist(
        self, 
        playlist_tracks: List[Dict[str, Any]], 
        playlist_name: str, 
        playlist_description: str = "",
        callback_info: Optional[Callable[[str], None]] = None,
        callback_warning: Optional[Callable[[str], None]] = None,
        callback_success: Optional[Callable[[str], None]] = None,
        callback_error: Optional[Callable[[str], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Exporte une playlist vers Spotify avec recherche automatique des URIs manquants.
        
        Args:
            playlist_tracks: Liste des tracks à exporter.
            playlist_name: Nom de la playlist.
            playlist_description: Description de la playlist.
            callback_info: Fonction callback pour les messages info.
            callback_warning: Fonction callback pour les warnings.
            callback_success: Fonction callback pour les succès.
            callback_error: Fonction callback pour les erreurs.
        
        Returns:
            Playlist créée ou None en cas d'erreur.
        
        Raises:
            SpotifyException: Si l'export échoue.
        """
        client = self.get_client()
        if not client:
            if callback_error:
                callback_error("❌ Client Spotify non disponible")
            return None
        
        try:
            user_info = client.current_user()
            user_id = user_info['id']
            
            playlist = client.user_playlist_create(
                user=user_id,
                name=playlist_name,
                public=False,
                description=playlist_description
            )
            
            track_uris: List[str] = []
            skipped = 0
            found_on_search = 0
            
            for track in playlist_tracks:
                uri: Optional[str] = None
                
                # Essayer d'obtenir l'URI existant
                if track.get('uri') and track['uri'].startswith('spotify:track:'):
                    uri = track['uri']
                elif track.get('spotify_id'):
                    spotify_id = track['spotify_id']
                    if (
                        isinstance(spotify_id, str) 
                        and len(spotify_id) == SPOTIFY_TRACK_ID_LENGTH 
                        and not spotify_id.isdigit()
                    ):
                        uri = f"spotify:track:{spotify_id}"
                
                # Si pas d'URI valide, chercher sur Spotify
                if not uri:
                    track_name = track.get('name', '')
                    artists = track.get('artists', '')
                    
                    if track_name and artists:
                        try:
                            query = f"{track_name} {artists}"
                            results = client.search(q=query, type='track', limit=1)
                            
                            if results['tracks']['items']:
                                found_track = results['tracks']['items'][0]
                                uri = found_track['uri']
                                found_on_search += 1
                                if callback_info:
                                    callback_info(f"🔍 Trouvé sur Spotify: {track_name}")
                        except SpotifyException:
                            if callback_warning:
                                callback_warning(f"⚠️ Recherche échouée pour: {track_name}")
                
                if uri:
                    track_uris.append(uri)
                else:
                    skipped += 1
                    if callback_warning:
                        callback_warning(f"⚠️ Ignoré: {track.get('name', 'Unknown')}")
            
            if track_uris:
                # Ajouter par batch de 100
                for i in range(0, len(track_uris), SPOTIFY_PLAYLIST_BATCH_SIZE):
                    batch = track_uris[i:i+SPOTIFY_PLAYLIST_BATCH_SIZE]
                    client.playlist_add_items(playlist['id'], batch)
                
                msg = (
                    f"✅ Playlist '{playlist_name}' créée avec {len(track_uris)} titres!"
                )
                if found_on_search > 0:
                    msg += f" ({found_on_search} trouvés par recherche)"
                if skipped > 0:
                    msg += f" ({skipped} ignorés)"
                
                if callback_success:
                    callback_success(msg)
                
                return playlist
            else:
                if callback_warning:
                    callback_warning("⚠️ Aucun URI Spotify valide trouvé")
                return None
        except (SpotifyException, KeyError) as e:
            if callback_error:
                callback_error(f"❌ Erreur lors de la création de la playlist: {e}")
            return None
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations de l'utilisateur actuel.
        
        Returns:
            Informations utilisateur ou None en cas d'erreur.
        
        Raises:
            SpotifyException: Si la récupération échoue.
        """
        client = self.get_client()
        if not client:
            return None
        
        try:
            return client.current_user()
        except SpotifyException:
            return None
