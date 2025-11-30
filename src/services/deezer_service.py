"""Service pour gérer les interactions avec l'API Deezer."""

from typing import List, Dict, Any, Optional, Tuple
import requests

from src.config import DEEZER_BASE_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeezerService:
    """Service pour interagir avec l'API Deezer."""
    
    def __init__(self) -> None:
        """Initialise le service Deezer."""
        self.base_url = DEEZER_BASE_URL
    
    def search_tracks(
        self, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recherche des tracks sur Deezer.
        
        Args:
            query: Terme de recherche (nom de la chanson ou artiste).
            limit: Nombre maximum de résultats à retourner. Par défaut: 10.
        
        Returns:
            Liste de dictionnaires contenant les informations des tracks trouvées.
            Chaque dictionnaire contient: id, name, artists, preview_url, album, duration, deezer_id.
        
        Raises:
            requests.RequestException: Si la requête HTTP échoue.
        """
        try:
            url = f"{self.base_url}/search"
            params = {
                'q': query,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
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
            
            logger.info(
                f"Recherche Deezer réussie: {len(tracks)} tracks trouvées",
                extra={"extra": {"query": query, "limit": limit, "results_count": len(tracks)}}
            )
            
            return tracks
            
        except requests.RequestException as e:
            logger.exception(
                f"Erreur lors de la recherche Deezer: {e}",
                extra={"extra": {"query": query, "limit": limit, "error_type": type(e).__name__}}
            )
            return []
        except (KeyError, ValueError) as e:
            logger.exception(
                f"Erreur lors du parsing de la réponse Deezer: {e}",
                extra={"extra": {"query": query, "error_type": type(e).__name__}}
            )
            return []
    
    def download_preview(
        self, 
        preview_url: str, 
        output_path: str
    ) -> bool:
        """
        Télécharge l'extrait audio de 30s depuis Deezer.
        
        Args:
            preview_url: URL de l'extrait audio Deezer.
            output_path: Chemin où sauvegarder le fichier audio.
        
        Returns:
            True si le téléchargement a réussi, False sinon.
        
        Raises:
            requests.RequestException: Si la requête HTTP échoue.
            OSError: Si l'écriture du fichier échoue.
        """
        if not preview_url:
            logger.warning(
                "URL de preview vide",
                extra={"extra": {"output_path": output_path}}
            )
            return False
        
        try:
            response = requests.get(preview_url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(
                f"Téléchargement Deezer réussi: {output_path}",
                extra={"extra": {"preview_url": preview_url, "output_path": output_path}}
            )
            
            return True
            
        except requests.RequestException as e:
            logger.exception(
                f"Erreur lors du téléchargement Deezer: {e}",
                extra={"extra": {"preview_url": preview_url, "output_path": output_path, "error_type": type(e).__name__}}
            )
            return False
        except OSError as e:
            logger.exception(
                f"Erreur lors de l'écriture du fichier: {e}",
                extra={"extra": {"output_path": output_path, "error_type": type(e).__name__}}
            )
            return False
    
    def find_track_from_spotify(
        self, 
        track_name: str, 
        artist_name: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Recherche un morceau sur Deezer à partir des infos Spotify.
        
        Args:
            track_name: Nom de la track Spotify.
            artist_name: Nom de l'artiste Spotify.
        
        Returns:
            Tuple (preview_url, deezer_id) si trouvé, (None, None) sinon.
        
        Raises:
            requests.RequestException: Si la requête HTTP échoue.
        """
        try:
            # Construire la requête de recherche
            query = f"{artist_name} {track_name}"
            
            url = f"{self.base_url}/search"
            params = {
                'q': query,
                'limit': 5  # Prendre les 5 premiers résultats
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
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
                        deezer_id = item.get('id')
                        logger.info(
                            f"Track Deezer trouvée pour {track_name} - {artist_name}",
                            extra={"extra": {"track_name": track_name, "artist_name": artist_name, "deezer_id": deezer_id}}
                        )
                        return preview_url, deezer_id
            
            # Si aucune correspondance exacte, prendre le premier résultat avec preview
            for item in data.get('data', []):
                preview_url = item.get('preview')
                if preview_url:
                    deezer_id = item.get('id')
                    logger.info(
                        f"Track Deezer trouvée (premier résultat) pour {track_name} - {artist_name}",
                        extra={"extra": {"track_name": track_name, "artist_name": artist_name, "deezer_id": deezer_id}}
                    )
                    return preview_url, deezer_id
            
            logger.warning(
                f"Aucun track Deezer trouvé pour {track_name} - {artist_name}",
                extra={"extra": {"track_name": track_name, "artist_name": artist_name}}
            )
            return None, None
            
        except requests.RequestException as e:
            logger.exception(
                f"Erreur lors de la recherche Deezer depuis Spotify: {e}",
                extra={"extra": {"track_name": track_name, "artist_name": artist_name, "error_type": type(e).__name__}}
            )
            return None, None
        except (KeyError, ValueError) as e:
            logger.exception(
                f"Erreur lors du parsing de la réponse Deezer: {e}",
                extra={"extra": {"track_name": track_name, "artist_name": artist_name, "error_type": type(e).__name__}}
            )
            return None, None

