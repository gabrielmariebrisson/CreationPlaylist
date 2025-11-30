"""Service pour traiter l'analyse et l'ajout de tracks."""

from typing import Dict, Any, Optional, Callable, List
import os
import tempfile
import numpy as np

from src.models.audio_classifier import AudioClassifier
from src.services.deezer_service import DeezerService
from src.services.spotify_service import SpotifyService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrackProcessor:
    """Service pour orchestrer l'analyse et l'ajout de tracks."""

    def __init__(self) -> None:
        """Initialise le processeur de tracks."""
        pass

    def process_track_analysis(
        self,
        track: Dict[str, Any],
        track_data: Dict[str, Any],
        classifier: Optional[AudioClassifier],
        deezer_service: DeezerService,
        spotify_service: Optional[SpotifyService],
        analyzed_tracks: List[Dict[str, Any]],
        callback_info: Optional[Callable[[str], None]] = None,
        callback_warning: Optional[Callable[[str], None]] = None,
        callback_success: Optional[Callable[[str], None]] = None,
        callback_error: Optional[Callable[[str], None]] = None,
        callback_spinner: Optional[Callable[[str], Any]] = None,
    ) -> bool:
        """
        Traite l'analyse d'une track (recherche automatique sur Deezer si Spotify).

        Args:
            track: Dictionnaire contenant les informations de la track.
            track_data: Dictionnaire contenant le type et l'index de la track.
            classifier: Instance d'AudioClassifier pour la prédiction de genre.
            deezer_service: Instance de DeezerService pour les opérations Deezer.
            spotify_service: Instance optionnelle de SpotifyService pour le matching.
            analyzed_tracks: Liste des tracks déjà analysées (sera modifiée en place).
            callback_info: Fonction optionnelle pour afficher des messages info.
            callback_warning: Fonction optionnelle pour afficher des warnings.
            callback_success: Fonction optionnelle pour afficher des messages de succès.
            callback_error: Fonction optionnelle pour afficher des erreurs.
            callback_spinner: Fonction optionnelle pour afficher un spinner
                (doit retourner un context manager).

        Returns:
            True si l'analyse a réussi, False sinon.
        """
        if classifier is None:
            if callback_warning:
                callback_warning("⚠️ Modèle non chargé")
            logger.warning("Tentative d'analyse sans classifier")
            return False

        try:
            # Utiliser le spinner si disponible
            spinner_context = (
                callback_spinner("Analyse en cours...") if callback_spinner else None
            )
            if spinner_context:
                spinner = spinner_context.__enter__()
            else:
                spinner = None

            try:
                temp_dir = tempfile.mkdtemp()
                audio_path = os.path.join(
                    temp_dir, f"{track_data['type']}_{track_data['index']}.mp3"
                )

                preview_url = None
                deezer_id = None

                # Si c'est une track Deezer, utiliser directement son preview
                if track_data["type"] in ["deezer", "search_deezer"]:
                    preview_url = track.get("preview_url")
                    deezer_id = track.get("deezer_id")

                    if not preview_url:
                        if callback_warning:
                            callback_warning(
                                "⚠️ Aucun aperçu audio disponible sur Deezer"
                            )
                        logger.warning(
                            "Aucun preview_url pour track Deezer",
                            extra={"extra": {"track_name": track.get("name")}},
                        )
                        return False

                # Si c'est une track Spotify, rechercher sur Deezer
                else:
                    if callback_info:
                        callback_info("🔍 Recherche de l'extrait audio sur Deezer...")

                    # Extraire le nom de l'artiste (peut être une string ou une liste)
                    artists = track.get("artists", "")
                    if isinstance(artists, list):
                        artist_name = ", ".join([artist["name"] for artist in artists])
                    else:
                        artist_name = artists

                    # Rechercher sur Deezer
                    preview_url, deezer_id = deezer_service.find_track_from_spotify(
                        track["name"], artist_name
                    )

                    if not preview_url:
                        if callback_error:
                            callback_error("❌ Impossible de trouver cet extrait")
                            callback_info(
                                "💡 Astuce : Essayez de rechercher directement "
                                "dans l'onglet dédié"
                            )
                        logger.warning(
                            "Track Deezer non trouvée pour track Spotify",
                            extra={
                                "extra": {
                                    "track_name": track.get("name"),
                                    "artist_name": artist_name,
                                }
                            },
                        )
                        return False

                    if callback_success:
                        callback_success("✅ Extrait trouvé !")

                # Télécharger l'extrait Deezer
                download_success = deezer_service.download_preview(
                    preview_url, audio_path
                )

                if download_success:
                    # Analyser le fichier audio
                    result = classifier.predict(
                        audio_path, return_features=True, return_probabilities=True
                    )

                    if result and result.get("genre"):
                        genre = result["genre"]
                        confidence = result["confidence"]
                        features = result.get("features")
                        probs = result.get("probabilities")

                        artists = track.get("artists", "")
                        if isinstance(artists, list):
                            artists = ", ".join([artist["name"] for artist in artists])

                        # Matcher avec Spotify si c'est une track Deezer
                        spotify_match = None
                        if (
                            track_data["type"] in ["deezer", "search_deezer"]
                            and spotify_service
                        ):
                            spotify_match = spotify_service.match_deezer_to_spotify(
                                track["name"], artists
                            )

                        # Créer l'objet track analysé
                        analyzed_track = {
                            "name": track["name"],
                            "artists": artists,
                            "spotify_id": (
                                spotify_match["spotify_id"]
                                if spotify_match
                                else track.get("id")
                            ),
                            "deezer_id": deezer_id,
                            "uri": (
                                (
                                    spotify_match["uri"]
                                    if spotify_match
                                    else track.get("uri")
                                )
                            ),
                            "preview_url": preview_url,
                            "source": (
                                "deezer"
                                if track_data["type"] in ["deezer", "search_deezer"]
                                else "spotify"
                            ),
                            "genre": genre,
                            "confidence": confidence,
                            "features": features,
                            "probabilities": probs,
                        }

                        if spotify_match and callback_info:
                            callback_info(
                                f"🎵 Trouvé sur Spotify: {spotify_match['name']}"
                            )

                        analyzed_tracks.append(analyzed_track)

                        if callback_success:
                            callback_success(
                                f"✅ {track['name']} - {genre} ({confidence:.1%})"
                            )

                        logger.info(
                            f"Track analysée avec succès: {track['name']}",
                            extra={
                                "extra": {
                                    "track_name": track["name"],
                                    "genre": genre,
                                    "confidence": confidence,
                                }
                            },
                        )
                        return True
                    else:
                        if callback_error:
                            callback_error("❌ Erreur lors de la prédiction du genre")
                        logger.error(
                            "Erreur lors de la prédiction du genre",
                            extra={"extra": {"track_name": track.get("name")}},
                        )
                        return False
                else:
                    if callback_error:
                        callback_error("❌ Impossible de télécharger l'aperçu audio")
                    logger.error(
                        "Échec du téléchargement Deezer",
                        extra={"extra": {"preview_url": preview_url}},
                    )
                    return False
            finally:
                if spinner_context:
                    spinner_context.__exit__(None, None, None)

        except Exception as e:
            if callback_error:
                callback_error(f"❌ Erreur lors de l'analyse: {e}")
            logger.exception(
                f"Erreur lors de l'analyse de track: {e}",
                extra={
                    "extra": {
                        "track_name": track.get("name"),
                        "error_type": type(e).__name__,
                    }
                },
            )
            return False

    def process_track_addition(
        self,
        track: Dict[str, Any],
        source: str,
        analyzed_tracks: List[Dict[str, Any]],
        playlist_tracks: List[Dict[str, Any]],
        callback_info: Optional[Callable[[str], None]] = None,
        callback_success: Optional[Callable[[str], None]] = None,
        callback_error: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Ajoute une track à la playlist, en réutilisant l'analyse existante
        si disponible.

        Args:
            track: Dictionnaire contenant les informations de la track.
            source: Source de la track ('spotify', 'deezer', etc.).
            analyzed_tracks: Liste des tracks déjà analysées.
            playlist_tracks: Liste des tracks de la playlist (sera modifiée en place).
            callback_info: Fonction optionnelle pour afficher des messages info.
            callback_success: Fonction optionnelle pour afficher des messages de succès.
            callback_error: Fonction optionnelle pour afficher des erreurs.

        Returns:
            True si l'ajout a réussi, False sinon.
        """
        try:
            # Vérifier si le morceau est déjà dans la playlist
            already_in_playlist = any(
                t.get("deezer_id") == track.get("id")
                or t["name"].lower() == track["name"].lower()
                for t in playlist_tracks
            )
            if already_in_playlist:
                if callback_info:
                    callback_info(f"✅ '{track['name']}' est déjà dans la playlist.")
                logger.debug(f"Track déjà dans la playlist: {track['name']}")
                return False

            # Vérifier si le morceau a déjà été analysé
            existing_analysis = next(
                (
                    t
                    for t in analyzed_tracks
                    if t.get("deezer_id") == track.get("id")
                    or t.get("spotify_id") == track.get("id")
                    or t["name"].lower() == track["name"].lower()
                ),
                None,
            )

            artists = track.get("artists", "")
            if isinstance(artists, list):
                artists = ", ".join([artist["name"] for artist in artists])

            is_deezer = track.get("deezer_id") is not None or source == "deezer"

            # Si on a déjà une analyse : réutiliser les infos
            if existing_analysis:
                playlist_entry = {
                    "name": existing_analysis["name"],
                    "artists": existing_analysis["artists"],
                    "spotify_id": existing_analysis.get("spotify_id"),
                    "deezer_id": existing_analysis.get("deezer_id"),
                    "uri": existing_analysis.get("uri"),
                    "preview_url": existing_analysis.get("preview_url"),
                    "source": existing_analysis.get("source"),
                    "genre": existing_analysis.get("genre"),
                    "confidence": existing_analysis.get("confidence"),
                    "features": existing_analysis.get("features", {}),
                    "probabilities": existing_analysis.get("probabilities", {}),
                }

                playlist_tracks.append(playlist_entry)

                if callback_success:
                    callback_success(
                        f"✅ '{track['name']}' ajouté à la playlist depuis "
                        "les analyses existantes."
                    )

                logger.info(f"Track ajoutée depuis analyse existante: {track['name']}")
                return True

            # Sinon, ajout brut sans analyse
            track_data = {
                "name": track["name"],
                "artists": artists,
                "spotify_id": track.get("id") if not is_deezer else None,
                "deezer_id": track.get("deezer_id"),
                "uri": track.get("uri"),
                "preview_url": track.get("preview_url"),
                "source": "deezer" if is_deezer else "spotify",
                "genre": "Non analysé",
                "confidence": 0.0,
                "features": {},
                "probabilities": {},
            }

            playlist_tracks.append(track_data)

            if callback_success:
                callback_success(
                    f"✅ '{track['name']}' ajouté à la playlist (non analysé)."
                )

            logger.info(f"Track ajoutée sans analyse: {track['name']}")
            return True

        except Exception as e:
            if callback_error:
                callback_error(f"❌ Erreur lors de l'ajout du morceau: {e}")
            logger.exception(
                f"Erreur lors de l'ajout de track: {e}",
                extra={
                    "extra": {
                        "track_name": track.get("name"),
                        "error_type": type(e).__name__,
                    }
                },
            )
            return False
