"""Module de visualisation pour les playlists."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.config import GENRE_LABEL_MAPPING
from src.utils.logger import get_logger

logger = get_logger(__name__)


def visualize_playlist_transition(
    pca_df: pd.DataFrame,
    playlist: List[Dict[str, Any]],
    line_points: Optional[np.ndarray],
    p1: Optional[np.ndarray],
    p2: Optional[np.ndarray],
    track1_id: Any,
    track2_id: Any,
    genre_names: List[str],
    label_mapping: Optional[Dict[int, str]] = None
) -> None:
    """
    Visualise la transition de playlist entre deux tracks.
    
    Args:
        pca_df: DataFrame avec les coordonnées PCA de tous les tracks.
        playlist: Liste des tracks de la playlist générée.
        line_points: Points interpolés en 2D (pour la ligne de transition).
        p1: Point de départ en 2D.
        p2: Point d'arrivée en 2D.
        track1_id: ID ou index de la track de départ.
        track2_id: ID ou index de la track d'arrivée.
        genre_names: Liste des noms de genres.
        label_mapping: Mapping des genres (utilise GENRE_LABEL_MAPPING par défaut).
    
    Raises:
        ValueError: Si la playlist est vide ou None.
        KeyError: Si les colonnes nécessaires sont manquantes dans pca_df.
    """
    if playlist is None or len(playlist) == 0:
        st.warning("Aucune playlist à visualiser")
        logger.warning("Tentative de visualisation d'une playlist vide")
        return
    
    if line_points is None or p1 is None or p2 is None:
        st.warning("⚠️ Impossible de visualiser une playlist vide ou incomplète.")
        logger.warning("Données de visualisation incomplètes", extra={"extra": {
            "line_points": line_points is not None,
            "p1": p1 is not None,
            "p2": p2 is not None
        }})
        return
    
    # Utiliser le label_mapping fourni ou celui par défaut
    if label_mapping is None:
        label_mapping = GENRE_LABEL_MAPPING
    
    try:
        plt.figure(figsize=(14, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(genre_names)))
        for i, genre in enumerate(genre_names):
            mask = pca_df["genre"] == genre
            genre_label = label_mapping.get(genre, genre) if isinstance(genre, int) else genre
            plt.scatter(
                pca_df[mask]["PC1"],
                pca_df[mask]["PC2"],
                c=[colors[i]],
                label=genre_label,
                alpha=0.3,
                s=30
            )
        
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k--", linewidth=2, alpha=0.7, label="Playlist Line")
        
        plt.scatter(
            line_points[:, 0], line_points[:, 1],
            c="red", s=50, alpha=0.5, marker="x", label="Target Points"
        )
        
        # Points de départ et d'arrivée
        plt.scatter([p1[0]], [p1[1]], c="blue", s=200, marker="*",
                    edgecolors="black", linewidth=2, label=f"Start: {track1_id}")
        plt.scatter([p2[0]], [p2[1]], c="green", s=200, marker="*",
                    edgecolors="black", linewidth=2, label=f"End: {track2_id}")
        
        # Points de la playlist
        playlist_points = np.array([
            [track.get("PC1", 0), track.get("PC2", 0)]
            for track in playlist
            if "PC1" in track and "PC2" in track
        ])
        if len(playlist_points) > 0:
            plt.scatter(
                playlist_points[:, 0], playlist_points[:, 1],
                c="red", s=100, alpha=0.8, edgecolors="black", linewidth=1, label="Playlist Tracks"
            )
        
        # Annotations des genres
        for track in playlist:
            if "PC1" in track and "PC2" in track:
                genre = track.get("genre")
                genre_label = label_mapping.get(genre, genre) if isinstance(genre, int) else genre
                if genre_label is None:
                    genre_label = track.get("genre", "Unknown")
                plt.annotate(
                    genre_label,
                    (track["PC1"], track["PC2"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold"
                )
        
        # Dessiner les lignes entre les target points et les actual points
        if line_points is not None and len(line_points) > 0:
            for i, track in enumerate(playlist):
                # Vérifier que le track a PC1 et PC2
                if "PC1" in track and "PC2" in track and i < len(line_points):
                    # Utiliser le point interpolé correspondant comme target
                    target = line_points[i]
                    # Utiliser PC1/PC2 comme actual point
                    actual = np.array([track["PC1"], track["PC2"]])
                    plt.plot([target[0], actual[0]], [target[1], actual[1]], "r-", alpha=0.3, linewidth=1)
        
        plt.xlabel("PC1 (variance)")
        plt.ylabel("PC2 (variance)")
        plt.title(f"Playlist Generation: {track1_id} -> {track2_id}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
        
        logger.debug(
            "Visualisation de playlist générée avec succès",
            extra={"extra": {
                "playlist_length": len(playlist),
                "track1_id": str(track1_id),
                "track2_id": str(track2_id)
            }}
        )
        
    except KeyError as e:
        logger.exception(
            f"Colonne manquante dans pca_df: {e}",
            extra={"extra": {"missing_column": str(e)}}
        )
        st.error(f"Erreur de visualisation: colonne manquante {e}")
        plt.close()
        raise
    except Exception as e:
        logger.exception(
            f"Erreur lors de la visualisation: {e}",
            extra={"extra": {"error_type": type(e).__name__}}
        )
        st.error(f"Erreur lors de la visualisation: {e}")
        plt.close()
        raise

