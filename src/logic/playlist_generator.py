"""Générateur de playlist basé sur la réduction de dimensionnalité."""

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.config import (
    PCA_N_COMPONENTS,
    TSNE_N_COMPONENTS,
    TSNE_DEFAULT_PERPLEXITY,
    TSNE_DEFAULT_RANDOM_STATE,
    TSNE_MAX_ITER,
    MIN_FEATURES_FOR_REDUCTION,
    DEFAULT_PLAYLIST_SIZE,
)


class PlaylistPathfinder:
    """Classe pour générer des playlists basées sur la similarité musicale."""
    
    def __init__(self) -> None:
        """Initialise le générateur de playlist."""
        self.pca_model: Optional[PCA] = None
        self.tsne_model: Optional[TSNE] = None
        self.scaler: Optional[StandardScaler] = None
    
    def perform_pca(
        self, 
        features_list: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[PCA], Optional[StandardScaler]]:
        """
        Effectue une PCA sur les features extraites.
        
        Args:
            features_list: Liste des features extraites (arrays numpy).
        
        Returns:
            Tuple (features_pca, pca_model, scaler) ou (None, None, None) si erreur.
        
        Raises:
            ValueError: Si le nombre de features est insuffisant.
            RuntimeError: Si la PCA échoue.
        """
        if len(features_list) < MIN_FEATURES_FOR_REDUCTION:
            return None, None, None
        
        try:
            features_array = np.array(features_list)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            pca = PCA(n_components=PCA_N_COMPONENTS)
            features_pca = pca.fit_transform(features_scaled)
            
            self.pca_model = pca
            self.scaler = scaler
            
            return features_pca, pca, scaler
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Erreur lors de la PCA: {e}") from e
    
    def perform_tsne(
        self, 
        features_list: List[np.ndarray], 
        random_state: int = TSNE_DEFAULT_RANDOM_STATE, 
        perplexity: int = TSNE_DEFAULT_PERPLEXITY
    ) -> Tuple[Optional[np.ndarray], Optional[TSNE], Optional[StandardScaler]]:
        """
        Effectue une t-SNE sur les features extraites.
        
        Args:
            features_list: Liste des features extraites (arrays numpy).
            random_state: Seed pour la reproductibilité. Par défaut: TSNE_DEFAULT_RANDOM_STATE.
            perplexity: Perplexité pour t-SNE. Par défaut: TSNE_DEFAULT_PERPLEXITY.
        
        Returns:
            Tuple (features_tsne, tsne_model, scaler) ou (None, None, None) si erreur.
        
        Raises:
            ValueError: Si le nombre de features est insuffisant ou si perplexity est invalide.
            RuntimeError: Si la t-SNE échoue.
        """
        if len(features_list) < MIN_FEATURES_FOR_REDUCTION:
            return None, None, None
        
        try:
            features_array = np.array(features_list)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            n_samples = len(features_list)
            perplexity = min(perplexity, n_samples - 1)
            
            tsne = TSNE(
                n_components=TSNE_N_COMPONENTS,
                perplexity=perplexity,
                random_state=random_state,
                max_iter=TSNE_MAX_ITER
            )
            features_tsne = tsne.fit_transform(features_scaled)
            
            self.tsne_model = tsne
            self.scaler = scaler
            
            return features_tsne, tsne, scaler
        except (ValueError, TypeError) as e:
            raise RuntimeError(f"Erreur lors de la t-SNE: {e}") from e
    
    def perform_dimensionality_reduction(
        self, 
        features_list: List[np.ndarray], 
        method: str = 'pca', 
        **kwargs: Any
    ) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[StandardScaler]]:
        """
        Effectue une réduction de dimensionnalité (PCA ou t-SNE).
        
        Args:
            features_list: Liste des features extraites.
            method: Méthode à utiliser ('pca' ou 'tsne'). Par défaut: 'pca'.
            **kwargs: Arguments additionnels pour t-SNE (perplexity, random_state, etc.).
        
        Returns:
            Tuple (features_reduced, model, scaler).
        
        Raises:
            ValueError: Si la méthode n'est pas reconnue.
        """
        if method == 'tsne':
            return self.perform_tsne(features_list, **kwargs)
        elif method == 'pca':
            return self.perform_pca(features_list)
        else:
            raise ValueError(f"Méthode de réduction non reconnue: {method}")
    
    def generate_playlist_line(
        self, 
        pca_df: pd.DataFrame, 
        track1_idx: int, 
        track2_idx: int, 
        num_tracks: int = DEFAULT_PLAYLIST_SIZE
    ) -> Tuple[
        Optional[List[Dict[str, Any]]], 
        Optional[np.ndarray], 
        Optional[np.ndarray], 
        Optional[np.ndarray]
    ]:
        """
        Génère une playlist progressive (linéaire) entre deux morceaux.
        
        Args:
            pca_df: DataFrame contenant les tracks avec leurs coordonnées PC1 et PC2.
            track1_idx: Index de la track de départ.
            track2_idx: Index de la track d'arrivée.
            num_tracks: Nombre de tracks dans la playlist. Par défaut: DEFAULT_PLAYLIST_SIZE.
        
        Returns:
            Tuple (playlist_tracks, line_points, p1, p2) ou (None, None, None, None) si erreur.
        
        Raises:
            ValueError: Si les indices sont invalides ou identiques.
            KeyError: Si les colonnes PC1/PC2 sont manquantes dans le DataFrame.
            RuntimeError: Si la génération échoue.
        """
        try:
            if track1_idx >= len(pca_df) or track2_idx >= len(pca_df):
                raise ValueError(
                    f"Indices invalides : {track1_idx}, {track2_idx}. "
                    f"Le DataFrame contient {len(pca_df)} éléments."
                )
            
            if track1_idx == track2_idx:
                raise ValueError("Les deux tracks doivent être différentes")
            
            # Vérifier la présence des colonnes nécessaires
            required_columns = ['PC1', 'PC2', 'track_id', 'genre', 'name', 'artists', 'confidence']
            missing_columns = [col for col in required_columns if col not in pca_df.columns]
            if missing_columns:
                raise KeyError(f"Colonnes manquantes dans le DataFrame: {missing_columns}")
            
            # Points PCA
            p1 = np.array([pca_df.iloc[track1_idx]['PC1'], pca_df.iloc[track1_idx]['PC2']])
            p2 = np.array([pca_df.iloc[track2_idx]['PC1'], pca_df.iloc[track2_idx]['PC2']])
            t_values = np.linspace(0, 1, num_tracks)
            line_points = np.array([p1 + t * (p2 - p1) for t in t_values])
            
            playlist_tracks: List[Dict[str, Any]] = []
            used_tracks: set[str] = set()
            
            for i, target_point in enumerate(line_points):
                distances: List[Tuple[float, Any, ...]] = []
                for idx, row in pca_df.iterrows():
                    if row['track_id'] not in used_tracks:
                        track_point = np.array([row['PC1'], row['PC2']])
                        distance = np.linalg.norm(track_point - target_point)
                        distances.append((
                            distance, 
                            row['track_id'], 
                            row['genre'], 
                            track_point, 
                            row['name'], 
                            row['artists'], 
                            row['confidence'], 
                            row.get('uri'), 
                            row.get('spotify_id'), 
                            row.get('deezer_id'), 
                            row.get('preview_url')
                        ))
                
                if distances:
                    distances.sort(key=lambda x: x[0])
                    (
                        closest_distance, 
                        closest_track, 
                        closest_genre, 
                        closest_point, 
                        closest_name, 
                        closest_artists, 
                        closest_confidence, 
                        closest_uri, 
                        closest_spotify_id, 
                        closest_deezer_id, 
                        closest_preview_url
                    ) = distances[0]
                    
                    playlist_tracks.append({
                        'position': i + 1,
                        'track_id': idx,
                        'name': closest_name,
                        'artists': closest_artists,
                        'genre': closest_genre,
                        'confidence': closest_confidence,
                        'uri': closest_uri,
                        'spotify_id': closest_spotify_id,
                        'deezer_id': closest_deezer_id,
                        'preview_url': closest_preview_url,
                        'distance_to_line': closest_distance,
                        'target_point': target_point,
                        'actual_point': closest_point,
                        'PC1': closest_point[0],
                        'PC2': closest_point[1]
                    })
                    used_tracks.add(closest_track)
            
            return playlist_tracks, line_points, p1, p2
            
        except (ValueError, KeyError, IndexError) as e:
            raise RuntimeError(f"Erreur génération playlist: {e}") from e
    
    def analyze_playlist_quality(
        self, 
        playlist: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse la qualité d'une playlist générée.
        Compatible avec mode transition ET mode genre.
        
        Args:
            playlist: Liste des tracks de la playlist.
        
        Returns:
            Dictionnaire d'analyse ou None si la playlist est vide.
        
        Raises:
            KeyError: Si les clés attendues sont manquantes dans les tracks.
            RuntimeError: Si l'analyse échoue.
        """
        if not playlist:
            return None
        
        try:
            genres_in_playlist = [track['genre'] for track in playlist]
            unique_genres = len(set(genres_in_playlist))
            genre_distribution = pd.Series(genres_in_playlist).value_counts()
            
            distances = [track.get('distance_to_line', 0) for track in playlist]
            avg_distance = np.mean(distances) if distances else 0.0
            std_distance = np.std(distances) if distances else 0.0
            max_distance = np.max(distances) if distances else 0.0
            
            # Calcul de la fluidité (smoothness) - seulement si PC1/PC2 disponibles
            smoothness_distances: List[float] = []
            for i in range(len(playlist) - 1):
                if 'PC1' in playlist[i] and 'PC2' in playlist[i]:
                    p1 = np.array([playlist[i]['PC1'], playlist[i]['PC2']])
                    p2 = np.array([playlist[i+1]['PC1'], playlist[i+1]['PC2']])
                    smoothness_distances.append(np.linalg.norm(p2 - p1))
            
            avg_smoothness = (
                np.mean(smoothness_distances) 
                if smoothness_distances 
                else 0.0
            )
            
            analysis: Dict[str, Any] = {
                'num_tracks': len(playlist),
                'unique_genres': unique_genres,
                'genre_diversity_ratio': (
                    unique_genres / len(playlist) 
                    if playlist 
                    else 0.0
                ),
                'avg_distance_to_line': avg_distance,
                'std_distance_to_line': std_distance,
                'max_distance_to_line': max_distance,
                'avg_smoothness': avg_smoothness,
                'genre_distribution': genre_distribution.to_dict()
            }
            
            return analysis
            
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Erreur lors de l'analyse : {e}") from e
