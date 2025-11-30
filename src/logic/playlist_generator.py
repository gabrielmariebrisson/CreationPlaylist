"""Générateur de playlist basé sur la réduction de dimensionnalité."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class PlaylistPathfinder:
    """Classe pour générer des playlists basées sur la similarité musicale."""
    
    def __init__(self):
        """Initialise le générateur de playlist."""
        self.pca_model: Optional[PCA] = None
        self.tsne_model: Optional[TSNE] = None
        self.scaler: Optional[StandardScaler] = None
    
    def perform_pca(self, features_list: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[PCA], Optional[StandardScaler]]:
        """
        Effectue une PCA sur les features extraites.
        
        Args:
            features_list: Liste des features extraites (arrays numpy)
            
        Returns:
            Tuple (features_pca, pca_model, scaler) ou (None, None, None) si erreur
        """
        if len(features_list) < 2:
            return None, None, None
        
        try:
            features_array = np.array(features_list)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            pca = PCA(n_components=2)
            features_pca = pca.fit_transform(features_scaled)
            
            self.pca_model = pca
            self.scaler = scaler
            
            return features_pca, pca, scaler
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la PCA: {e}")
    
    def perform_tsne(
        self, 
        features_list: List[np.ndarray], 
        random_state: int = 42, 
        perplexity: int = 30
    ) -> Tuple[Optional[np.ndarray], Optional[TSNE], Optional[StandardScaler]]:
        """
        Effectue une t-SNE sur les features extraites.
        
        Args:
            features_list: Liste des features extraites (arrays numpy)
            random_state: Seed pour la reproductibilité
            perplexity: Perplexité pour t-SNE
            
        Returns:
            Tuple (features_tsne, tsne_model, scaler) ou (None, None, None) si erreur
        """
        if len(features_list) < 2:
            return None, None, None
        
        try:
            features_array = np.array(features_list)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            n_samples = len(features_list)
            perplexity = min(perplexity, n_samples - 1)
            
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, max_iter=1000)
            features_tsne = tsne.fit_transform(features_scaled)
            
            self.tsne_model = tsne
            self.scaler = scaler
            
            return features_tsne, tsne, scaler
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la t-SNE: {e}")
    
    def perform_dimensionality_reduction(
        self, 
        features_list: List[np.ndarray], 
        method: str = 'pca', 
        **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[Any], Optional[StandardScaler]]:
        """
        Effectue une réduction de dimensionnalité (PCA ou t-SNE).
        
        Args:
            features_list: Liste des features extraites
            method: Méthode à utiliser ('pca' ou 'tsne')
            **kwargs: Arguments additionnels pour t-SNE (perplexity, random_state, etc.)
            
        Returns:
            Tuple (features_reduced, model, scaler)
        """
        if method == 'tsne':
            return self.perform_tsne(features_list, **kwargs)
        else:
            return self.perform_pca(features_list)
    
    def generate_playlist_line(
        self, 
        pca_df: pd.DataFrame, 
        track1_idx: int, 
        track2_idx: int, 
        num_tracks: int = 10
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Génère une playlist progressive (linéaire) entre deux morceaux.
        
        Args:
            pca_df: DataFrame contenant les tracks avec leurs coordonnées PC1 et PC2
            track1_idx: Index de la track de départ
            track2_idx: Index de la track d'arrivée
            num_tracks: Nombre de tracks dans la playlist
            
        Returns:
            Tuple (playlist_tracks, line_points, p1, p2) ou (None, None, None, None) si erreur
        """
        try:
            if track1_idx >= len(pca_df) or track2_idx >= len(pca_df):
                raise ValueError(f"Indices invalides : {track1_idx}, {track2_idx}")
            
            if track1_idx == track2_idx:
                raise ValueError("Les deux tracks doivent être différentes")
            
            # Points PCA
            p1 = np.array([pca_df.iloc[track1_idx]['PC1'], pca_df.iloc[track1_idx]['PC2']])
            p2 = np.array([pca_df.iloc[track2_idx]['PC1'], pca_df.iloc[track2_idx]['PC2']])
            t_values = np.linspace(0, 1, num_tracks)
            line_points = np.array([p1 + t * (p2 - p1) for t in t_values])
            
            playlist_tracks = []
            used_tracks = set()
            
            for i, target_point in enumerate(line_points):
                distances = []
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
            
        except Exception as e:
            raise RuntimeError(f"Erreur génération playlist: {e}")
    
    def analyze_playlist_quality(self, playlist: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyse la qualité d'une playlist générée.
        Compatible avec mode transition ET mode genre.
        
        Args:
            playlist: Liste des tracks de la playlist
            
        Returns:
            Dictionnaire d'analyse ou None si erreur
        """
        if not playlist:
            return None
        
        try:
            genres_in_playlist = [track['genre'] for track in playlist]
            unique_genres = len(set(genres_in_playlist))
            genre_distribution = pd.Series(genres_in_playlist).value_counts()
            
            distances = [track.get('distance_to_line', 0) for track in playlist]
            avg_distance = np.mean(distances) if distances else 0
            std_distance = np.std(distances) if distances else 0
            max_distance = np.max(distances) if distances else 0
            
            # Calcul de la fluidité (smoothness) - seulement si PC1/PC2 disponibles
            smoothness_distances = []
            for i in range(len(playlist) - 1):
                if 'PC1' in playlist[i] and 'PC2' in playlist[i]:
                    p1 = np.array([playlist[i]['PC1'], playlist[i]['PC2']])
                    p2 = np.array([playlist[i+1]['PC1'], playlist[i+1]['PC2']])
                    smoothness_distances.append(np.linalg.norm(p2 - p1))
            
            avg_smoothness = np.mean(smoothness_distances) if smoothness_distances else 0
            
            analysis = {
                'num_tracks': len(playlist),
                'unique_genres': unique_genres,
                'genre_diversity_ratio': unique_genres / len(playlist) if playlist else 0,
                'avg_distance_to_line': avg_distance,
                'std_distance_to_line': std_distance,
                'max_distance_to_line': max_distance,
                'avg_smoothness': avg_smoothness,
                'genre_distribution': genre_distribution.to_dict()
            }
            
            return analysis
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de l'analyse : {str(e)}")

