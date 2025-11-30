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
    FEATURE_VIEW_SIZE,
)


class PlaylistPathfinder:
    """Classe pour générer des playlists basées sur la similarité musicale."""
    
    def __init__(self) -> None:
        """Initialise le générateur de playlist."""
        self.pca_model: Optional[PCA] = None
        self.tsne_model: Optional[TSNE] = None
        self.scaler: Optional[StandardScaler] = None
    
    def _cosine_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs.
        
        Args:
            vec1: Premier vecteur.
            vec2: Deuxième vecteur.
        
        Returns:
            Similarité cosinus (entre -1 et 1, mais généralement entre 0 et 1 pour des features normalisées).
        """
        # Normaliser les vecteurs pour éviter les problèmes numériques
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-9)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-9)
        return np.dot(vec1_norm, vec2_norm)
    
    def _cosine_distance(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """
        Calcule la distance cosinus (1 - cosine similarity) entre deux vecteurs.
        
        Args:
            vec1: Premier vecteur.
            vec2: Deuxième vecteur.
        
        Returns:
            Distance cosinus (entre 0 et 2, où 0 = identique, 2 = opposé).
        """
        similarity = self._cosine_similarity(vec1, vec2)
        return 1.0 - similarity
    
    def perform_pca(
        self, 
        features_list: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[PCA], Optional[StandardScaler]]:
        """
        Effectue une PCA sur les features extraites.
        Utilisé uniquement pour la visualisation, pas pour le calcul de similarité.
        
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
        Utilisé uniquement pour la visualisation, pas pour le calcul de similarité.
        
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
        Utilisé uniquement pour la visualisation, pas pour le calcul de similarité.
        
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
        tracks_df: pd.DataFrame,
        raw_features: np.ndarray,
        track1_idx: int, 
        track2_idx: int, 
        num_tracks: int = DEFAULT_PLAYLIST_SIZE,
        use_pca_for_visualization: bool = True
    ) -> Tuple[
        Optional[List[Dict[str, Any]]], 
        Optional[np.ndarray], 
        Optional[np.ndarray], 
        Optional[np.ndarray]
    ]:
        """
        Génère une playlist progressive (linéaire) entre deux morceaux.
        
        Utilise les embeddings bruts (1536 dimensions) pour l'interpolation et la recherche
        de voisins avec cosine similarity. La PCA est utilisée uniquement pour la visualisation.
        
        Args:
            tracks_df: DataFrame contenant les métadonnées des tracks (name, artists, genre, etc.).
            raw_features: Array numpy de shape (n_tracks, feature_dim) contenant les features brutes.
            track1_idx: Index de la track de départ.
            track2_idx: Index de la track d'arrivée.
            num_tracks: Nombre de tracks dans la playlist. Par défaut: DEFAULT_PLAYLIST_SIZE.
            use_pca_for_visualization: Si True, calcule les coordonnées PCA pour la visualisation.
        
        Returns:
            Tuple (playlist_tracks, line_points_2d, p1_2d, p2_2d) où:
                - playlist_tracks: Liste des tracks de la playlist avec métadonnées.
                - line_points_2d: Points interpolés en 2D (pour visualisation).
                - p1_2d: Point de départ en 2D (pour visualisation).
                - p2_2d: Point d'arrivée en 2D (pour visualisation).
        
        Raises:
            ValueError: Si les indices sont invalides ou identiques.
            KeyError: Si les colonnes nécessaires sont manquantes dans le DataFrame.
            RuntimeError: Si la génération échoue.
        """
        try:
            # Validation des entrées
            if track1_idx >= len(tracks_df) or track2_idx >= len(tracks_df):
                raise ValueError(
                    f"Indices invalides : {track1_idx}, {track2_idx}. "
                    f"Le DataFrame contient {len(tracks_df)} éléments."
                )
            
            if track1_idx == track2_idx:
                raise ValueError("Les deux tracks doivent être différentes")
            
            if len(raw_features) != len(tracks_df):
                raise ValueError(
                    f"Le nombre de features ({len(raw_features)}) ne correspond pas "
                    f"au nombre de tracks ({len(tracks_df)})"
                )
            
            # Vérifier la présence des colonnes nécessaires
            required_columns = ['track_id', 'genre', 'name', 'artists', 'confidence']
            missing_columns = [col for col in required_columns if col not in tracks_df.columns]
            if missing_columns:
                raise KeyError(f"Colonnes manquantes dans le DataFrame: {missing_columns}")
            
            # Normaliser les features brutes pour une meilleure interpolation
            features_normalized = raw_features / (
                np.linalg.norm(raw_features, axis=1, keepdims=True) + 1e-9
            )
            
            # Extraire les features des deux tracks de référence
            feature1 = features_normalized[track1_idx]
            feature2 = features_normalized[track2_idx]
            
            # Interpolation linéaire dans l'espace des features brutes
            t_values = np.linspace(0, 1, num_tracks)
            interpolated_features = np.array([
                (1 - t) * feature1 + t * feature2 for t in t_values
            ])
            
            # Normaliser les features interpolées
            interpolated_features = interpolated_features / (
                np.linalg.norm(interpolated_features, axis=1, keepdims=True) + 1e-9
            )
            
            # Calculer les coordonnées PCA pour la visualisation (si demandé)
            p1_2d: Optional[np.ndarray] = None
            p2_2d: Optional[np.ndarray] = None
            line_points_2d: Optional[np.ndarray] = None
            
            if use_pca_for_visualization and self.pca_model is not None and self.scaler is not None:
                # Transformer les features brutes en coordonnées PCA pour la visualisation
                features_scaled = self.scaler.transform(raw_features)
                features_pca = self.pca_model.transform(features_scaled)
                
                p1_2d = features_pca[track1_idx]
                p2_2d = features_pca[track2_idx]
                
                # Interpoler aussi en 2D pour la visualisation
                line_points_2d = np.array([
                    (1 - t) * p1_2d + t * p2_2d for t in t_values
                ])
            
            # Recherche des voisins les plus proches avec cosine similarity
            playlist_tracks: List[Dict[str, Any]] = []
            used_tracks: set[str] = set()
            
            for i, target_feature in enumerate(interpolated_features):
                similarities: List[Tuple[float, Any, ...]] = []
                
                for idx, row in tracks_df.iterrows():
                    if row['track_id'] not in used_tracks:
                        track_feature = features_normalized[idx]
                        
                        # Utiliser cosine similarity (plus grande = plus similaire)
                        similarity = self._cosine_similarity(target_feature, track_feature)
                        # Convertir en distance pour trier (plus petite = plus similaire)
                        distance = 1.0 - similarity
                        
                        # Récupérer les coordonnées PCA pour la visualisation
                        track_pca_point = None
                        if use_pca_for_visualization and self.pca_model is not None:
                            if hasattr(self, '_features_pca_cache'):
                                track_pca_point = self._features_pca_cache[idx]
                            else:
                                # Calculer à la volée si pas en cache
                                feature_scaled = self.scaler.transform([raw_features[idx]])
                                track_pca_point = self.pca_model.transform(feature_scaled)[0]
                        
                        similarities.append((
                            distance,  # Distance cosinus (pour trier)
                            similarity,  # Similarité cosinus (pour métrique)
                            row['track_id'], 
                            row['genre'], 
                            track_feature,  # Feature brute
                            track_pca_point,  # Coordonnées PCA pour visualisation
                            row['name'], 
                            row['artists'], 
                            row['confidence'], 
                            row.get('uri'), 
                            row.get('spotify_id'), 
                            row.get('deezer_id'), 
                            row.get('preview_url')
                        ))
                
                if similarities:
                    # Trier par distance (plus petite = plus similaire)
                    similarities.sort(key=lambda x: x[0])
                    (
                        cosine_distance,
                        cosine_similarity_val,
                        closest_track, 
                        closest_genre, 
                        closest_feature,
                        closest_pca_point,
                        closest_name, 
                        closest_artists, 
                        closest_confidence, 
                        closest_uri, 
                        closest_spotify_id, 
                        closest_deezer_id, 
                        closest_preview_url
                    ) = similarities[0]
                    
                    # Extraire les coordonnées PCA pour la visualisation
                    pc1 = closest_pca_point[0] if closest_pca_point is not None else 0.0
                    pc2 = closest_pca_point[1] if closest_pca_point is not None else 0.0
                    
                    playlist_tracks.append({
                        'position': i + 1,
                        'track_id': closest_track,
                        'name': closest_name,
                        'artists': closest_artists,
                        'genre': closest_genre,
                        'confidence': closest_confidence,
                        'uri': closest_uri,
                        'spotify_id': closest_spotify_id,
                        'deezer_id': closest_deezer_id,
                        'preview_url': closest_preview_url,
                        'cosine_similarity': cosine_similarity_val,  # Nouvelle métrique
                        'cosine_distance': cosine_distance,  # Pour compatibilité
                        'distance_to_line': cosine_distance,  # Alias pour compatibilité
                        'target_feature': target_feature,  # Feature interpolée cible
                        'actual_feature': closest_feature,  # Feature réelle du track
                        'PC1': pc1,  # Pour visualisation
                        'PC2': pc2,  # Pour visualisation
                    })
                    used_tracks.add(closest_track)
            
            return playlist_tracks, line_points_2d, p1_2d, p2_2d
            
        except (ValueError, KeyError, IndexError) as e:
            raise RuntimeError(f"Erreur génération playlist: {e}") from e
    
    def generate_playlist_line_from_pca_df(
        self, 
        pca_df: pd.DataFrame, 
        raw_features: np.ndarray,
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
        Wrapper pour compatibilité avec l'ancienne interface.
        
        Génère une playlist en utilisant les features brutes pour le calcul,
        mais accepte un DataFrame avec colonnes PCA pour la compatibilité.
        
        Args:
            pca_df: DataFrame contenant les métadonnées (peut contenir PC1/PC2 pour compatibilité).
            raw_features: Array numpy de shape (n_tracks, feature_dim) contenant les features brutes.
            track1_idx: Index de la track de départ.
            track2_idx: Index de la track d'arrivée.
            num_tracks: Nombre de tracks dans la playlist.
        
        Returns:
            Tuple (playlist_tracks, line_points_2d, p1_2d, p2_2d).
        """
        return self.generate_playlist_line(
            tracks_df=pca_df,
            raw_features=raw_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=True
        )
    
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
            
            # Utiliser cosine_similarity si disponible, sinon distance_to_line
            similarities = [
                track.get('cosine_similarity', 1.0 - track.get('distance_to_line', 0.0))
                for track in playlist
            ]
            distances = [
                track.get('cosine_distance', track.get('distance_to_line', 0.0))
                for track in playlist
            ]
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            avg_distance = np.mean(distances) if distances else 0.0
            std_distance = np.std(distances) if distances else 0.0
            max_distance = np.max(distances) if distances else 0.0
            
            # Calcul de la fluidité (smoothness) dans l'espace des features
            smoothness_distances: List[float] = []
            for i in range(len(playlist) - 1):
                if 'actual_feature' in playlist[i] and 'actual_feature' in playlist[i+1]:
                    feat1 = playlist[i]['actual_feature']
                    feat2 = playlist[i+1]['actual_feature']
                    smoothness_distances.append(self._cosine_distance(feat1, feat2))
                elif 'PC1' in playlist[i] and 'PC2' in playlist[i]:
                    # Fallback sur PCA si features brutes non disponibles
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
                'avg_cosine_similarity': avg_similarity,
                'avg_distance_to_line': avg_distance,
                'std_distance_to_line': std_distance,
                'max_distance_to_line': max_distance,
                'avg_smoothness': avg_smoothness,
                'genre_distribution': genre_distribution.to_dict()
            }
            
            return analysis
            
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"Erreur lors de l'analyse : {e}") from e
