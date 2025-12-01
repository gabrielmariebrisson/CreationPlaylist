"""Tests unitaires pour la classe PlaylistPathfinder."""

import pytest
import numpy as np
import pandas as pd

from src.logic.playlist_generator import PlaylistPathfinder
from src.config import (
    FEATURE_VIEW_SIZE,
    PCA_N_COMPONENTS,
)


@pytest.fixture
def pathfinder() -> PlaylistPathfinder:
    """Fixture pour créer une instance de PlaylistPathfinder."""
    return PlaylistPathfinder()


@pytest.fixture
def sample_features(
    n_tracks: int = 20, feature_dim: int = FEATURE_VIEW_SIZE
) -> np.ndarray:
    """
    Crée un array de features aléatoires pour les tests.

    Args:
        n_tracks: Nombre de tracks.
        feature_dim: Dimension des features.

    Returns:
        Array numpy de shape (n_tracks, feature_dim).
    """
    np.random.seed(42)  # Pour la reproductibilité
    features = np.random.randn(n_tracks, feature_dim)
    # Normaliser pour que les features soient plus réalistes
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)
    return features


@pytest.fixture
def sample_tracks_df(n_tracks: int = 20) -> pd.DataFrame:
    """
    Crée un DataFrame de test avec des métadonnées de tracks.

    Args:
        n_tracks: Nombre de tracks.

    Returns:
        DataFrame avec les colonnes nécessaires.
    """
    genres = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    artists = ["Artist A", "Artist B", "Artist C", "Artist D", "Artist E"]

    np.random.seed(42)
    data = {
        "track_id": [f"track_{i}" for i in range(n_tracks)],
        "name": [f"Song {i}" for i in range(n_tracks)],
        "artists": [np.random.choice(artists) for _ in range(n_tracks)],
        "genre": [np.random.choice(genres) for _ in range(n_tracks)],
        "confidence": np.random.uniform(0.6, 0.95, n_tracks),
        "uri": [f"spotify:track:uri_{i}" for i in range(n_tracks)],
        "spotify_id": [f"spotify_id_{i}" for i in range(n_tracks)],
        "deezer_id": [f"deezer_id_{i}" for i in range(n_tracks)],
        "preview_url": [f"https://preview.url/{i}" for i in range(n_tracks)],
    }
    return pd.DataFrame(data)


class TestPlaylistPathfinder:
    """Tests pour la classe PlaylistPathfinder."""

    def test_initialization(self, pathfinder: PlaylistPathfinder):
        """Test l'initialisation de PlaylistPathfinder."""
        assert pathfinder.pca_model is None
        assert pathfinder.tsne_model is None
        assert pathfinder.scaler is None

    def test_cosine_similarity(self, pathfinder: PlaylistPathfinder):
        """Test le calcul de similarité cosinus."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = pathfinder._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0, abs=1e-6)

        vec3 = np.array([0.0, 1.0, 0.0])
        similarity_orthogonal = pathfinder._cosine_similarity(vec1, vec3)
        assert similarity_orthogonal == pytest.approx(0.0, abs=1e-6)

        vec4 = np.array([-1.0, 0.0, 0.0])
        similarity_opposite = pathfinder._cosine_similarity(vec1, vec4)
        assert similarity_opposite == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_distance(self, pathfinder: PlaylistPathfinder):
        """Test le calcul de distance cosinus."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        distance = pathfinder._cosine_distance(vec1, vec2)
        assert distance == pytest.approx(0.0, abs=1e-6)

        vec3 = np.array([0.0, 1.0, 0.0])
        distance_orthogonal = pathfinder._cosine_distance(vec1, vec3)
        assert distance_orthogonal == pytest.approx(1.0, abs=1e-6)

    def test_perform_pca(
        self, pathfinder: PlaylistPathfinder, sample_features: np.ndarray
    ):
        """Test la méthode perform_pca."""
        features_list = [sample_features[i] for i in range(len(sample_features))]

        result, pca_model, scaler = pathfinder.perform_pca(features_list)

        assert result is not None
        assert pca_model is not None
        assert scaler is not None
        assert result.shape[0] == len(features_list)
        assert result.shape[1] == PCA_N_COMPONENTS
        assert pathfinder.pca_model is not None
        assert pathfinder.scaler is not None

    def test_perform_pca_insufficient_features(self, pathfinder: PlaylistPathfinder):
        """Test perform_pca avec un nombre insuffisant de features."""
        features_list = [np.random.randn(FEATURE_VIEW_SIZE)]

        result, pca_model, scaler = pathfinder.perform_pca(features_list)

        assert result is None
        assert pca_model is None
        assert scaler is None

    def test_perform_tsne(
        self, pathfinder: PlaylistPathfinder, sample_features: np.ndarray
    ):
        """Test la méthode perform_tsne."""
        features_list = [sample_features[i] for i in range(len(sample_features))]

        result, tsne_model, scaler = pathfinder.perform_tsne(
            features_list, random_state=42, perplexity=5
        )

        assert result is not None
        assert tsne_model is not None
        assert scaler is not None
        assert result.shape[0] == len(features_list)
        assert result.shape[1] == 2
        assert pathfinder.tsne_model is not None
        assert pathfinder.scaler is not None

    def test_perform_dimensionality_reduction_pca(
        self, pathfinder: PlaylistPathfinder, sample_features: np.ndarray
    ):
        """Test perform_dimensionality_reduction avec méthode PCA."""
        features_list = [sample_features[i] for i in range(len(sample_features))]

        result, model, scaler = pathfinder.perform_dimensionality_reduction(
            features_list, method="pca"
        )

        assert result is not None
        assert model is not None
        assert scaler is not None
        assert result.shape[1] == PCA_N_COMPONENTS

    def test_perform_dimensionality_reduction_invalid_method(
        self, pathfinder: PlaylistPathfinder, sample_features: np.ndarray
    ):
        """Test perform_dimensionality_reduction avec méthode invalide."""
        features_list = [sample_features[i] for i in range(len(sample_features))]

        with pytest.raises(ValueError, match="Méthode de réduction non reconnue"):
            pathfinder.perform_dimensionality_reduction(features_list, method="invalid")

    def test_generate_playlist_line_basic(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la génération de playlist de base."""
        num_tracks = 5
        track1_idx = 0
        track2_idx = 5

        # Initialiser la PCA pour la visualisation
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        playlist, line_points_2d, p1_2d, p2_2d = pathfinder.generate_playlist_line(
            tracks_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=True,
        )

        assert playlist is not None
        assert len(playlist) == num_tracks, (
            f"Playlist devrait avoir {num_tracks} tracks, " f"mais a {len(playlist)}"
        )
        assert line_points_2d is not None
        assert p1_2d is not None
        assert p2_2d is not None
        assert line_points_2d.shape[0] == num_tracks

    def test_generate_playlist_line_length_variations(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la génération de playlist avec différentes longueurs."""
        track1_idx = 0
        track2_idx = 10

        # Initialiser la PCA
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        for num_tracks in [3, 5, 10, 15]:
            playlist, _, _, _ = pathfinder.generate_playlist_line(
                tracks_df=sample_tracks_df,
                raw_features=sample_features,
                track1_idx=track1_idx,
                track2_idx=track2_idx,
                num_tracks=num_tracks,
                use_pca_for_visualization=True,
            )

            assert playlist is not None
            assert len(playlist) == num_tracks, (
                f"Pour num_tracks={num_tracks}, la playlist devrait avoir "
                f"{num_tracks} tracks, mais a {len(playlist)}"
            )

    def test_generate_playlist_line_without_pca(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la génération de playlist sans PCA (uniquement pour le calcul)."""
        num_tracks = 5
        track1_idx = 0
        track2_idx = 5

        playlist, line_points_2d, p1_2d, p2_2d = pathfinder.generate_playlist_line(
            tracks_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=False,
        )

        assert playlist is not None
        assert len(playlist) == num_tracks
        # Sans PCA, les coordonnées 2D peuvent être None
        # Mais la playlist devrait quand même être générée

    def test_generate_playlist_line_invalid_indices(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la génération avec des indices invalides."""
        with pytest.raises(RuntimeError, match="Indices invalides"):
            pathfinder.generate_playlist_line(
                tracks_df=sample_tracks_df,
                raw_features=sample_features,
                track1_idx=100,  # Index invalide
                track2_idx=0,
                num_tracks=5,
            )

    def test_generate_playlist_line_same_indices(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la génération avec les mêmes indices."""
        with pytest.raises(
            RuntimeError, match="Les deux tracks doivent être différentes"
        ):
            pathfinder.generate_playlist_line(
                tracks_df=sample_tracks_df,
                raw_features=sample_features,
                track1_idx=0,
                track2_idx=0,  # Même index
                num_tracks=5,
            )

    def test_generate_playlist_line_mismatched_features(
        self, pathfinder: PlaylistPathfinder, sample_tracks_df: pd.DataFrame
    ):
        """Test la génération avec un nombre de features qui ne correspond pas."""
        # 10 features au lieu de 20
        wrong_features = np.random.randn(10, FEATURE_VIEW_SIZE)

        with pytest.raises(RuntimeError, match="ne correspond pas"):
            pathfinder.generate_playlist_line(
                tracks_df=sample_tracks_df,
                raw_features=wrong_features,
                track1_idx=0,
                track2_idx=5,
                num_tracks=5,
            )

    def test_generate_playlist_line_missing_columns(
        self, pathfinder: PlaylistPathfinder, sample_features: np.ndarray
    ):
        """Test la génération avec des colonnes manquantes."""
        incomplete_df = pd.DataFrame(
            {
                "track_id": ["track_0", "track_1"],
                "name": ["Song 0", "Song 1"],
                # Manque 'genre', 'artists', 'confidence'
            }
        )
        incomplete_features = sample_features[:2]

        with pytest.raises(RuntimeError, match="Colonnes manquantes"):
            pathfinder.generate_playlist_line(
                tracks_df=incomplete_df,
                raw_features=incomplete_features,
                track1_idx=0,
                track2_idx=1,
                num_tracks=2,
            )

    def test_generate_playlist_line_playlist_structure(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la structure des tracks dans la playlist générée."""
        num_tracks = 5
        track1_idx = 0
        track2_idx = 5

        # Initialiser la PCA
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        playlist, _, _, _ = pathfinder.generate_playlist_line(
            tracks_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=True,
        )

        assert playlist is not None
        assert len(playlist) == num_tracks

        # Vérifier la structure de chaque track
        required_keys = [
            "position",
            "track_id",
            "name",
            "artists",
            "genre",
            "confidence",
            "cosine_similarity",
            "cosine_distance",
        ]

        for i, track in enumerate(playlist):
            assert track["position"] == i + 1
            for key in required_keys:
                assert key in track, f"Clé '{key}' manquante dans le track {i+1}"

            # Vérifier que cosine_similarity est entre -1 et 1
            assert -1.0 <= track["cosine_similarity"] <= 1.0
            # Vérifier que cosine_distance est entre 0 et 2
            assert 0.0 <= track["cosine_distance"] <= 2.0

    def test_generate_playlist_line_no_duplicates(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test qu'il n'y a pas de doublons dans la playlist."""
        num_tracks = 10
        track1_idx = 0
        track2_idx = 5

        # Initialiser la PCA
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        playlist, _, _, _ = pathfinder.generate_playlist_line(
            tracks_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=True,
        )

        assert playlist is not None

        # Vérifier qu'il n'y a pas de doublons
        track_ids = [track["track_id"] for track in playlist]
        assert len(track_ids) == len(
            set(track_ids)
        ), "Il y a des doublons dans la playlist"

    def test_generate_playlist_line_from_pca_df(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test le wrapper generate_playlist_line_from_pca_df."""
        num_tracks = 5
        track1_idx = 0
        track2_idx = 5

        # Initialiser la PCA
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        (
            playlist,
            line_points_2d,
            p1_2d,
            p2_2d,
        ) = pathfinder.generate_playlist_line_from_pca_df(
            pca_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
        )

        assert playlist is not None
        assert len(playlist) == num_tracks
        assert line_points_2d is not None

    def test_analyze_playlist_quality(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test l'analyse de qualité de playlist."""
        num_tracks = 5
        track1_idx = 0
        track2_idx = 5

        # Initialiser la PCA
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        playlist, _, _, _ = pathfinder.generate_playlist_line(
            tracks_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=True,
        )

        analysis = pathfinder.analyze_playlist_quality(playlist)

        assert analysis is not None
        assert analysis["num_tracks"] == num_tracks
        assert analysis["unique_genres"] > 0
        assert 0.0 <= analysis["genre_diversity_ratio"] <= 1.0
        assert "avg_cosine_similarity" in analysis
        assert "avg_distance_to_line" in analysis
        assert "genre_distribution" in analysis

    def test_analyze_playlist_quality_empty(self, pathfinder: PlaylistPathfinder):
        """Test l'analyse avec une playlist vide."""
        analysis = pathfinder.analyze_playlist_quality([])
        assert analysis is None

    def test_analyze_playlist_quality_structure(
        self,
        pathfinder: PlaylistPathfinder,
        sample_tracks_df: pd.DataFrame,
        sample_features: np.ndarray,
    ):
        """Test la structure de l'analyse de qualité."""
        num_tracks = 5
        track1_idx = 0
        track2_idx = 5

        # Initialiser la PCA
        features_list = [sample_features[i] for i in range(len(sample_features))]
        pathfinder.perform_pca(features_list)

        playlist, _, _, _ = pathfinder.generate_playlist_line(
            tracks_df=sample_tracks_df,
            raw_features=sample_features,
            track1_idx=track1_idx,
            track2_idx=track2_idx,
            num_tracks=num_tracks,
            use_pca_for_visualization=True,
        )

        analysis = pathfinder.analyze_playlist_quality(playlist)

        required_keys = [
            "num_tracks",
            "unique_genres",
            "genre_diversity_ratio",
            "avg_cosine_similarity",
            "avg_distance_to_line",
            "std_distance_to_line",
            "max_distance_to_line",
            "avg_smoothness",
            "genre_distribution",
        ]

        for key in required_keys:
            assert key in analysis, f"Clé '{key}' manquante dans l'analyse"
