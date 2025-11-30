"""Configuration centralisée pour l'application de génération de playlist."""

# --- Configuration Audio ---
AUDIO_DURATION_SECONDS: int = 30  # Durée d'extraction audio en secondes
SPECTROGRAM_SIZE: int = 599  # Taille du spectrogramme
SPECTROGRAM_N_FFT_DIVISOR: int = 10  # Diviseur pour calculer n_fft
FEATURE_DIMENSION: int = 2048  # Dimension des features extraites
FEATURE_VIEW_SIZE: int = 1536  # Taille pour le view des features

# --- Configuration Genres ---
NUM_GENRES: int = 10  # Nombre total de genres musicaux
GENRE_LABEL_MAPPING: dict[int, str] = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock",
}
GENRES_LIST: list[str] = list(GENRE_LABEL_MAPPING.values())

# Couleurs pour la visualisation des genres (Plotly)
GENRE_COLORS: dict[str, str] = {
    "blues": "#1f77b4",
    "classical": "#ff7f0e",
    "country": "#2ca02c",
    "disco": "#d62728",
    "hiphop": "#9467bd",
    "jazz": "#8c564b",
    "metal": "#e377c2",
    "pop": "#7f7f7f",
    "reggae": "#bcbd22",
    "rock": "#17becf",
}

# --- Configuration Fallback (quand modèle non chargé) ---
FALLBACK_CONFIDENCE_MIN: float = 0.6
FALLBACK_CONFIDENCE_MAX: float = 0.95

# --- Configuration Spotify ---
SPOTIFY_SCOPE: str = (
    "user-library-read user-top-read "
    "playlist-modify-public playlist-modify-private "
    "user-read-recently-played"
)
SPOTIFY_DEFAULT_REDIRECT_URI: str = "http://localhost:8501"
SPOTIFY_TRACK_ID_LENGTH: int = 22  # Longueur standard d'un ID Spotify
SPOTIFY_SEARCH_LIMIT_DEFAULT: int = 5  # Limite par défaut pour les recherches
SPOTIFY_SEARCH_LIMIT_MATCH: int = 5  # Limite pour le matching Deezer->Spotify
SPOTIFY_PLAYLIST_BATCH_SIZE: int = 100  # Taille des batches pour ajout de tracks

# --- Configuration Réduction de Dimensionnalité ---
PCA_N_COMPONENTS: int = 2  # Nombre de composantes pour PCA
TSNE_N_COMPONENTS: int = 2  # Nombre de composantes pour t-SNE
TSNE_DEFAULT_PERPLEXITY: int = 30
TSNE_DEFAULT_RANDOM_STATE: int = 42
TSNE_MAX_ITER: int = 1000
MIN_FEATURES_FOR_REDUCTION: int = 2  # Nombre minimum de features pour réduction

# --- Configuration Playlist ---
DEFAULT_PLAYLIST_SIZE: int = 10  # Taille par défaut d'une playlist générée
MIN_PLAYLIST_SIZE: int = 2  # Taille minimum d'une playlist

# --- Configuration Numérique ---
EPSILON: float = 1e-9  # Petite valeur pour éviter division par zéro

# --- Configuration Deezer ---
DEEZER_BASE_URL: str = "https://api.deezer.com"

# --- Configuration UI ---
# Langues supportées pour la traduction automatique
LANGUAGES: dict[str, str] = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ja": "🇯🇵 日本語",
    "zh-CN": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский",
}
