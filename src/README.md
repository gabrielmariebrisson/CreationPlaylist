# Structure modulaire - Music Playlist Generator

Cette structure respecte le principe de responsabilité unique (SRP) en séparant les différentes responsabilités du code monolithique.

## Structure

```
src/
├── services/
│   └── spotify_service.py      # Gestion de l'API Spotify
├── models/
│   └── audio_classifier.py      # Classification audio avec CNN
└── logic/
    └── playlist_generator.py    # Génération de playlist
```

## Utilisation

### SpotifyService

```python
from src.services.spotify_service import SpotifyService
from spotipy.oauth2 import SpotifyOAuth

# Avec injection de dépendance
auth_manager = SpotifyOAuth(...)
service = SpotifyService(auth_manager=auth_manager, session_state=st.session_state)

# Ou sans injection (crée automatiquement depuis .env)
service = SpotifyService(session_state=st.session_state)

# Utilisation
client = service.get_client()
playlist = service.export_playlist(
    playlist_tracks, 
    "Ma Playlist",
    "Description",
    callback_info=st.info,
    callback_warning=st.warning,
    callback_success=st.success,
    callback_error=st.error
)
```

### AudioClassifier

```python
from src.models.audio_classifier import AudioClassifier

classifier = AudioClassifier("templates/assets/music/best_model_original_loss.pth")

result = classifier.predict("path/to/audio.mp3")
print(f"Genre: {result['genre']}, Confiance: {result['confidence']:.2%}")
```

### PlaylistPathfinder

```python
from src.logic.playlist_generator import PlaylistPathfinder
import pandas as pd

pathfinder = PlaylistPathfinder()

# Réduction de dimensionnalité
features_list = [track['features'] for track in tracks]
features_2d, model, scaler = pathfinder.perform_dimensionality_reduction(
    features_list, 
    method='pca'
)

# Créer DataFrame
pca_df = pd.DataFrame({
    'PC1': features_2d[:, 0],
    'PC2': features_2d[:, 1],
    'name': [t['name'] for t in tracks],
    # ... autres colonnes
})

# Générer playlist
playlist, line_points, p1, p2 = pathfinder.generate_playlist_line(
    pca_df, 
    track1_idx=0, 
    track2_idx=5, 
    num_tracks=10
)

# Analyser la qualité
analysis = pathfinder.analyze_playlist_quality(playlist)
```

## Migration depuis le code monolithique

Le code original dans `CreationPlaylist.py` reste intact. Pour migrer progressivement :

1. Remplacer les appels de fonctions par les nouvelles classes
2. Adapter les callbacks pour les messages Streamlit
3. Tester chaque composant indépendamment

