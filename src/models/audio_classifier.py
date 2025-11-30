"""Classe pour la classification audio avec CNN."""

import os
import sys
from pathlib import Path
from typing import Optional, Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    AUDIO_DURATION_SECONDS,
    SPECTROGRAM_SIZE,
    SPECTROGRAM_N_FFT_DIVISOR,
    FEATURE_DIMENSION,
    FEATURE_VIEW_SIZE,
    NUM_GENRES,
    GENRE_LABEL_MAPPING,
    FALLBACK_CONFIDENCE_MIN,
    FALLBACK_CONFIDENCE_MAX,
    EPSILON,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ajouter le chemin du module architecture au path
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent
_architecture_path = _project_root / 'templates' / 'assets' / 'music'
sys.path.insert(0, str(_architecture_path))

try:
    from architecture import SimpleCNN
except ImportError as e:
    raise ImportError(
        f"Impossible d'importer SimpleCNN. "
        f"Vérifiez que le fichier architecture.py existe dans {_architecture_path}"
    ) from e


class AudioClassifier:
    """Classe pour charger le modèle PyTorch et effectuer des prédictions de genre musical."""
    
    # Mapping des labels de genre (référence vers config)
    LABEL_MAPPING: dict[int, str] = GENRE_LABEL_MAPPING
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None) -> None:
        """
        Initialise le classifieur audio.
        
        Args:
            model_path: Chemin vers le fichier de modèle PyTorch (.pth).
            device: Device PyTorch (CPU ou CUDA). Si None, utilise CPU.
        
        Raises:
            FileNotFoundError: Si le fichier de modèle n'existe pas.
            RuntimeError: Si le chargement du modèle échoue.
        """
        self.model_path = model_path
        self.device = device or torch.device('cpu')
        self.model: Optional[SimpleCNN] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Charge le modèle CNN pré-entraîné.
        
        Raises:
            FileNotFoundError: Si le fichier de modèle n'existe pas.
            RuntimeError: Si le chargement du modèle échoue.
        """
        if not os.path.exists(self.model_path):
            logger.error(
                "Fichier de modèle non trouvé",
                extra={"extra": {"model_path": self.model_path}}
            )
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        try:
            logger.info(
                "Chargement du modèle CNN",
                extra={"extra": {
                    "model_path": self.model_path,
                    "device": str(self.device)
                }}
            )
            
            self.model = SimpleCNN()
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            logger.info(
                "Modèle CNN chargé avec succès",
                extra={"extra": {
                    "model_path": self.model_path,
                    "device": str(self.device),
                    "num_genres": NUM_GENRES
                }}
            )
        except OSError as e:
            logger.exception(
                "Erreur OS lors du chargement du modèle",
                extra={"extra": {
                    "model_path": self.model_path,
                    "error_type": "OSError"
                }}
            )
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}") from e
        except KeyError as e:
            logger.exception(
                "Clé manquante dans le state_dict",
                extra={"extra": {
                    "model_path": self.model_path,
                    "error_type": "KeyError",
                    "missing_key": str(e)
                }}
            )
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}") from e
        except RuntimeError as e:
            logger.exception(
                "Erreur runtime lors du chargement du modèle",
                extra={"extra": {
                    "model_path": self.model_path,
                    "error_type": "RuntimeError"
                }}
            )
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}") from e
    
    def convert_song_to_matrix(
        self, 
        audio_path: str, 
        size: int = SPECTROGRAM_SIZE
    ) -> np.ndarray:
        """
        Convertit un fichier audio en spectrogramme normalisé.
        
        Args:
            audio_path: Chemin vers le fichier audio.
            size: Taille cible du spectrogramme. Par défaut: SPECTROGRAM_SIZE.
        
        Returns:
            Spectrogramme normalisé sous forme de numpy array.
        
        Raises:
            FileNotFoundError: Si le fichier audio n'existe pas.
            RuntimeError: Si la conversion audio échoue.
        """
        if not os.path.exists(audio_path):
            logger.error(
                "Fichier audio non trouvé",
                extra={"extra": {"audio_path": audio_path}}
            )
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")
        
        try:
            logger.debug(
                "Conversion audio en spectrogramme",
                extra={"extra": {
                    "audio_path": audio_path,
                    "target_size": size,
                    "duration_seconds": AUDIO_DURATION_SECONDS
                }}
            )
            
            y, sr = librosa.load(audio_path, duration=AUDIO_DURATION_SECONDS)
            n_fft = int((sr / SPECTROGRAM_N_FFT_DIVISOR) / 2 + 3)
            D = np.abs(librosa.stft(y, hop_length=int(n_fft)))
            spectrogram = librosa.feature.melspectrogram(S=D, sr=sr)
            S = librosa.util.fix_length(spectrogram, size=size)
            S_db = librosa.power_to_db(S, ref=np.max)
            S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + EPSILON)
            
            logger.debug(
                "Spectrogramme généré avec succès",
                extra={"extra": {
                    "audio_path": audio_path,
                    "spectrogram_shape": S_db_norm.shape,
                    "sample_rate": sr
                }}
            )
            
            return S_db_norm
        except librosa.util.exceptions.NoDataError as e:
            logger.exception(
                "Erreur: pas de données audio",
                extra={"extra": {
                    "audio_path": audio_path,
                    "error_type": "NoDataError"
                }}
            )
            raise RuntimeError(f"Erreur conversion audio: {e}") from e
        except OSError as e:
            logger.exception(
                "Erreur OS lors de la conversion audio",
                extra={"extra": {
                    "audio_path": audio_path,
                    "error_type": "OSError"
                }}
            )
            raise RuntimeError(f"Erreur conversion audio: {e}") from e
    
    def extract_features(self, spectrogram_tensor: torch.Tensor) -> np.ndarray:
        """
        Extrait les features avant la couche de classification.
        
        Args:
            spectrogram_tensor: Tenseur du spectrogramme (shape: [1, 1, H, W]).
        
        Returns:
            Features extraites sous forme de numpy array.
        
        Raises:
            RuntimeError: Si le modèle n'est pas chargé.
        """
        if self.model is None:
            logger.error("Tentative d'extraction de features avec modèle non chargé")
            raise RuntimeError("Modèle non chargé")
        
        with torch.no_grad():
            x = spectrogram_tensor.to(self.device)
            x = self.model.norm1(x)
            x = self.model.conv1(x)
            x = self.model.relu(x)
            x = self.model.cbam1(x)
            x = self.model.pool1(x)
            x = torch.permute(x, (0, 2, 1, 3))
            
            x = self.model.norm2(x)
            x = self.model.conv2(x)
            x = self.model.relu(x)
            x = self.model.cbam1(x)
            x = self.model.pool2(x)
            x = torch.permute(x, (0, 2, 1, 3))
            
            x = self.model.norm3(x)
            x = self.model.conv3(x)
            x = self.model.relu(x)
            x = self.model.cbam2(x)
            x = self.model.pool2(x)
            x = torch.permute(x, (0, 2, 1, 3))
            
            x = self.model.norm4(x)
            x = self.model.conv4(x)
            x = self.model.cbam2(x)
            x = self.model.relu(x)
            x = torch.permute(x, (0, 2, 1, 3))
            
            mean_values = torch.mean(x, dim=3, keepdim=True)
            max_values, _ = torch.max(x, dim=3, keepdim=True)
            l2_norm = torch.linalg.norm(x, dim=3, ord=2, keepdim=True)
            
            x = torch.cat([max_values, mean_values, l2_norm], dim=1)
            x = x.view(-1, FEATURE_VIEW_SIZE)
            
            x = self.model.normfc2(x)
            x = self.model.fc2(x)
            features = F.relu(x)
            
            return features.cpu().numpy()
    
    def predict(
        self, 
        audio_path: str, 
        return_features: bool = True,
        return_probabilities: bool = True
    ) -> dict[str, Any]:
        """
        Analyse un fichier audio et prédit son genre.
        
        Args:
            audio_path: Chemin vers le fichier audio.
            return_features: Si True, retourne aussi les features extraites.
            return_probabilities: Si True, retourne toutes les probabilités par genre.
        
        Returns:
            Dictionnaire contenant:
                - 'genre': Genre prédit (str).
                - 'confidence': Confiance de la prédiction (float).
                - 'features': Features extraites (np.ndarray, optionnel).
                - 'probabilities': Probabilités pour tous les genres (np.ndarray, optionnel).
        
        Raises:
            FileNotFoundError: Si le fichier audio n'existe pas.
            ValueError: Si la conversion audio échoue.
            RuntimeError: Si la prédiction échoue.
        """
        if self.model is None:
            # Mode fallback si modèle non chargé
            logger.warning(
                "Modèle non chargé, utilisation du mode fallback",
                extra={"extra": {"audio_path": audio_path}}
            )
            genre_id = np.random.randint(0, NUM_GENRES)
            confidence = np.random.uniform(
                FALLBACK_CONFIDENCE_MIN, 
                FALLBACK_CONFIDENCE_MAX
            )
            return {
                'genre': self.LABEL_MAPPING[genre_id],
                'confidence': confidence,
                'features': (
                    None if not return_features 
                    else np.random.rand(FEATURE_DIMENSION)
                ),
                'probabilities': (
                    None if not return_probabilities 
                    else np.random.rand(NUM_GENRES)
                )
            }
        
        logger.info(
            "Prédiction de genre audio",
            extra={"extra": {
                "audio_path": audio_path,
                "return_features": return_features,
                "return_probabilities": return_probabilities
            }}
        )
        
        spectrogram = self.convert_song_to_matrix(audio_path)
        
        try:
            spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()
            features: Optional[np.ndarray] = None
            
            if return_features:
                features = self.extract_features(spectrogram_tensor)
            
            with torch.no_grad():
                output = self.model(spectrogram_tensor.to(self.device))
                probabilities = F.softmax(output, dim=1)
                genre_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][genre_id].item()
                all_probs = (
                    probabilities[0].cpu().numpy() 
                    if return_probabilities 
                    else None
                )
            
            predicted_genre = self.LABEL_MAPPING[genre_id]
            
            logger.info(
                "Prédiction terminée avec succès",
                extra={"extra": {
                    "audio_path": audio_path,
                    "predicted_genre": predicted_genre,
                    "confidence": confidence,
                    "genre_id": genre_id
                }}
            )
            
            result: dict[str, Any] = {
                'genre': predicted_genre,
                'confidence': confidence
            }
            
            if return_features:
                result['features'] = features[0] if features is not None else None
            
            if return_probabilities:
                result['probabilities'] = all_probs
            
            return result
            
        except RuntimeError as e:
            logger.exception(
                "Erreur runtime lors de la prédiction",
                extra={"extra": {
                    "audio_path": audio_path,
                    "error_type": "RuntimeError"
                }}
            )
            raise RuntimeError(f"Erreur lors de la prédiction: {e}") from e
        except ValueError as e:
            logger.exception(
                "Erreur de valeur lors de la prédiction",
                extra={"extra": {
                    "audio_path": audio_path,
                    "error_type": "ValueError"
                }}
            )
            raise RuntimeError(f"Erreur lors de la prédiction: {e}") from e
    
    def is_loaded(self) -> bool:
        """
        Vérifie si le modèle est chargé.
        
        Returns:
            True si le modèle est chargé, False sinon.
        """
        return self.model is not None
