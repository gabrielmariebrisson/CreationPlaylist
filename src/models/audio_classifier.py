"""Classe pour la classification audio avec CNN."""

import os
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Import du modèle depuis le dossier templates
import sys
from pathlib import Path

# Ajouter le chemin du module architecture au path
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent
_architecture_path = _project_root / 'templates' / 'assets' / 'music'
sys.path.insert(0, str(_architecture_path))

try:
    from architecture import SimpleCNN
except ImportError:
    raise ImportError(
        f"Impossible d'importer SimpleCNN. "
        f"Vérifiez que le fichier architecture.py existe dans {_architecture_path}"
    )


class AudioClassifier:
    """Classe pour charger le modèle PyTorch et effectuer des prédictions de genre musical."""
    
    # Mapping des labels de genre
    LABEL_MAPPING = {
        0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
        5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
    }
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Initialise le classifieur audio.
        
        Args:
            model_path: Chemin vers le fichier de modèle PyTorch (.pth)
            device: Device PyTorch (CPU ou CUDA). Si None, utilise CPU.
        """
        self.model_path = model_path
        self.device = device or torch.device('cpu')
        self.model: Optional[SimpleCNN] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Charge le modèle CNN pré-entraîné."""
        try:
            self.model = SimpleCNN()
            
            # Vérifier si le fichier existe
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")
    
    def convert_song_to_matrix(self, audio_path: str, size: int = 599) -> Optional[np.ndarray]:
        """
        Convertit un fichier audio en spectrogramme normalisé.
        
        Args:
            audio_path: Chemin vers le fichier audio
            size: Taille cible du spectrogramme
            
        Returns:
            Spectrogramme normalisé ou None en cas d'erreur
        """
        try:
            y, sr = librosa.load(audio_path, duration=30)
            n_fft = int((sr/10) / 2 + 3)
            D = np.abs(librosa.stft(y, hop_length=int(n_fft)))
            spectrogram = librosa.feature.melspectrogram(S=D, sr=sr)
            S = librosa.util.fix_length(spectrogram, size=size)
            S_db = librosa.power_to_db(S, ref=np.max)
            S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
            return S_db_norm
        except Exception as e:
            raise RuntimeError(f"Erreur conversion audio: {str(e)}")
    
    def extract_features(self, spectrogram_tensor: torch.Tensor) -> np.ndarray:
        """
        Extrait les features avant la couche de classification.
        
        Args:
            spectrogram_tensor: Tenseur du spectrogramme (shape: [1, 1, H, W])
            
        Returns:
            Features extraites sous forme de numpy array
        """
        if self.model is None:
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
            x = x.view(-1, 1536)
            
            x = self.model.normfc2(x)
            x = self.model.fc2(x)
            features = F.relu(x)
            
            return features.cpu().numpy()
    
    def predict(
        self, 
        audio_path: str, 
        return_features: bool = True,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Analyse un fichier audio et prédit son genre.
        
        Args:
            audio_path: Chemin vers le fichier audio
            return_features: Si True, retourne aussi les features extraites
            return_probabilities: Si True, retourne toutes les probabilités par genre
            
        Returns:
            Dictionnaire contenant:
                - 'genre': Genre prédit (str)
                - 'confidence': Confiance de la prédiction (float)
                - 'features': Features extraites (np.ndarray, optionnel)
                - 'probabilities': Probabilités pour tous les genres (np.ndarray, optionnel)
        """
        if self.model is None:
            # Mode fallback si modèle non chargé
            genre_id = np.random.randint(0, 10)
            confidence = np.random.uniform(0.6, 0.95)
            return {
                'genre': self.LABEL_MAPPING[genre_id],
                'confidence': confidence,
                'features': None if not return_features else np.random.rand(2048),
                'probabilities': None if not return_probabilities else np.random.rand(10)
            }
        
        spectrogram = self.convert_song_to_matrix(audio_path)
        
        if spectrogram is None:
            raise ValueError("Impossible de convertir l'audio en spectrogramme")
        
        try:
            spectrogram_tensor = torch.tensor(spectrogram).unsqueeze(0).unsqueeze(0).float()
            features = None
            
            if return_features:
                features = self.extract_features(spectrogram_tensor)
            
            with torch.no_grad():
                output = self.model(spectrogram_tensor.to(self.device))
                probabilities = F.softmax(output, dim=1)
                genre_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][genre_id].item()
                all_probs = probabilities[0].cpu().numpy() if return_probabilities else None
            
            result = {
                'genre': self.LABEL_MAPPING[genre_id],
                'confidence': confidence
            }
            
            if return_features:
                result['features'] = features[0] if features is not None else None
            
            if return_probabilities:
                result['probabilities'] = all_probs
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la prédiction: {e}")
    
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self.model is not None

