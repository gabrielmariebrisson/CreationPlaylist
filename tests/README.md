# Tests Unitaires

Ce dossier contient les tests unitaires pour l'application de génération de playlist.

## Structure

- `test_playlist_generator.py` : Tests pour la classe `PlaylistPathfinder`
- `conftest.py` : Configuration globale pour pytest
- `check_dependencies.py` : Script pour vérifier les dépendances

## Installation

### Vérification des dépendances

Avant d'exécuter les tests, vérifiez que toutes les dépendances sont installées :

```bash
python tests/check_dependencies.py
```

### Avec conda

**IMPORTANT** : Assurez-vous que votre environnement conda est activé avant d'exécuter les tests.

```bash
# Activer votre environnement conda
conda activate Creationplaylist

# Vérifier que vous utilisez le bon Python
which python
# Devrait afficher le chemin vers votre environnement conda

# Installer les dépendances si nécessaire
conda install pytest pytest-cov pandas numpy scikit-learn -c conda-forge
# ou
pip install pytest pytest-cov
```

### Avec pip

```bash
pip install pytest pytest-cov
```

## Exécution des tests

**⚠️ IMPORTANT avec conda** : Assurez-vous que votre environnement conda est activé :

```bash
# Activer l'environnement conda
conda activate Creationplaylist

# Vérifier que pytest utilise le bon Python
which pytest
# ou
python -m pytest tests/
```

Pour exécuter tous les tests :

```bash
pytest tests/
```

Pour exécuter un fichier de test spécifique :

```bash
pytest tests/test_playlist_generator.py
```

Pour exécuter avec couverture de code :

```bash
pytest tests/ --cov=src --cov-report=html
```

Pour exécuter avec affichage verbose :

```bash
pytest tests/ -v
```

## Dépannage

### Erreur "ModuleNotFoundError: No module named 'pandas'"

Cela signifie que pytest utilise le Python système au lieu de votre environnement conda.

**Solution** :
1. Activez votre environnement conda : `conda activate Creationplaylist`
2. Utilisez `python -m pytest` au lieu de `pytest` :
   ```bash
   python -m pytest tests/ -v
   ```
3. Ou installez les dépendances dans l'environnement conda :
   ```bash
   conda install pandas numpy scikit-learn -c conda-forge
   ```

### Vérifier quel Python est utilisé

```bash
which python
which pytest
python --version
```

## Tests implémentés

### PlaylistPathfinder

- ✅ Initialisation
- ✅ Calcul de similarité cosinus
- ✅ Calcul de distance cosinus
- ✅ Réduction de dimensionnalité (PCA)
- ✅ Réduction de dimensionnalité (t-SNE)
- ✅ Génération de playlist de base
- ✅ Génération de playlist avec différentes longueurs
- ✅ Génération sans PCA
- ✅ Gestion des erreurs (indices invalides, colonnes manquantes)
- ✅ Structure de la playlist générée
- ✅ Absence de doublons
- ✅ Analyse de qualité de playlist

## Fixtures

Les fixtures suivantes sont disponibles :

- `pathfinder` : Instance de `PlaylistPathfinder`
- `sample_features` : Array numpy de features aléatoires (20 tracks, 1536 dimensions)
- `sample_tracks_df` : DataFrame pandas avec métadonnées de tracks

## Notes

Les tests utilisent des données mockées (aléatoires mais reproductibles grâce à `np.random.seed(42)`) pour éviter de dépendre de données réelles.
