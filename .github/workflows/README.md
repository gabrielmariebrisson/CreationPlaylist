# GitHub Actions CI/CD Workflows

Ce dossier contient les workflows GitHub Actions pour l'automatisation des tests, du linting et du déploiement.

## Workflow CI (`ci.yml`)

### Déclencheurs

Le workflow se déclenche automatiquement sur :
- **Push** vers les branches `main` et `develop`
- **Pull Request** vers `main` et `develop`
- **Déclenchement manuel** via l'interface GitHub Actions

### Jobs

#### 1. **Lint** - Linting & Code Quality
- ✅ **Black** : Vérification du formatage du code (bloquant)
- ✅ **Ruff** : Linting rapide (remplace flake8) (bloquant)
- ✅ **MyPy** : Vérification des type hints (bloquant)

**Durée estimée** : ~2-3 minutes

**Note** : Toutes les vérifications de linting sont maintenant bloquantes. Si une vérification échoue, la pull request ne pourra pas être mergée. La configuration MyPy se trouve dans `mypy.ini` à la racine du projet.

#### 2. **Test** - Unit Tests
- ✅ Exécution des tests avec `pytest`
- ✅ Couverture de code avec `pytest-cov`
- ✅ Tests sur Python 3.10, 3.11, 3.12 (matrix strategy)
- ✅ Upload des rapports de couverture vers Codecov
- ✅ Génération de rapports HTML de couverture

**Durée estimée** : ~5-8 minutes

#### 3. **Security** - Security Scan
- ✅ **Safety** : Vérification des vulnérabilités dans les dépendances
- ✅ **Bandit** : Détection de problèmes de sécurité dans le code Python

**Durée estimée** : ~2-3 minutes

#### 4. **Build Check** - Build Verification
- ✅ Vérification de la structure du package
- ✅ Vérification des imports de configuration
- ✅ Validation de la structure modulaire
- ✅ Vérification des imports des modules principaux (SpotifyService, AudioClassifier, PlaylistPathfinder, AsyncSpotifyService)

**Durée estimée** : ~1-2 minutes

#### 5. **Notify** - Status Notification
- ✅ Résumé du statut de tous les jobs
- ✅ S'exécute toujours (`if: always()`)

### Artifacts

Les workflows génèrent des artifacts :
- **Coverage reports** : Rapports HTML de couverture par version Python (7 jours de rétention)
- **Security reports** : Rapports Bandit (30 jours de rétention)

### Configuration

#### Variables d'environnement
- `PYTHON_VERSION`: '3.10' (par défaut)
- `POETRY_VERSION`: '1.6.0' (pour référence future)

#### Timeouts
- Lint: 10 minutes
- Test: 15 minutes
- Security: 10 minutes
- Build: 10 minutes

### Badges

Vous pouvez ajouter des badges à votre README :

```markdown
![CI](https://github.com/votre-username/CreationPlaylist/workflows/CI/badge.svg)
![Tests](https://github.com/votre-username/CreationPlaylist/workflows/CI/badge.svg?branch=main)
```

### Résolution des problèmes

#### Les tests échouent
1. Vérifiez les logs dans l'onglet "Actions" de GitHub
2. Exécutez les tests localement : `pytest tests/ -v`
3. Vérifiez que toutes les dépendances sont installées

#### Le linting échoue
1. **Black** : Exécutez `black src/ tests/` pour formater automatiquement le code
2. **Ruff** : Exécutez `ruff check src/ tests/` pour voir les erreurs, puis `ruff check --fix src/ tests/` pour les corriger automatiquement
3. **MyPy** : Exécutez `mypy src/` pour voir les erreurs de typage. Consultez `mypy.ini` pour la configuration. Les erreurs doivent être corrigées manuellement.

**Important** : Toutes ces vérifications sont maintenant bloquantes. Les pull requests ne pourront pas être mergées si l'une d'elles échoue.

#### Les imports échouent
1. Vérifiez que tous les `__init__.py` sont présents
2. Vérifiez que les chemins d'import sont corrects
3. Testez localement : `python -c "from src.services.spotify_service import SpotifyService"`
