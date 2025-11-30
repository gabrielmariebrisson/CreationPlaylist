# 🔍 Code Review FAANG - Music Playlist Generator
## Senior Software Architect Analysis

**Reviewer**: Senior Software Architect (Google/Meta standards)  
**Date**: 2025  
**Project**: Music Playlist Generator - Spotify Integration  
**Author**: Gabriel Marie-Brisson, MSc 2025

---

## 📊 Executive Summary

**Overall Grade: B+ (Good foundation, needs refinement for FAANG standards)**

### Strengths ✅
- ✅ Clean modular architecture (SRP respected)
- ✅ Type hints and docstrings (Google format)
- ✅ Unit tests with pytest
- ✅ Separation of concerns (services/models/logic)
- ✅ Configuration centralization

### Critical Gaps ⚠️
- ❌ **No logging system** (critical for production)
- ❌ **No async/await** for API calls (performance bottleneck)
- ❌ **No retry logic** for external APIs
- ❌ **No caching strategy** for expensive operations
- ❌ **No monitoring/metrics**
- ❌ **Security vulnerabilities** (token handling)
- ❌ **No CI/CD pipeline**
- ❌ **Missing error recovery mechanisms**

---

## 1. 🏗️ ARCHITECTURE & DESIGN PATTERNS

### Current State: **B+**

**Strengths:**
- Clean separation: `services/`, `models/`, `logic/`
- Dependency injection in `SpotifyService`
- Configuration centralization in `src/config.py`

**Critical Issues:**

#### 1.1 Missing Repository Pattern
**Problem**: Direct database/API access scattered across codebase.

**Impact**: Hard to test, mock, or swap data sources.

**Recommendation**:
```python
# src/repositories/spotify_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class SpotifyRepository(ABC):
    @abstractmethod
    async def search_track(self, query: str, limit: int) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def create_playlist(self, name: str, tracks: List[str]) -> Dict[str, Any]:
        pass

class SpotifyAPIRepository(SpotifyRepository):
    def __init__(self, client: spotipy.Spotify):
        self.client = client
    
    async def search_track(self, query: str, limit: int) -> List[Dict[str, Any]]:
        # Implementation with retry logic
        pass
```

#### 1.2 Missing Factory Pattern for Model Loading
**Problem**: Model initialization scattered, no versioning.

**Recommendation**:
```python
# src/models/factory.py
class ModelFactory:
    @staticmethod
    def create_classifier(
        model_type: str = "SimpleCNN",
        version: str = "v1.0",
        device: Optional[torch.device] = None
    ) -> AudioClassifier:
        model_registry = {
            "SimpleCNN": {
                "v1.0": "templates/assets/music/best_model_original_loss.pth",
                "v2.0": "templates/assets/music/best_model_v2.pth"
            }
        }
        # Load with versioning
```

#### 1.3 Missing Strategy Pattern for Pathfinding
**Problem**: Only one pathfinding algorithm (linear interpolation).

**Recommendation**:
```python
# src/logic/pathfinding/strategies.py
class PathfindingStrategy(ABC):
    @abstractmethod
    def generate_path(
        self, 
        features: np.ndarray,
        start_idx: int,
        end_idx: int,
        num_tracks: int
    ) -> List[int]:
        pass

class LinearInterpolationStrategy(PathfindingStrategy):
    # Current implementation
    pass

class AStarStrategy(PathfindingStrategy):
    # Advanced pathfinding
    pass

class GeneticAlgorithmStrategy(PathfindingStrategy):
    # Evolutionary approach
    pass
```

**Priority**: Medium (shows design thinking)

---

## 2. 💻 CODE QUALITY & BEST PRACTICES

### Current State: **B**

**Strengths:**
- Type hints (Python 3.10+)
- Google-style docstrings
- Specific exception handling

**Critical Issues:**

#### 2.1 Silent Exception Swallowing
**Location**: `src/services/spotify_service.py:107-108`

```python
except Exception as e:
    return None  # ❌ Silent failure - no logging!
```

**Fix**:
```python
import logging

logger = logging.getLogger(__name__)

except SpotifyException as e:
    logger.error(
        "Failed to refresh Spotify token",
        extra={"error": str(e), "refresh_token_length": len(refresh_token)}
    )
    return None
except Exception as e:
    logger.exception("Unexpected error refreshing token")
    return None
```

#### 2.2 Magic Numbers & Hardcoded Values
**Location**: Multiple files

**Issues**:
- `1e-9` scattered (should be `EPSILON` from config)
- Batch size `100` hardcoded in `export_playlist`
- Timeout values missing

**Fix**: Move all to `src/config.py`:
```python
# src/config.py
SPOTIFY_API_TIMEOUT_SECONDS: int = 30
SPOTIFY_MAX_RETRIES: int = 3
SPOTIFY_RETRY_BACKOFF_SECONDS: float = 1.0
FEATURE_NORMALIZATION_EPSILON: float = 1e-9
```

#### 2.3 Missing Input Validation
**Location**: `src/logic/playlist_generator.py:182`

**Issue**: No validation of `num_tracks` range, `raw_features` shape.

**Fix**:
```python
from pydantic import BaseModel, Field, validator

class PlaylistGenerationRequest(BaseModel):
    tracks_df: pd.DataFrame
    raw_features: np.ndarray
    track1_idx: int = Field(..., ge=0)
    track2_idx: int = Field(..., ge=0)
    num_tracks: int = Field(..., ge=2, le=100)
    
    @validator('raw_features')
    def validate_features_shape(cls, v, values):
        if 'tracks_df' in values:
            if len(v) != len(values['tracks_df']):
                raise ValueError("Features length must match tracks_df")
        if v.shape[1] != FEATURE_VIEW_SIZE:
            raise ValueError(f"Features must have {FEATURE_VIEW_SIZE} dimensions")
        return v
```

**Priority**: High (data integrity)

---

## 3. ⚡ PERFORMANCE & SCALABILITY

### Current State: **C+**

**Critical Issues:**

#### 3.1 Synchronous API Calls (Blocking I/O)
**Problem**: All Spotify API calls are synchronous, blocking the event loop.

**Impact**: 
- Poor user experience (UI freezes)
- Cannot handle concurrent requests
- Wastes resources

**Fix**: Implement async/await
```python
# src/services/spotify_service_async.py
import asyncio
from aiohttp import ClientSession
import aiohttp

class AsyncSpotifyService:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://api.spotify.com/v1"
        self._session: Optional[ClientSession] = None
    
    async def __aenter__(self):
        self._session = ClientSession()
        return self
    
    async def __aexit__(self, *args):
        await self._session.close()
    
    async def search_track(
        self, 
        query: str, 
        limit: int = 5
    ) -> Optional[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"q": query, "type": "track", "limit": limit}
        
        async with self._session.get(
            f"{self.base_url}/search",
            headers=headers,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                return await response.json()
            return None
```

#### 3.2 No Caching Strategy
**Problem**: Repeated API calls for same tracks, no feature caching.

**Fix**: Implement Redis/memory cache
```python
# src/services/cache_service.py
from functools import lru_cache
from typing import Optional
import hashlib
import json

class CacheService:
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.memory_cache = {}
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        key_data = json.dumps(kwargs, sort_keys=True)
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get_track_features(
        self, 
        track_id: str
    ) -> Optional[np.ndarray]:
        cache_key = self._generate_key("track_features", track_id=track_id)
        
        # Try Redis first
        if self.redis:
            cached = await self.redis.get(cache_key)
            if cached:
                return np.frombuffer(cached, dtype=np.float32)
        
        # Try memory cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        return None
    
    async def set_track_features(
        self, 
        track_id: str, 
        features: np.ndarray,
        ttl: int = 86400  # 24 hours
    ):
        cache_key = self._generate_key("track_features", track_id=track_id)
        features_bytes = features.tobytes()
        
        if self.redis:
            await self.redis.setex(cache_key, ttl, features_bytes)
        
        self.memory_cache[cache_key] = features
```

#### 3.3 Inefficient Feature Extraction
**Problem**: Sequential processing, no batching.

**Fix**: Batch processing
```python
# src/models/audio_classifier.py
def predict_batch(
    self,
    audio_paths: List[str],
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """Process multiple audio files in batches."""
    results = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        spectrograms = [
            self.convert_song_to_matrix(path) 
            for path in batch_paths
        ]
        
        # Batch inference
        batch_tensor = torch.stack([
            torch.tensor(spec).unsqueeze(0).unsqueeze(0).float()
            for spec in spectrograms
        ])
        
        with torch.no_grad():
            outputs = self.model(batch_tensor.to(self.device))
            # Process batch results
            # ...
    
    return results
```

**Priority**: High (user experience)

---

## 4. 🧪 TESTING & QUALITY ASSURANCE

### Current State: **B-**

**Strengths:**
- Unit tests with pytest
- Good test coverage for `PlaylistPathfinder`
- Mock fixtures

**Critical Gaps:**

#### 4.1 Missing Integration Tests
**Problem**: No tests for API integration, end-to-end workflows.

**Fix**:
```python
# tests/integration/test_spotify_integration.py
import pytest
from unittest.mock import AsyncMock, patch
from src.services.spotify_service import SpotifyService

@pytest.mark.integration
@pytest.mark.asyncio
async def test_spotify_playlist_creation_flow():
    """Test complete playlist creation workflow."""
    service = SpotifyService()
    
    # Mock API responses
    with patch('spotipy.Spotify') as mock_spotify:
        mock_client = mock_spotify.return_value
        mock_client.current_user.return_value = {"id": "test_user"}
        mock_client.user_playlist_create.return_value = {
            "id": "playlist_123",
            "name": "Test Playlist"
        }
        
        playlist = service.export_playlist(
            playlist_tracks=[{"uri": "spotify:track:123"}],
            playlist_name="Test",
            callback_success=lambda x: None
        )
        
        assert playlist is not None
        assert playlist["id"] == "playlist_123"
```

#### 4.2 Missing Performance Tests
**Fix**:
```python
# tests/performance/test_playlist_generation_performance.py
import pytest
import time
from src.logic.playlist_generator import PlaylistPathfinder

@pytest.mark.performance
def test_playlist_generation_performance(benchmark):
    """Ensure playlist generation completes in < 1 second."""
    pathfinder = PlaylistPathfinder()
    # Setup...
    
    result = benchmark(
        pathfinder.generate_playlist_line,
        tracks_df=sample_df,
        raw_features=sample_features,
        track1_idx=0,
        track2_idx=10,
        num_tracks=10
    )
    
    assert result[0] is not None
```

#### 4.3 Missing Property-Based Tests
**Fix**: Use Hypothesis
```python
# tests/property/test_playlist_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    num_tracks=st.integers(min_value=2, max_value=50),
    feature_dim=st.just(1536)
)
def test_playlist_length_property(num_tracks, feature_dim):
    """Property: Generated playlist always has correct length."""
    # Test implementation
    pass
```

**Priority**: Medium (shows testing maturity)

---

## 5. 🔒 SECURITY & API INTEGRATION

### Current State: **C** (Critical Issues)

#### 5.1 Token Storage in Session State
**Problem**: `src/services/spotify_service.py:86-90` stores tokens in Streamlit session.

**Risk**: 
- Tokens in memory (XSS risk)
- No encryption
- Tokens logged in error traces

**Fix**:
```python
# src/services/secure_token_manager.py
from cryptography.fernet import Fernet
import os
import base64

class SecureTokenManager:
    def __init__(self):
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            # Store in secure vault (AWS Secrets Manager, etc.)
        self.cipher = Fernet(key)
    
    def encrypt_token(self, token: str) -> str:
        return self.cipher.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        return self.cipher.decrypt(encrypted_token.encode()).decode()
```

#### 5.2 No Rate Limiting
**Problem**: No protection against API abuse.

**Fix**:
```python
# src/services/rate_limiter.py
from collections import defaultdict
import time
from typing import Dict

class RateLimiter:
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, key: str) -> bool:
        now = time.time()
        # Remove old calls
        self.calls[key] = [
            call_time for call_time in self.calls[key]
            if now - call_time < self.time_window
        ]
        
        if len(self.calls[key]) >= self.max_calls:
            return False
        
        self.calls[key].append(now)
        return True
```

#### 5.3 No Input Sanitization
**Problem**: User inputs not sanitized before API calls.

**Fix**:
```python
# src/utils/input_sanitizer.py
import re
from typing import Optional

def sanitize_search_query(query: str, max_length: int = 100) -> Optional[str]:
    """Sanitize user input for Spotify search."""
    if not query or len(query) > max_length:
        return None
    
    # Remove special characters that could break API
    sanitized = re.sub(r'[<>"\']', '', query)
    sanitized = sanitized.strip()
    
    if len(sanitized) < 1:
        return None
    
    return sanitized
```

**Priority**: **CRITICAL** (security vulnerability)

---

## 6. 🤖 ML/DATA ENGINEERING PRACTICES

### Current State: **B+**

**Strengths:**
- Feature extraction from CNN
- Cosine similarity (appropriate metric)
- PCA for visualization

**Improvements Needed:**

#### 6.1 Missing Model Versioning
**Problem**: No way to track model versions, A/B testing.

**Fix**:
```python
# src/models/model_registry.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ModelMetadata:
    version: str
    path: str
    created_at: datetime
    accuracy: float
    dataset: str
    architecture: str

class ModelRegistry:
    def __init__(self):
        self.models = {
            "SimpleCNN": {
                "v1.0": ModelMetadata(
                    version="v1.0",
                    path="templates/assets/music/best_model_original_loss.pth",
                    created_at=datetime(2024, 1, 1),
                    accuracy=0.73,
                    dataset="GTZAN",
                    architecture="SimpleCNN"
                )
            }
        }
    
    def get_model(self, name: str, version: str) -> Optional[ModelMetadata]:
        return self.models.get(name, {}).get(version)
```

#### 6.2 Missing Feature Store
**Problem**: Features recomputed every time.

**Fix**: Implement feature store (Redis/PostgreSQL)
```python
# src/data/feature_store.py
class FeatureStore:
    async def get_or_compute_features(
        self,
        track_id: str,
        audio_path: str,
        classifier: AudioClassifier
    ) -> np.ndarray:
        # Check cache
        cached = await self.get_features(track_id)
        if cached is not None:
            return cached
        
        # Compute
        result = classifier.predict(audio_path, return_features=True)
        features = result['features']
        
        # Store
        await self.store_features(track_id, features)
        return features
```

#### 6.3 No Model Monitoring
**Problem**: No tracking of model performance in production.

**Fix**: Integrate MLflow/Weights & Biases
```python
# src/monitoring/model_monitor.py
import mlflow

class ModelMonitor:
    def log_prediction(
        self,
        track_id: str,
        predicted_genre: str,
        confidence: float,
        actual_genre: Optional[str] = None
    ):
        mlflow.log_metric("prediction_confidence", confidence)
        if actual_genre:
            mlflow.log_metric(
                "prediction_accuracy",
                1.0 if predicted_genre == actual_genre else 0.0
            )
```

**Priority**: Medium (shows ML maturity)

---

## 7. 📝 LOGGING & OBSERVABILITY

### Current State: **D** (Critical Gap)

**Problem**: No structured logging system.

**Impact**: 
- Cannot debug production issues
- No performance monitoring
- No error tracking

**Fix**: Implement structured logging
```python
# src/utils/logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter for production
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"extra": %(extra)s}'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ):
        self.logger.info(
            f"API call: {method} {endpoint}",
            extra={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration_ms": duration_ms,
                **kwargs
            }
        )
```

**Priority**: **CRITICAL**

---

## 8. 🔄 ERROR HANDLING & RESILIENCE

### Current State: **C+**

**Issues:**

#### 8.1 No Retry Logic
**Fix**: Implement exponential backoff
```python
# src/utils/retry.py
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(SpotifyException)
)
async def search_track_with_retry(
    service: SpotifyService,
    query: str
) -> Optional[Dict[str, Any]]:
    return await service.search_track(query)
```

#### 8.2 No Circuit Breaker
**Fix**: Implement circuit breaker pattern
```python
# src/utils/circuit_breaker.py
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
```

**Priority**: High (resilience)

---

## 9. 📦 DEPENDENCY MANAGEMENT

### Current State: **B**

**Issues:**
- No version pinning strategy
- Missing dev dependencies separation
- No dependency vulnerability scanning

**Fix**:
```python
# requirements.txt - Production
torch==2.9.0
spotipy==2.25.1
# ... pinned versions

# requirements-dev.txt - Development
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
mypy>=1.0.0
ruff>=0.1.0

# requirements-test.txt - Testing
pytest-asyncio>=0.21.0
hypothesis>=6.0.0
```

**Add**: `pyproject.toml` for modern Python packaging
```toml
[project]
name = "music-playlist-generator"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.9.0",
    "spotipy>=2.25.0",
    # ...
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]
```

**Priority**: Medium

---

## 10. 🚀 CI/CD & DEPLOYMENT

### Current State: **F** (Missing)

**Critical Gap**: No CI/CD pipeline.

**Fix**: Create GitHub Actions workflow
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run linters
        run: |
          pip install black mypy ruff
          black --check src/
          mypy src/
          ruff check src/
  
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security scan
        run: |
          pip install safety
          safety check --json
```

**Priority**: **HIGH** (shows DevOps maturity)

---

## 📋 PRIORITY ACTION ITEMS

### 🔴 Critical (Do First)
1. **Implement structured logging** (Section 7)
2. **Fix security vulnerabilities** (Section 5)
3. **Add retry logic for API calls** (Section 8.1)
4. **Create CI/CD pipeline** (Section 10)

### 🟡 High Priority (Next Sprint)
5. **Implement async/await for API calls** (Section 3.1)
6. **Add caching strategy** (Section 3.2)
7. **Add input validation with Pydantic** (Section 2.3)
8. **Implement circuit breaker** (Section 8.2)

### 🟢 Medium Priority (Future)
9. **Add integration tests** (Section 4.1)
10. **Implement feature store** (Section 6.2)
11. **Add model versioning** (Section 6.1)
12. **Implement repository pattern** (Section 1.1)

---

## 🎯 RECOMMENDATIONS FOR FAANG PORTFOLIO

### What Makes This Stand Out:
1. ✅ **Modular architecture** - Shows design thinking
2. ✅ **ML integration** - CNN for audio classification
3. ✅ **API integration** - Real-world Spotify integration
4. ✅ **Type safety** - Type hints throughout

### What Needs Improvement:
1. ❌ **Production readiness** - Missing logging, monitoring
2. ❌ **Scalability** - Synchronous code, no caching
3. ❌ **Security** - Token handling, input validation
4. ❌ **DevOps** - No CI/CD, no deployment strategy

### Portfolio Presentation Tips:
1. **Add a "System Design" section** to README explaining:
   - Architecture decisions
   - Scalability considerations
   - Trade-offs made

2. **Create a "Performance Benchmarks" section**:
   - Latency metrics
   - Throughput numbers
   - Resource usage

3. **Document "Lessons Learned"**:
   - What worked well
   - What you'd do differently
   - Technical debt acknowledged

4. **Add "Future Improvements" roadmap**:
   - Shows forward thinking
   - Demonstrates understanding of production systems

---

## 📊 FINAL SCORE BREAKDOWN

| Category | Score | Weight | Weighted |
|----------|-------|--------|----------|
| Architecture | B+ | 20% | 17% |
| Code Quality | B | 15% | 13% |
| Performance | C+ | 15% | 10% |
| Testing | B- | 15% | 12% |
| Security | C | 20% | 12% |
| ML Practices | B+ | 10% | 9% |
| Observability | D | 5% | 2% |
| **TOTAL** | **B-** | **100%** | **75%** |

**Overall Assessment**: Solid foundation with good architectural decisions, but needs significant work on production-readiness, security, and observability to meet FAANG standards.

---

## 🚀 QUICK WINS (Can implement in 1-2 days)

1. Add structured logging (2 hours)
2. Implement retry logic with tenacity (1 hour)
3. Add input validation with Pydantic (2 hours)
4. Create basic CI/CD pipeline (3 hours)
5. Add security fixes for token handling (2 hours)

**Total**: ~10 hours of focused work for significant improvement.

---

*This review follows Google/Meta code review standards and focuses on production-readiness, maintainability, and scalability.*

