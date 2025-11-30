"""Exceptions personnalisées pour les services."""

from typing import Optional


class SpotifyAPIError(Exception):
    """
    Exception de base pour les erreurs de l'API Spotify.
    
    Utilisée pour identifier les erreurs transitoires qui peuvent être retentées.
    """
    pass


class SpotifyRateLimitError(SpotifyAPIError):
    """
    Exception levée lors d'un rate limit de l'API Spotify (429).
    
    Cette erreur est transitoire et peut être retentée après un délai.
    """
    def __init__(self, message: str = "Rate limit atteint", retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class SpotifyTimeoutError(SpotifyAPIError):
    """
    Exception levée lors d'un timeout de l'API Spotify.
    
    Cette erreur est transitoire et peut être retentée.
    """
    pass


class SpotifyServerError(SpotifyAPIError):
    """
    Exception levée lors d'une erreur serveur de l'API Spotify (5xx).
    
    Cette erreur est transitoire et peut être retentée.
    """
    def __init__(self, message: str = "Erreur serveur Spotify", status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

