"""Service asynchrone pour gérer les interactions avec l'API Spotify."""

import os
import asyncio
from typing import Optional, Dict, List, Any

import aiohttp
from dotenv import load_dotenv
from tenacity import (
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    AsyncRetrying,
)

from src.config import (
    SPOTIFY_SEARCH_LIMIT_DEFAULT,
    SPOTIFY_SEARCH_LIMIT_MATCH,
)
from src.utils.logger import get_logger
import logging
from src.services.exceptions import (
    SpotifyAPIError,
    SpotifyRateLimitError,
    SpotifyTimeoutError,
    SpotifyServerError,
)

load_dotenv()

logger = get_logger(__name__)

# Configuration du retry pour les appels API Spotify async
ASYNC_RETRY_CONFIG = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=2, min=2, max=8),  # 2s, 4s, 8s
    "retry": retry_if_exception_type(SpotifyAPIError),
    "before_sleep": before_sleep_log(logger, logging.WARNING),
    "reraise": True,
}


def _convert_http_exception(status: int, message: str) -> SpotifyAPIError:
    """
    Convertit un code HTTP en exception custom pour le retry logic.

    Args:
        status: Code de statut HTTP.
        message: Message d'erreur.

    Returns:
        Exception custom appropriée.
    """
    if status == 429:
        return SpotifyRateLimitError(f"Rate limit atteint: {message}")
    elif 500 <= status < 600:
        return SpotifyServerError(
            f"Erreur serveur Spotify ({status}): {message}", status_code=status
        )
    elif "timeout" in message.lower() or "timed out" in message.lower():
        return SpotifyTimeoutError(f"Timeout Spotify: {message}")

    return SpotifyAPIError(f"Erreur API Spotify ({status}): {message}")


class AsyncSpotifyService:
    """Service asynchrone pour gérer l'authentification et les opérations Spotify."""

    BASE_URL = "https://api.spotify.com/v1"

    def __init__(self, access_token: Optional[str] = None) -> None:
        """
        Initialise le service Spotify asynchrone.

        Args:
            access_token: Token d'accès Spotify. Si None, sera récupéré
                depuis les variables d'environnement.
        """
        self.access_token = access_token or os.getenv("SPOTIFY_ACCESS_TOKEN")
        self._session: Optional[aiohttp.ClientSession] = None
        self._refresh_token = os.getenv("REFRESH_TOKEN_SPOTIFY")
        self._client_id = os.getenv("CLIENT_ID_SPOTIFY")
        self._client_secret = os.getenv("CLIENT_SECRET_SPOTIFY")

    async def __aenter__(self):
        """Context manager entry - crée la session HTTP."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ferme la session HTTP."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Assure qu'une session HTTP est créée."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.debug("Session HTTP aiohttp créée")

    async def close(self) -> None:
        """Ferme la session HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Session HTTP aiohttp fermée")

    async def _refresh_access_token(self) -> str:
        """
        Rafraîchit le token d'accès Spotify.

        Returns:
            Nouveau token d'accès.

        Raises:
            SpotifyAPIError: Si le rafraîchissement échoue.
        """
        if not self._refresh_token or not self._client_id or not self._client_secret:
            raise SpotifyAPIError(
                "Credentials Spotify manquants pour le rafraîchissement"
            )

        await self._ensure_session()

        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }
        auth = aiohttp.BasicAuth(self._client_id, self._client_secret)

        # MyPy: _session ne devrait pas être None ici
        assert self._session is not None

        try:
            async with self._session.post(url, data=data, auth=auth) as response:
                if response.status == 200:
                    token_data = await response.json()
                    new_token = token_data.get("access_token")
                    if new_token:
                        self.access_token = new_token
                        logger.info("Token Spotify rafraîchi avec succès")
                        return new_token
                    else:
                        raise SpotifyAPIError(
                            "Token d'accès non présent dans la réponse"
                        )
                else:
                    error_text = await response.text()
                    raise _convert_http_exception(response.status, error_text)
        except aiohttp.ClientError as e:
            raise SpotifyAPIError(
                f"Erreur réseau lors du rafraîchissement: {str(e)}"
            ) from e

    async def _get_access_token(self) -> str:
        """
        Obtient un token d'accès valide (rafraîchit si nécessaire).

        Returns:
            Token d'accès valide.
        """
        if not self.access_token:
            return await self._refresh_access_token()
        return self.access_token

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_on_auth_error: bool = True,
    ) -> Dict[str, Any]:
        """
        Effectue une requête HTTP vers l'API Spotify avec retry logic.

        Args:
            method: Méthode HTTP (GET, POST, etc.).
            endpoint: Endpoint de l'API (sans le base URL).
            params: Paramètres de requête (optionnel).
            json_data: Données JSON pour POST/PUT (optionnel).
            retry_on_auth_error: Si True, réessaie avec un nouveau token
                en cas d'erreur 401.

        Returns:
            Réponse JSON de l'API.

        Raises:
            SpotifyAPIError: Si la requête échoue après tous les retries.
        """
        await self._ensure_session()
        access_token = await self._get_access_token()

        # MyPy: _ensure_session garantit que _session n'est pas None
        assert self._session is not None

        url = f"{self.BASE_URL}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        async def _do_request():
            # MyPy: _session ne devrait pas être None après _ensure_session
            assert self._session is not None
            try:
                async with self._session.request(
                    method, url, headers=headers, params=params, json=json_data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401 and retry_on_auth_error:
                        # Token expiré, rafraîchir et réessayer
                        logger.warning("Token expiré, rafraîchissement en cours")
                        new_token = await self._refresh_access_token()
                        headers["Authorization"] = f"Bearer {new_token}"
                        # Réessayer une fois avec le nouveau token
                        async with self._session.request(
                            method, url, headers=headers, params=params, json=json_data
                        ) as retry_response:
                            if retry_response.status == 200:
                                return await retry_response.json()
                            else:
                                error_text = await retry_response.text()
                                raise _convert_http_exception(
                                    retry_response.status, error_text
                                )
                    else:
                        error_text = await response.text()
                        raise _convert_http_exception(response.status, error_text)
            except aiohttp.ClientError as e:
                raise SpotifyAPIError(f"Erreur réseau: {str(e)}") from e

        # Appliquer le retry avec AsyncRetrying
        retryer = AsyncRetrying(**ASYNC_RETRY_CONFIG)
        async for attempt in retryer:
            with attempt:
                return await _do_request()
        # Cette ligne ne devrait jamais être atteinte grâce à AsyncRetrying
        raise SpotifyAPIError("Tous les retries ont échoué")

    async def search_track(
        self, query: str, limit: int = SPOTIFY_SEARCH_LIMIT_DEFAULT
    ) -> Optional[Dict[str, Any]]:
        """
        Recherche un morceau sur Spotify de manière asynchrone.

        Args:
            query: Requête de recherche (nom du morceau + artiste).
            limit: Nombre maximum de résultats.
                Par défaut: SPOTIFY_SEARCH_LIMIT_DEFAULT.

        Returns:
            Résultat de recherche ou None en cas d'erreur.
        """
        logger.info(
            "Recherche async de track sur Spotify",
            extra={"extra": {"query": query, "limit": limit}},
        )

        try:
            results = await self._make_request(
                "GET", "search", params={"q": query, "type": "track", "limit": limit}
            )

            num_results = len(results.get("tracks", {}).get("items", []))
            logger.info(
                "Recherche Spotify async réussie",
                extra={
                    "extra": {
                        "query": query,
                        "results_count": num_results,
                        "limit": limit,
                    }
                },
            )
            return results
        except SpotifyAPIError as e:
            logger.exception(
                "Erreur API Spotify après retries (async)",
                extra={
                    "extra": {
                        "query": query,
                        "limit": limit,
                        "error_type": type(e).__name__,
                        "attempts": 3,
                    }
                },
            )
            return None

    async def match_deezer_to_spotify(
        self, track_name: str, artist_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Trouve le track Spotify correspondant à un track Deezer de manière asynchrone.

        Args:
            track_name: Nom du morceau.
            artist_name: Nom de l'artiste.

        Returns:
            Dictionnaire avec les infos Spotify ou None si non trouvé.
        """
        logger.debug(
            "Matching async Deezer vers Spotify",
            extra={"extra": {"track_name": track_name, "artist_name": artist_name}},
        )

        try:
            query = f"{track_name} {artist_name}"
            results = await self.search_track(query, SPOTIFY_SEARCH_LIMIT_MATCH)

            if results and results.get("tracks", {}).get("items"):
                best_match = results["tracks"]["items"][0]
                match_info = {
                    "spotify_id": best_match["id"],
                    "uri": best_match["uri"],
                    "name": best_match["name"],
                    "artists": ", ".join([a["name"] for a in best_match["artists"]]),
                }
                logger.info(
                    "Match Deezer->Spotify trouvé (async)",
                    extra={
                        "extra": {
                            "track_name": track_name,
                            "spotify_id": match_info["spotify_id"],
                            "spotify_name": match_info["name"],
                        }
                    },
                )
                return match_info

            logger.debug(
                "Aucun match trouvé pour Deezer track (async)",
                extra={"extra": {"track_name": track_name, "artist_name": artist_name}},
            )
            return None
        except Exception as e:
            logger.exception(
                "Erreur lors du matching async",
                extra={
                    "extra": {
                        "track_name": track_name,
                        "artist_name": artist_name,
                        "error_type": type(e).__name__,
                    }
                },
            )
            return None

    async def get_track_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les features audio d'un track Spotify de manière asynchrone.

        Args:
            track_id: ID Spotify du track.

        Returns:
            Features audio ou None en cas d'erreur.
        """
        logger.debug(
            "Récupération async des features audio",
            extra={"extra": {"track_id": track_id}},
        )

        try:
            features = await self._make_request("GET", f"audio-features/{track_id}")

            logger.debug(
                "Features audio récupérées (async)",
                extra={"extra": {"track_id": track_id}},
            )
            return features
        except SpotifyAPIError as e:
            logger.warning(
                "Erreur lors de la récupération des features (async)",
                extra={"extra": {"track_id": track_id, "error_type": type(e).__name__}},
            )
            return None

    async def get_tracks_features_batch(
        self, track_ids: List[str], batch_size: int = 50
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Récupère les features audio de plusieurs tracks en parallèle (batching).

        Utilise asyncio.gather pour exécuter les requêtes en parallèle au lieu
        d'une boucle séquentielle, améliorant significativement les performances.

        Args:
            track_ids: Liste des IDs Spotify des tracks.
            batch_size: Nombre de tracks à traiter par batch (pour éviter la surcharge).

        Returns:
            Dictionnaire {track_id: features} avec None pour les tracks non trouvés.
        """
        logger.info(
            "Récupération batch async des features audio",
            extra={"extra": {"total_tracks": len(track_ids), "batch_size": batch_size}},
        )

        results: Dict[str, Optional[Dict[str, Any]]] = {}

        # Traiter par batches pour éviter la surcharge
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i : i + batch_size]

            # Créer les coroutines pour ce batch
            tasks = [self.get_track_features(track_id) for track_id in batch]

            # Exécuter en parallèle avec gather
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Mapper les résultats aux track_ids
            for track_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Erreur lors de la récupération des features dans le batch",
                        extra={
                            "extra": {
                                "track_id": track_id,
                                "error_type": type(result).__name__,
                            }
                        },
                    )
                    results[track_id] = None
                elif result is not None:
                    # MyPy: result n'est pas une Exception ici grâce au isinstance check
                    results[track_id] = result  # type: ignore[assignment]

            logger.debug(
                f"Batch {i // batch_size + 1} traité",
                extra={
                    "extra": {
                        "batch_start": i,
                        "batch_end": min(i + batch_size, len(track_ids)),
                        "successful": sum(
                            1 for r in batch_results if not isinstance(r, Exception)
                        ),
                    }
                },
            )

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(
            "Récupération batch async terminée",
            extra={
                "extra": {
                    "total_tracks": len(track_ids),
                    "successful": successful,
                    "failed": len(track_ids) - successful,
                }
            },
        )

        return results

    async def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations de l'utilisateur actuel de manière asynchrone.

        Returns:
            Informations utilisateur ou None en cas d'erreur.
        """
        logger.debug("Récupération async des informations utilisateur Spotify")

        try:
            user_info = await self._make_request("GET", "me")
            logger.info(
                "Informations utilisateur récupérées (async)",
                extra={"extra": {"user_id": user_info.get("id", "unknown")}},
            )
            return user_info
        except SpotifyAPIError as e:
            logger.exception(
                "Erreur API Spotify lors de la récupération des informations "
                "utilisateur après retries (async)",
                extra={"extra": {"error_type": type(e).__name__, "attempts": 3}},
            )
            return None

    async def search_tracks_batch(
        self, queries: List[str], limit: int = SPOTIFY_SEARCH_LIMIT_DEFAULT
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Recherche plusieurs tracks en parallèle.

        Args:
            queries: Liste des requêtes de recherche.
            limit: Nombre maximum de résultats par requête.

        Returns:
            Dictionnaire {query: results} avec None pour les recherches échouées.
        """
        logger.info(
            "Recherche batch async de tracks",
            extra={"extra": {"num_queries": len(queries), "limit": limit}},
        )

        # Créer les coroutines pour toutes les recherches
        tasks = [self.search_track(query, limit) for query in queries]

        # Exécuter en parallèle avec gather
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Mapper les résultats aux queries
        results: Dict[str, Optional[Dict[str, Any]]] = {}
        for query, result in zip(queries, results_list):
            if isinstance(result, Exception):
                logger.warning(
                    "Erreur lors de la recherche dans le batch",
                    extra={
                        "extra": {"query": query, "error_type": type(result).__name__}
                    },
                )
                results[query] = None
            elif result is not None:
                # MyPy: result n'est pas une Exception ici grâce au isinstance check
                results[query] = result  # type: ignore[assignment]

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(
            "Recherche batch async terminée",
            extra={
                "extra": {
                    "total_queries": len(queries),
                    "successful": successful,
                    "failed": len(queries) - successful,
                }
            },
        )

        return results
