"""Système de logging structuré pour la production."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formatter personnalisé pour produire des logs au format JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formate un log record en JSON.
        
        Args:
            record: LogRecord à formater.
        
        Returns:
            Chaîne JSON formatée.
        """
        # Utiliser datetime.now(timezone.utc) au lieu de datetime.utcnow() (déprécié)
        timestamp = datetime.now(timezone.utc).isoformat()
        # Normaliser le format UTC en 'Z' pour compatibilité (ISO 8601)
        if timestamp.endswith('+00:00'):
            timestamp = timestamp.replace('+00:00', 'Z')
        elif not timestamp.endswith('Z'):
            # Si le format n'a pas de timezone, ajouter 'Z' pour UTC
            timestamp = timestamp + 'Z'
        
        log_data: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Ajouter les champs extra si présents
        if hasattr(record, "extra") and record.extra:
            log_data["extra"] = record.extra
        
        # Ajouter exception info si présente
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        # Ajouter stack info si demandé
        if record.stack_info:
            log_data["stack"] = record.stack_info
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class StructuredLogger:
    """
    Logger structuré pour la production avec format JSON.
    
    Utilise le logging standard de Python mais formate les sorties en JSON
    pour faciliter l'intégration avec des systèmes de monitoring (ELK, Datadog, etc.).
    """
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured: bool = False
    
    @classmethod
    def configure(
        cls,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        use_json: bool = True,
        stream: Any = sys.stdout
    ) -> None:
        """
        Configure le système de logging global.
        
        Args:
            level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Chemin vers le fichier de log (optionnel).
            use_json: Si True, utilise le format JSON. Sinon, utilise le format standard.
            stream: Stream pour la sortie console (par défaut: stdout).
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Supprimer les handlers existants pour éviter les doublons
        root_logger.handlers.clear()
        
        # Handler pour console
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(level)
        
        if use_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Handler pour fichier si spécifié
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Obtient ou crée un logger pour un module spécifique.
        
        Args:
            name: Nom du logger (généralement __name__ du module).
        
        Returns:
            Instance de logger configurée.
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def log_api_call(
        cls,
        logger: logging.Logger,
        endpoint: str,
        method: str,
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Log une requête API avec métadonnées structurées.
        
        Args:
            logger: Instance de logger.
            endpoint: Endpoint de l'API.
            method: Méthode HTTP (GET, POST, etc.).
            status_code: Code de statut HTTP (optionnel).
            duration_ms: Durée de la requête en millisecondes (optionnel).
            **kwargs: Métadonnées additionnelles à logger.
        """
        extra = {
            "endpoint": endpoint,
            "method": method,
            **kwargs
        }
        
        if status_code is not None:
            extra["status_code"] = status_code
        
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        
        logger.info(
            f"API call: {method} {endpoint}",
            extra={"extra": extra}
        )
    
    @classmethod
    def log_error_with_context(
        cls,
        logger: logging.Logger,
        message: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log une erreur avec contexte structuré.
        
        Args:
            logger: Instance de logger.
            message: Message d'erreur.
            error: Exception levée.
            context: Contexte additionnel (optionnel).
        """
        extra = {"error_type": type(error).__name__, "error_message": str(error)}
        
        if context:
            extra.update(context)
        
        logger.error(
            message,
            extra={"extra": extra},
            exc_info=True
        )


def get_logger(name: str) -> logging.Logger:
    """
    Fonction helper pour obtenir un logger.
    
    Args:
        name: Nom du logger (généralement __name__).
    
    Returns:
        Instance de logger configurée.
    """
    # Configurer automatiquement si pas encore fait
    if not StructuredLogger._configured:
        StructuredLogger.configure()
    
    return StructuredLogger.get_logger(name)

