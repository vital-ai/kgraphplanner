"""
Singleton manager for the Weaviate v4 client connection.
Uses connect_to_custom() for self-hosted instances.

Connection parameters come from AgentConfig.weaviate (WeaviateConfig),
populated via KGPLAN__WEAVIATE__* env vars or YAML config.

Authentication:
  - auth_mode == "bearer": caller provides JWT via init_weaviate(jwt=...).
  - auth_mode == "api_key": reads WEAVIATE_API_KEY from environment.
  - auth_mode == "none": no authentication.

This module has NO dependency on any external auth API.  JWT acquisition
(e.g. via Keycloak password grant) is the caller's responsibility.
"""

import os
import logging
import weaviate
from weaviate.classes.init import Auth
from weaviate.config import AdditionalConfig, Timeout

from kgraphplanner.config.agent_config import AgentConfig, WeaviateConfig

logger = logging.getLogger(__name__)

_client: weaviate.WeaviateClient | None = None
_config: WeaviateConfig | None = None
_jwt_token: str | None = None


def _resolve_auth(cfg: WeaviateConfig):
    """Build auth_credentials based on WeaviateConfig.auth_mode."""
    mode = cfg.auth_mode.lower()

    if mode == "bearer":
        if not _jwt_token:
            logger.warning("auth_mode=bearer but no JWT provided via init_weaviate(jwt=...)")
            return None
        return Auth.bearer_token(_jwt_token)

    if mode == "api_key":
        key = os.getenv("WEAVIATE_API_KEY", "")
        if not key:
            logger.warning("auth_mode=api_key but WEAVIATE_API_KEY is empty")
            return None
        return Auth.api_key(key)

    return None  # mode == "none"


def init_weaviate(
    config: AgentConfig | WeaviateConfig | None = None,
    jwt: str | None = None,
) -> None:
    """Configure the client manager before the first connection.

    Args:
        config: WeaviateConfig (or full AgentConfig) for connection params.
                If not provided, config is loaded from env vars on first use.
        jwt:    Optional JWT bearer token for auth_mode=bearer.
                How the JWT is obtained is the caller's responsibility.
    """
    global _config, _jwt_token
    if isinstance(config, AgentConfig):
        _config = config.weaviate
    elif isinstance(config, WeaviateConfig):
        _config = config
    if jwt is not None:
        _jwt_token = jwt


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Return a cached Weaviate client, creating one if needed."""
    global _client
    if _client is not None and _client.is_ready():
        return _client

    cfg = _config or AgentConfig.from_env().weaviate
    grpc_host = cfg.grpc_host or cfg.http_host

    headers = {}
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        headers["X-OpenAI-Api-Key"] = openai_key

    connect_kwargs = dict(
        http_host=cfg.http_host,
        http_port=cfg.http_port,
        http_secure=cfg.http_secure,
        grpc_host=grpc_host,
        grpc_port=cfg.grpc_port,
        grpc_secure=cfg.grpc_secure,
        headers=headers,
        skip_init_checks=cfg.skip_init_checks,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=10, query=30, insert=60)
        ),
    )

    auth = _resolve_auth(cfg)
    if auth is not None:
        connect_kwargs["auth_credentials"] = auth

    _client = weaviate.connect_to_custom(**connect_kwargs)

    scheme = "https" if cfg.http_secure else "http"
    logger.info(
        f"Weaviate client connected: {scheme}://{cfg.http_host}:{cfg.http_port} "
        f"gRPC={grpc_host}:{cfg.grpc_port} (secure={cfg.grpc_secure}) "
        f"auth={cfg.auth_mode} ready={_client.is_ready()}"
    )
    return _client


def close_weaviate_client() -> None:
    """Gracefully close the Weaviate client."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("Weaviate client closed")
