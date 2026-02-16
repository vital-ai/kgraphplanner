"""
Keycloak JWT acquisition for Weaviate bearer-token auth.

Uses env vars:
  WEAVIATE_KEYCLOAK_URL            — token endpoint URL
  WEAVIATE_KEYCLOAK_CLIENT_ID      — client ID (e.g. "weaviate-client")
  WEAVIATE_KEYCLOAK_CLIENT_SECRET  — client secret (optional)
  WEAVIATE_KEYCLOAK_USERNAME       — Keycloak username
  WEAVIATE_KEYCLOAK_PASSWORD       — Keycloak password
"""

import os
import logging
import requests

logger = logging.getLogger(__name__)


def get_weaviate_jwt() -> tuple[str | None, str | None]:
    """Acquire a JWT from Keycloak via password grant.

    Returns:
        (token, None) on success, or (None, error_message) on failure.
    """
    token_url = os.getenv("WEAVIATE_KEYCLOAK_URL", "")
    client_id = os.getenv("WEAVIATE_KEYCLOAK_CLIENT_ID", "")
    client_secret = os.getenv("WEAVIATE_KEYCLOAK_CLIENT_SECRET", "")
    username = os.getenv("WEAVIATE_KEYCLOAK_USERNAME", "")
    password = os.getenv("WEAVIATE_KEYCLOAK_PASSWORD", "")

    if not token_url:
        return None, "WEAVIATE_KEYCLOAK_URL not set"
    if not client_id:
        return None, "WEAVIATE_KEYCLOAK_CLIENT_ID not set"
    if not username or not password:
        return None, "WEAVIATE_KEYCLOAK_USERNAME/PASSWORD not set"

    try:
        data = {
            "grant_type": "password",
            "client_id": client_id,
            "username": username,
            "password": password,
            "scope": "openid profile email",
        }
        if client_secret:
            data["client_secret"] = client_secret

        resp = requests.post(token_url, data=data, timeout=10)
        resp.raise_for_status()
        token = resp.json().get("access_token")
        if not token:
            return None, "No access_token in Keycloak response"
        logger.info("Acquired Keycloak JWT for Weaviate")
        return token, None
    except Exception as e:
        return None, f"Keycloak token request failed: {e}"
