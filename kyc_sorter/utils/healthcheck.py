"""
Health check and connectivity verification utility for Ollama API and VLM model.
Used for Docker smoke tests, Kubernetes readiness/liveness probes, and CLI preflight checks.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)


def extract_base_url(endpoint: str) -> str:
    """Extracts the base protocol + host + port from an endpoint URL."""
    parsed = urllib.parse.urlparse(endpoint.strip())
    netloc = parsed.netloc
    scheme = parsed.scheme or "http"
    return f"{scheme}://{netloc}"


def verify_ollama_connectivity(
    endpoint: str,
    model_name: str,
    timeout: int = 10,
) -> Tuple[bool, str]:
    """
    Checks whether the Ollama API is reachable and verifies that the specified
    model is present in the Ollama model cache.

    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    base_url = extract_base_url(endpoint)
    tags_url = f"{base_url}/api/tags"

    try:
        response = requests.get(tags_url, timeout=timeout)
        if response.status_code != 200:
            return (
                False,
                f"Ollama server responded with HTTP status {response.status_code} at {tags_url}.",
            )
        data = response.json()
    except requests.exceptions.ConnectionError as ce:
        return (
            False,
            f"Cannot connect to Ollama at '{base_url}'. Ensure Ollama is running and accessible. "
            f"If running in Docker, ensure endpoint is NOT 'localhost' (use 'host.docker.internal' "
            f"or the host IP). Details: {ce}",
        )
    except requests.exceptions.Timeout:
        return (
            False,
            f"Connection to Ollama at '{base_url}' timed out after {timeout} seconds.",
        )
    except Exception as e:
        return (
            False,
            f"Unexpected error communicating with Ollama at '{tags_url}': {e}",
        )

    # Check model availability in returned tags
    models_list = data.get("models", [])
    available_model_names = [m.get("name", "") for m in models_list]

    # Normalize target model name
    target = model_name.strip().lower()
    target_base = target.split(":")[0]

    matched = None
    matched_info = None
    for m in models_list:
        m_name = m.get("name", "").strip().lower()
        if m_name == target or m_name.startswith(f"{target}:") or f"{target}:latest" == m_name:
            matched = m.get("name")
            matched_info = m
            break
        # Also check model key if present
        m_model = m.get("model", "").strip().lower()
        if m_model == target or m_model.startswith(f"{target}:"):
            matched = m.get("name") or m.get("model")
            matched_info = m
            break

    if matched:
        size_bytes = matched_info.get("size", 0) if matched_info else 0
        size_str = f" ({size_bytes / (1024 ** 3):.2f} GB)" if size_bytes else ""
        return (
            True,
            f"Ollama API is reachable at '{base_url}'. Model '{matched}' is available{size_str}.",
        )

    avail_str = ", ".join(f"'{name}'" for name in available_model_names) if available_model_names else "none"
    return (
        False,
        f"Ollama API is reachable at '{base_url}', but model '{model_name}' was NOT found. "
        f"Available models: [{avail_str}]. Please run 'ollama pull {model_name}' on the host.",
    )
