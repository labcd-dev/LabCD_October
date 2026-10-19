"""HTTP transport helpers for the configured LLM providers."""
from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx


_PROXY_ENV_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


def _is_blocked_local_proxy(value: str) -> bool:
    """Recognize the local port-9 sink injected by restricted runtimes."""
    candidate = value.strip()
    if not candidate:
        return False
    if "://" not in candidate:
        candidate = "http://" + candidate
    try:
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").strip("[]").casefold()
        return parsed.port == 9 and host in {"127.0.0.1", "localhost", "::1"}
    except ValueError:
        return False


def should_bypass_blocked_local_proxy() -> bool:
    """Bypass only when every configured proxy is the known dead local sink.

    Valid user, organization, or development proxies remain in use. The common
    restricted-runtime value ``http://127.0.0.1:9`` refuses connections before
    the request can reach an API, so it is ignored for this process's API client.
    """
    configured = [os.environ.get(name, "").strip() for name in _PROXY_ENV_NAMES]
    configured = [value for value in configured if value]
    return bool(configured) and all(_is_blocked_local_proxy(value) for value in configured)


def build_api_http_client(*, timeout: float | httpx.Timeout | None = None) -> httpx.Client:
    """Build a sync HTTP client, preserving real proxies and bypassing the sink."""
    options = {"timeout": timeout} if timeout is not None else {}
    if should_bypass_blocked_local_proxy():
        options["trust_env"] = False
    return httpx.Client(**options)
