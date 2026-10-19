from __future__ import annotations

import httpx

from backend_core.AgentSysID.api_transport import (
    build_api_http_client,
    should_bypass_blocked_local_proxy,
)


def _clear_proxy_env(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_bypasses_only_the_known_dead_loopback_port_nine_proxy(monkeypatch):
    _clear_proxy_env(monkeypatch)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")

    assert should_bypass_blocked_local_proxy() is True
    client = build_api_http_client()
    try:
        assert client._trust_env is False
    finally:
        client.close()


def test_keeps_a_real_user_proxy(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")

    assert should_bypass_blocked_local_proxy() is False
    client = build_api_http_client()
    try:
        assert client._trust_env is True
    finally:
        client.close()


def test_ignores_malformed_proxy_without_disabling_environment(monkeypatch):
    _clear_proxy_env(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "not a proxy url")

    assert should_bypass_blocked_local_proxy() is False
    client = build_api_http_client(timeout=httpx.Timeout(12.0))
    try:
        assert client._trust_env is True
        assert client.timeout.read == 12.0
    finally:
        client.close()
