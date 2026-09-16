"""Graceful stop control for long-running identification loops."""

from __future__ import annotations

_STOP_STATE = {"flag": False}


def request_stop(*_args, **_kwargs) -> None:
    if not _STOP_STATE["flag"]:
        print("\n🛑 Stop requested — finishing the current step and compiling results so far...")
    _STOP_STATE["flag"] = True


def stop_requested() -> bool:
    return _STOP_STATE["flag"]


def reset_stop_flag() -> None:
    _STOP_STATE["flag"] = False
