"""Public-host lockdown.

THEARTIST_PUBLIC_HOST (off by default) marks a server that strangers reach: the DAW
handoff, the warmup route and the MCP mount are refused there. Score reading and
MIDI download remain available.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from fastapi import HTTPException

from app.services.env_flags import env_bool

PUBLIC_HOST_VAR = "THEARTIST_PUBLIC_HOST"


def public_host(env: Mapping[str, str] | None = None) -> bool:
    return env_bool(PUBLIC_HOST_VAR, False, env)


def refuse_on_public_host(reason: str) -> Callable[[], None]:
    """A route dependency that answers 403 with `reason` while THEARTIST_PUBLIC_HOST is on."""
    def refuse() -> None:
        if public_host():
            raise HTTPException(status_code=403, detail=reason)
    return refuse
