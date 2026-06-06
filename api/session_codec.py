"""Serialize simulator sessions for stateless serverless handlers."""

from __future__ import annotations

import base64
import pickle
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .service import RuntimeSession


def encode_runtime(runtime: RuntimeSession) -> str:
    """Return a URL-safe base64 pickle of the active runtime session."""
    payload = pickle.dumps(runtime, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.urlsafe_b64encode(payload).decode("ascii")


def decode_runtime(token: str) -> RuntimeSession:
    """Restore a runtime session from a session token."""
    from .service import RuntimeSession

    if not token:
        raise ValueError("Missing session token.")
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        runtime = pickle.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive decode path
        raise ValueError("Invalid or expired session token.") from exc
    if not isinstance(runtime, RuntimeSession):
        raise ValueError("Session token does not reference a trading session.")
    return runtime
