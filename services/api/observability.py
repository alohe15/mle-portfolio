"""One privacy-safe completion log per request.

The middleware is pure ASGI. Starlette's ServerErrorMiddleware sits outside
user middleware, so unexpected exceptions are caught here when the response
has not started. Request IDs are generated in this process and never taken
from the client.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.api.metrics import observe_request

ACCESS_LOGGER_NAME = "services.api.access"
_CONFIGURED = False

access_logger = logging.getLogger(ACCESS_LOGGER_NAME)


def disable_uvicorn_access_log() -> None:
    """Uvicorn configures its access logger after import. Keep it silent."""
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
    uvicorn_access.propagate = False


def configure_access_logger() -> None:
    """Stdout handler that writes the JSON message and nothing else."""
    global _CONFIGURED
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        access_logger.addHandler(handler)
        _CONFIGURED = True
    disable_uvicorn_access_log()


def _model_version() -> str:
    import services.api.app as api_app

    artifacts = api_app.ARTIFACTS
    if not isinstance(artifacts, dict):
        return "none"
    label = artifacts.get("model_version_label")
    if not isinstance(label, str) or not label:
        return "none"
    return label


def _route_template(scope: Scope) -> str:
    """Matched route template, never the raw URL."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path.startswith("/"):
        return path
    endpoint = scope.get("endpoint")
    app = scope.get("app")
    routes = getattr(app, "routes", ()) if app is not None else ()
    if endpoint is not None:
        for candidate in routes:
            if getattr(candidate, "endpoint", None) is endpoint:
                candidate_path = getattr(candidate, "path", None)
                if isinstance(candidate_path, str) and candidate_path.startswith("/"):
                    return candidate_path
    return "unmatched"


def _error_category(status_code: int, unhandled: bool) -> str:
    if unhandled:
        return "unhandled_exception"
    if status_code < 400:
        return "none"
    if status_code == 422:
        return "validation_error"
    if status_code == 404:
        return "not_found"
    if status_code == 405:
        return "method_not_allowed"
    if status_code == 503:
        return "model_not_loaded"
    if status_code == 500:
        return "prediction_error"
    if status_code >= 500:
        return "unhandled_exception"
    return "none"


def _add_request_id(message: Message, request_id: str) -> None:
    headers = list(message.get("headers") or [])
    headers.append((b"X-Request-ID", request_id.encode("ascii")))
    message["headers"] = headers


class AccessLogMiddleware:
    """Generate a request id, log one completion line, and return that id."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        configure_access_logger()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        disable_uvicorn_access_log()
        request_id = str(uuid.uuid4())
        state = scope.get("state")
        if not isinstance(state, dict):
            state = {}
            scope["state"] = state
        state["request_id"] = request_id
        state.pop("recommended_action", None)

        status_code: int | None = None
        response_started = False
        unhandled = False
        started = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                _add_request_id(message, request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            unhandled = True
            if response_started:
                raise
            status_code = 500
            await _send_internal_error(send, request_id)
            response_started = True
        finally:
            if status_code is None:
                status_code = 500
                unhandled = True
            _log_completion(
                scope,
                request_id=request_id,
                status_code=status_code,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                unhandled=unhandled,
            )


async def _send_internal_error(send: Send, request_id: str) -> None:
    body = json.dumps(
        {
            "detail": {
                "message": "Internal server error",
                "request_id": request_id,
            }
        }
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"X-Request-ID", request_id.encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _log_completion(
    scope: Scope,
    *,
    request_id: str,
    status_code: int,
    duration_ms: float,
    unhandled: bool,
) -> None:
    route = _route_template(scope)
    method = scope.get("method")
    status_class = f"{status_code // 100}xx"
    recommended: str | None = None
    state = scope.get("state")
    if route == "/predict" and 200 <= status_code < 300 and isinstance(state, dict):
        action = state.get("recommended_action")
        if isinstance(action, str):
            recommended = action
    # Same route, status class, and duration as the log line. Histogram is seconds.
    observe_request(
        route=route,
        method=method if isinstance(method, str) else "",
        status_class=status_class,
        duration_seconds=duration_ms / 1000.0,
    )
    payload = {
        "event": "request_completed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "route": route,
        "method": method,
        "status_class": status_class,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
        "model_version": _model_version(),
        "error_category": _error_category(status_code, unhandled),
        "recommended_action": recommended,
    }
    access_logger.info(json.dumps(payload, ensure_ascii=True))


configure_access_logger()
