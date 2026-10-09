"""
Classify and describe exceptions without exposing their content.

Exception messages from LLM providers, tool services and HTTP clients are free
text: they can echo parts of the request (a query, an address, a prompt
fragment) or the request URL. ``describe_error`` reports only non-content
facts unless content logging is switched on:

- the exception class,
- the HTTP status code,
- the failure category (the ``RetrySpec.retry_on`` vocabulary),
- the provider's error code and type, and the request id, when they are
  identifier-shaped.

So ``"BadRequestError (HTTP 400, client_error, code=context_length_exceeded, request_id=req_123)"``
rather than the provider's message.

``failure_kinds`` (used by retry classification) lives here so workers and the
exec graph runtime can share it without importing each other.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Iterator, List, Optional

from kgraphplanner.worker.worker_errors import WorkerError, WorkerTimeoutError

# Provider codes and request ids are enumerations or opaque ids. Anything that
# doesn't look like one (spaces, prose) is dropped rather than echoed.
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:\-]{1,80}")


def error_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield exc and the exceptions it wraps (WorkerError.cause, __cause__)."""
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        nxt = getattr(cur, "cause", None)
        cur = nxt if isinstance(nxt, BaseException) else cur.__cause__


def _class_names(exc: BaseException) -> set:
    return {c.__name__ for c in type(exc).__mro__}


def status_code(exc: BaseException) -> Optional[int]:
    """The HTTP status of an SDK or httpx error, if it has one."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def failure_kinds(exc: BaseException) -> set:
    """Classify an exception into the RetrySpec.retry_on vocabulary.

    Duck-typed on class names and status codes so it covers the OpenAI and
    Anthropic SDKs and httpx without importing them.
    """
    kinds = set()
    for e in error_chain(exc):
        names = _class_names(e)
        if isinstance(e, WorkerError):
            kinds.add(e.error_type)
        if isinstance(e, (WorkerTimeoutError, asyncio.TimeoutError)) or "APITimeoutError" in names:
            kinds.add("timeout")
        code = status_code(e)
        if code == 429 or "RateLimitError" in names:
            kinds.add("rate_limit")
        elif code is not None and code >= 500:
            kinds.add("server_error")
        elif code in (401, 403) or names & {"AuthenticationError", "PermissionDeniedError"}:
            kinds.add("auth_error")
        elif code is not None and 400 <= code < 500:
            kinds.add("client_error")
        if names & {"APIConnectionError", "ConnectError", "ConnectTimeout",
                    "ReadError", "RemoteProtocolError", "TransportError"} \
                or isinstance(e, ConnectionError):
            kinds.add("connection")
    return kinds


def _identifier(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _provider_details(exc: BaseException) -> List[str]:
    """``code=``, ``type=`` and ``request_id=`` from the first exception in the
    chain that carries them (OpenAI: ``.code``/``.type``; Anthropic: the error
    body's ``type``; both: ``.request_id``)."""
    code = err_type = request_id = None
    for e in error_chain(exc):
        code = code or _identifier(getattr(e, "code", None))
        err_type = err_type or _identifier(getattr(e, "type", None))
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            inner = body.get("error") if isinstance(body.get("error"), dict) else body
            code = code or _identifier(inner.get("code"))
            err_type = err_type or _identifier(inner.get("type"))
        request_id = request_id or _identifier(getattr(e, "request_id", None))
    details = []
    if code:
        details.append(f"code={code}")
    if err_type and err_type != code:
        details.append(f"type={err_type}")
    if request_id:
        details.append(f"request_id={request_id}")
    return details


def describe_error(exc: BaseException, *, log_content: bool = False) -> str:
    """A description of ``exc`` that is safe to log and store.

    With ``log_content=True`` this is the full ``"<Class>: <message>"``.
    Otherwise it has no message text, e.g. ``"RateLimitError (HTTP 429, rate_limit)"``.
    """
    if log_content:
        return f"{type(exc).__name__}: {exc}"
    parts: List[str] = []
    code = status_code(exc)
    if code is not None:
        parts.append(f"HTTP {code}")
    parts.extend(sorted(failure_kinds(exc)))
    parts.extend(_provider_details(exc))
    return f"{type(exc).__name__} ({', '.join(parts)})" if parts else type(exc).__name__
