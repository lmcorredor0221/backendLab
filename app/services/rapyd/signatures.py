from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse


def compact_json_body(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return ""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def generate_rapyd_salt(length: int = 12) -> str:
    return secrets.token_urlsafe(max(8, length))[: max(8, length)]


def rapyd_unix_timestamp() -> str:
    return str(int(time.time()))


def rapyd_url_path(url_or_path: str) -> str:
    candidate = str(url_or_path or "").strip()
    if not candidate:
        return ""
    if candidate.startswith("http://") or candidate.startswith("https://"):
        parsed = urlparse(candidate)
        path = parsed.path or "/"
        return f"{path}?{parsed.query}" if parsed.query else path
    return candidate


def sign_rapyd_request(
    *,
    http_method: str,
    url_path: str,
    salt: str,
    timestamp: str,
    access_key: str,
    secret_key: str,
    body_string: str = "",
) -> str:
    to_sign = "".join(
        [
            http_method.strip().lower(),
            rapyd_url_path(url_path),
            salt,
            timestamp,
            access_key,
            secret_key,
            body_string,
        ]
    )
    return _hmac_sha256_base64_hex(to_sign, secret_key=secret_key)


def sign_rapyd_webhook(
    *,
    webhook_url: str,
    salt: str,
    timestamp: str,
    access_key: str,
    secret_key: str,
    body_string: str,
) -> str:
    to_sign = "".join([webhook_url, salt, timestamp, access_key, secret_key, body_string])
    return _hmac_sha256_base64_hex(to_sign, secret_key=secret_key)


def verify_rapyd_webhook_signature(
    *,
    raw_body: bytes,
    request_headers: Mapping[str, str],
    candidate_urls: list[str],
    access_key: str,
    secret_key: str,
) -> bool:
    if not access_key or not secret_key:
        return False
    received_access_key = _header_value(request_headers, "access_key")
    signature = _header_value(request_headers, "signature")
    salt = _header_value(request_headers, "salt")
    timestamp = _header_value(request_headers, "timestamp")
    if not all([received_access_key, signature, salt, timestamp]):
        return False
    if not hmac.compare_digest(received_access_key, access_key):
        return False
    body_string = raw_body.decode("utf-8", errors="replace")
    candidates: set[str] = set()
    for url in candidate_urls:
        normalized = str(url or "").strip()
        if not normalized:
            continue
        candidates.add(
            sign_rapyd_webhook(
                webhook_url=normalized,
                salt=salt,
                timestamp=timestamp,
                access_key=access_key,
                secret_key=secret_key,
                body_string=body_string,
            )
        )
        path = rapyd_url_path(normalized)
        if path and path != normalized:
            candidates.add(
                sign_rapyd_webhook(
                    webhook_url=path,
                    salt=salt,
                    timestamp=timestamp,
                    access_key=access_key,
                    secret_key=secret_key,
                    body_string=body_string,
                )
            )
    return any(hmac.compare_digest(signature, candidate) for candidate in candidates)


def _hmac_sha256_base64_hex(value: str, *, secret_key: str) -> str:
    hex_digest = hmac.new(secret_key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return base64.b64encode(hex_digest.encode("utf-8")).decode("ascii")


def _header_value(headers: Mapping[str, str], key: str) -> str:
    expected = key.lower()
    for header_key, value in headers.items():
        if str(header_key).lower() == expected:
            return str(value or "").strip()
    return ""
