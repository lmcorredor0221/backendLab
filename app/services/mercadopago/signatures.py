from __future__ import annotations

import hashlib
import hmac
from typing import Mapping


def verify_mercadopago_webhook_signature(
    *,
    data_id: str,
    request_headers: Mapping[str, str],
    signing_secret: str,
) -> bool:
    secret = signing_secret.strip()
    signature_header = _header_value(request_headers, "x-signature")
    request_id = _header_value(request_headers, "x-request-id")
    if not data_id.strip() or not secret or not signature_header or not request_id:
        return False
    signature_parts = _signature_parts(signature_header)
    timestamp = signature_parts.get("ts", "")
    received = signature_parts.get("v1", "").lower()
    if not timestamp or not received:
        return False
    candidate_ids = {data_id.strip(), data_id.strip().lower()}
    for candidate_id in candidate_ids:
        manifest = f"id:{candidate_id};request-id:{request_id};ts:{timestamp};"
        expected = hmac.new(secret.encode("utf-8"), manifest.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(received, expected.lower()):
            return True
    return False


def sign_mercadopago_webhook(*, data_id: str, request_id: str, timestamp: str, signing_secret: str) -> str:
    manifest = f"id:{data_id};request-id:{request_id};ts:{timestamp};"
    return hmac.new(signing_secret.encode("utf-8"), manifest.encode("utf-8"), hashlib.sha256).hexdigest()


def _signature_parts(signature_header: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for item in signature_header.split(","):
        key, separator, value = item.partition("=")
        if separator:
            parts[key.strip()] = value.strip()
    return parts


def _header_value(headers: Mapping[str, str], key: str) -> str:
    expected = key.lower()
    for header_key, value in headers.items():
        if header_key.lower() == expected:
            return str(value)
    return ""
