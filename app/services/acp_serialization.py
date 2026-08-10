from __future__ import annotations

import json
from datetime import date, datetime, time
from enum import Enum
from hashlib import sha256
from typing import Any
from uuid import UUID

import yaml


def _to_serializable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, (datetime, date, time)):
        return payload.isoformat()
    if isinstance(payload, UUID):
        return str(payload)
    if isinstance(payload, Enum):
        return payload.value
    if isinstance(payload, dict):
        return {str(key): _to_serializable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_to_serializable(item) for item in payload]
    if isinstance(payload, tuple):
        return [_to_serializable(item) for item in payload]
    return payload


def normalize_text_document(content: str) -> str:
    text = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(normalized_lines).strip("\n")
    return f"{normalized}\n" if normalized else ""


def build_content_hash(content: str) -> str:
    normalized = normalize_text_document(content)
    return sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def serialize_json_document(payload: Any) -> str:
    serialized = json.dumps(
        _to_serializable(payload),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    return normalize_text_document(serialized)


def serialize_yaml_document(payload: Any) -> str:
    serialized = yaml.safe_dump(
        _to_serializable(payload),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    if serialized.endswith("...\n"):
        serialized = serialized[:-4]
    return normalize_text_document(serialized)


def serialize_markdown_document(content: str) -> str:
    return normalize_text_document(content)
