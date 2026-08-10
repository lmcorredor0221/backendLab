from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.services.text_sanitization import read_sanitized_utf8_text


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_repo_document_lineage(repo_root: Path, relative_path: str) -> str:
    normalized = relative_path.strip().replace("\\", "/")
    if not normalized:
        return ""
    path = (repo_root / normalized).resolve()
    if not path.exists() or not path.is_file():
        return ""
    content = read_sanitized_utf8_text(path)
    return f"{normalized}::doc::{stable_hash(content)[:16]}"


def build_virtual_source_lineage(uri: str, content: str, *, kind: str = "state") -> str:
    normalized_uri = uri.strip() or "virtual://source"
    return f"{normalized_uri}::{kind}::{stable_hash(content)[:16]}"


def build_source_version(lineages: list[str], *, fallback: str = "") -> str:
    normalized = sorted({item.strip() for item in lineages if item.strip()})
    if normalized:
        return f"lineage::{stable_hash(json.dumps(normalized, ensure_ascii=True))[:16]}"
    return fallback.strip()
