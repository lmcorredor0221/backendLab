from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.services.deliverable_catalog.contracts import DeliverableCatalog


REPO_ROOT = Path(__file__).resolve().parents[4]
DELIVERABLE_CATALOG_PATH = REPO_ROOT / "shared_specs" / "deliverable-catalog.v1.json"


class DeliverableCatalogValidationError(ValueError):
    pass


def validate_deliverable_catalog(payload: dict[str, Any]) -> list[str]:
    try:
        DeliverableCatalog.model_validate(payload)
    except ValidationError as exc:
        return [f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()]
    return []


@lru_cache(maxsize=1)
def load_seed_deliverable_catalog() -> DeliverableCatalog:
    payload = json.loads(DELIVERABLE_CATALOG_PATH.read_text(encoding="utf-8"))
    errors = validate_deliverable_catalog(payload)
    if errors:
        raise DeliverableCatalogValidationError("; ".join(errors))
    return DeliverableCatalog.model_validate(payload)
