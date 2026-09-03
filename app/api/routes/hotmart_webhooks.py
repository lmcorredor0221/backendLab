from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from sqlmodel import Session

from app.db import get_session
from app.models import HotmartWebhookIngestResponse
from app.services.hotmart.webhooks import process_hotmart_webhook


router = APIRouter(prefix="/webhooks/hotmart", tags=["hotmart-webhooks"])


@router.post("", response_model=HotmartWebhookIngestResponse)
async def receive_hotmart_webhook_route(
    request: Request,
    environment: str = Query(default="sandbox"),
    x_hotmart_hottok: str = Header(default="", alias="X-HOTMART-HOTTOK"),
    db: Session = Depends(get_session),
) -> HotmartWebhookIngestResponse:
    try:
        payload: dict[str, Any] = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload.") from exc
    try:
        hottok, hottok_source = _extract_hottok(
            request_headers=dict(request.headers),
            payload=payload,
            header_value=x_hotmart_hottok,
        )
        response = process_hotmart_webhook(
            db,
            payload=payload,
            hottok_header=hottok,
            hottok_source=hottok_source,
            request_headers=dict(request.headers),
            environment=environment,
        )
    except PermissionError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except ValueError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    db.commit()
    return response


def _extract_hottok(
    *,
    request_headers: dict[str, str],
    payload: dict[str, Any],
    header_value: str,
) -> tuple[str, str]:
    """Accept documented and observed HOTTOK placements without changing validation."""

    candidates = (
        ("x-hotmart-hottok", header_value),
        ("hottok-header", request_headers.get("hottok", "")),
        ("x-hottok-header", request_headers.get("x-hottok", "")),
        ("payload.hottok", str(payload.get("hottok") or "")),
    )
    for source, value in candidates:
        token = (value or "").strip()
        if token:
            return token, source
    return "", "missing"
