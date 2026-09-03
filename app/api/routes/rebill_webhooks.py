from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session

from app.db import get_session
from app.models import CommerceProviderWebhookIngestResponse
from app.services.rebill.webhooks import process_rebill_webhook


router = APIRouter(tags=["rebill-webhooks"])


@router.post("/webhooks/rebill/{url_secret}", response_model=CommerceProviderWebhookIngestResponse)
@router.post("/webhooks/rebill/{url_secret}/{environment}", response_model=CommerceProviderWebhookIngestResponse)
async def receive_rebill_webhook_route(
    request: Request,
    url_secret: str,
    environment: str = "sandbox",
    db: Session = Depends(get_session),
) -> CommerceProviderWebhookIngestResponse:
    raw_body = await request.body()
    headers = {key: value for key, value in request.headers.items()}
    try:
        response = process_rebill_webhook(
            db,
            raw_body=raw_body,
            request_headers=headers,
            url_secret=url_secret,
            environment=environment,
        )
    except PermissionError as exc:
        db.commit()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    db.commit()
    return response
