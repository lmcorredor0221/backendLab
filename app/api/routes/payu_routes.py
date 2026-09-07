from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session

from app.db import get_session
from app.models import CommerceProviderWebhookIngestResponse
from app.services.payu.checkout_redirect import render_payu_checkout_redirect, resolve_payu_response_redirect
from app.services.payu.webhooks import process_payu_webhook


router = APIRouter(tags=["payu"])


@router.get("/commerce/checkout-redirects/payu/{checkout_ref}", response_class=HTMLResponse)
def receive_payu_checkout_redirect_route(
    checkout_ref: str,
    db: Session = Depends(get_session),
) -> HTMLResponse:
    try:
        html = render_payu_checkout_redirect(db, checkout_ref=checkout_ref)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return HTMLResponse(content=html, headers={"Cache-Control": "no-store"})


@router.get("/commerce/checkout-responses/payu/{checkout_ref}")
def receive_payu_checkout_response_route(
    checkout_ref: str,
    request: Request,
    db: Session = Depends(get_session),
) -> RedirectResponse:
    try:
        redirect_url = resolve_payu_response_redirect(db, checkout_ref=checkout_ref, query_params=request.query_params)
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)


@router.post("/webhooks/payu/{url_secret}", response_model=CommerceProviderWebhookIngestResponse)
@router.post("/webhooks/payu/{url_secret}/{environment}", response_model=CommerceProviderWebhookIngestResponse)
async def receive_payu_webhook_route(
    request: Request,
    url_secret: str,
    environment: str = "sandbox",
    db: Session = Depends(get_session),
) -> CommerceProviderWebhookIngestResponse:
    raw_body = await request.body()
    headers = {key: value for key, value in request.headers.items()}
    try:
        response = process_payu_webhook(
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
