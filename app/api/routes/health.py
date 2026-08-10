from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db import get_session
from app.services.operations_service import build_minimal_health_payload


router = APIRouter(tags=["health"])


@router.get("/health")
def healthcheck(db: Session = Depends(get_session)) -> dict[str, object]:
    return build_minimal_health_payload(db)
