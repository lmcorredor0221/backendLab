from __future__ import annotations

from urllib.parse import urlparse

from fastapi import HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from sqlmodel import Session, select

from app.core.config import get_settings
from app.models import (
    GoogleAuthProfile,
    GoogleAuthRequest,
    GoogleAuthResponse,
    UserAuthIdentityRecord,
    UserRecord,
)
from app.services.auth_service import (
    create_user_account,
    issue_access_token,
    utc_now,
    verify_password,
)
from app.services.workspace_access import build_auth_user


GOOGLE_PROVIDER = "google"


def _normalize_origin(value: str) -> str:
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _validate_origin(origin: str | None) -> None:
    if not origin:
        return

    settings = get_settings()
    allowed_origins = {
        normalized
        for candidate in [*settings.cors_origins, settings.commerce_public_base_url]
        if (normalized := _normalize_origin(candidate))
    }
    if _normalize_origin(origin) not in allowed_origins:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Origen de autenticación no permitido.")


def _verify_google_credential(credential: str) -> dict[str, object]:
    settings = get_settings()
    if not settings.google_auth_enabled or not settings.google_client_id.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El acceso con Google todavía no está configurado.",
        )

    if not credential.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Credencial de Google requerida.")

    try:
        claims = id_token.verify_oauth2_token(
            credential.strip(),
            google_requests.Request(),
            settings.google_client_id.strip(),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La credencial de Google no es válida o expiró.",
        ) from exc

    subject = str(claims.get("sub") or "").strip()
    email = str(claims.get("email") or "").strip().lower()
    full_name = str(claims.get("name") or "").strip() or email.split("@", 1)[0]
    if not subject or not email or claims.get("email_verified") is not True:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google no confirmó una identidad con correo verificado.",
        )

    return {
        "subject": subject,
        "email": email,
        "full_name": full_name,
    }


def _profile(claims: dict[str, object]) -> GoogleAuthProfile:
    return GoogleAuthProfile(
        email=str(claims["email"]),
        full_name=str(claims["full_name"]),
    )


def _authenticated_response(
    db: Session,
    user: UserRecord,
    *,
    profile: GoogleAuthProfile,
    is_new_user: bool,
) -> GoogleAuthResponse:
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Usuario inactivo.")

    access_token, expires_at = issue_access_token(db, user)
    return GoogleAuthResponse(
        status="authenticated",
        profile=profile,
        access_token=access_token,
        expires_at=expires_at,
        user=build_auth_user(db, user),
        is_new_user=is_new_user,
    )


def _link_google_identity(db: Session, user: UserRecord, *, subject: str, profile: GoogleAuthProfile) -> None:
    db.add(
        UserAuthIdentityRecord(
            user_id=user.id,
            provider=GOOGLE_PROVIDER,
            provider_subject=subject,
            email_at_link=profile.email,
        )
    )
    db.commit()


def authenticate_with_google(
    db: Session,
    payload: GoogleAuthRequest,
    *,
    origin: str | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> GoogleAuthResponse:
    _validate_origin(origin)
    claims = _verify_google_credential(payload.credential)
    profile = _profile(claims)
    subject = str(claims["subject"])

    identity = db.exec(
        select(UserAuthIdentityRecord).where(
            UserAuthIdentityRecord.provider == GOOGLE_PROVIDER,
            UserAuthIdentityRecord.provider_subject == subject,
        )
    ).first()
    if identity is not None:
        user = db.get(UserRecord, identity.user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="La identidad de Google no está asociada a una cuenta válida.",
            )
        identity.last_login_at = utc_now()
        identity.email_at_link = profile.email
        db.add(identity)
        db.commit()
        return _authenticated_response(db, user, profile=profile, is_new_user=False)

    existing_user = db.exec(select(UserRecord).where(UserRecord.email == profile.email)).first()
    if existing_user is not None:
        if not existing_user.password_hash:
            _link_google_identity(db, existing_user, subject=subject, profile=profile)
            return _authenticated_response(db, existing_user, profile=profile, is_new_user=False)

        if not payload.password:
            return GoogleAuthResponse(status="link_required", profile=profile)
        if not verify_password(payload.password, existing_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="La contraseña actual no permite vincular esta cuenta con Google.",
            )

        _link_google_identity(db, existing_user, subject=subject, profile=profile)
        return _authenticated_response(db, existing_user, profile=profile, is_new_user=False)

    if not (payload.accept_terms and payload.accept_privacy and payload.accept_data_treatment):
        return GoogleAuthResponse(status="registration_required", profile=profile)

    user = create_user_account(
        db,
        email=profile.email,
        full_name=profile.full_name,
        password_hash=None,
        workspace_name=payload.workspace_name,
        consent_system_notifications=payload.consent_system_notifications,
        consent_commercial_promotions=payload.consent_commercial_promotions,
        consent_events_newsletters=payload.consent_events_newsletters,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(
        UserAuthIdentityRecord(
            user_id=user.id,
            provider=GOOGLE_PROVIDER,
            provider_subject=subject,
            email_at_link=profile.email,
        )
    )
    db.commit()
    return _authenticated_response(db, user, profile=profile, is_new_user=True)
