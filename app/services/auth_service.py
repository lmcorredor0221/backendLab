from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db import engine, get_session
from app.models import AuthTokenRecord, UserRecord


security = HTTPBearer(auto_error=False)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt_bytes = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 390000)
    return f"{salt_bytes.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt_hex, digest_hex = password_hash.split("$", 1)
    except ValueError:
        return False

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        390000,
    ).hex()
    return hmac.compare_digest(derived, digest_hex)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_access_token(db: Session, user: UserRecord) -> tuple[str, datetime]:
    settings = get_settings()
    raw_token = secrets.token_urlsafe(32)
    expires_at = utc_now() + timedelta(hours=settings.auth_token_ttl_hours)
    db.add(
        AuthTokenRecord(
            user_id=user.id,
            token_hash=hash_token(raw_token),
            expires_at=expires_at,
        )
    )
    db.commit()
    return raw_token, expires_at


def revoke_access_token(db: Session, raw_token: str) -> None:
    token_hash = hash_token(raw_token)
    token_record = db.exec(select(AuthTokenRecord).where(AuthTokenRecord.token_hash == token_hash)).first()
    if token_record is None:
        return
    db.delete(token_record)
    db.commit()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_session),
) -> UserRecord:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    token_hash = hash_token(credentials.credentials)
    token_record = db.exec(select(AuthTokenRecord).where(AuthTokenRecord.token_hash == token_hash)).first()
    if token_record is None or token_record.expires_at <= utc_now():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = db.get(UserRecord, token_record.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")

    token_record.last_used_at = utc_now()
    db.add(token_record)
    db.commit()
    return user


def seed_default_user() -> None:
    settings = get_settings()
    with Session(engine) as db:
        from app.services.workspace_access import ensure_personal_workspace
        from app.services.runtime_governance_bootstrap import ensure_platform_admin_role

        existing = db.exec(select(UserRecord).where(UserRecord.email == settings.local_admin_email)).first()
        if existing is not None:
            if existing.full_name != settings.local_admin_name:
                existing.full_name = settings.local_admin_name
                existing.updated_at = utc_now()
                db.add(existing)
                db.commit()
            ensure_personal_workspace(db, existing)
            ensure_platform_admin_role(db, existing)
            return

        user = UserRecord(
            email=settings.local_admin_email,
            full_name=settings.local_admin_name,
            password_hash=hash_password(settings.local_admin_password),
            email_verified=True,
            email_verified_at=utc_now(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        ensure_personal_workspace(db, user)
        ensure_platform_admin_role(db, user)


def validate_password_strength(password: str) -> None:
    if len(password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe tener al menos 8 caracteres.",
        )
    if not any(c.isupper() for c in password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe incluir al menos una letra mayúscula.",
        )
    if not any(c.islower() for c in password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe incluir al menos una letra minúscula.",
        )
    if not any(c.isdigit() for c in password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe incluir al menos un número.",
        )
    special_chars = "@$!%*?&#_-+=/(){}[]"
    if not any(c in special_chars for c in password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe incluir al menos un carácter especial (@$!%*?&#_-).",
        )


def register_user(
    db: Session,
    payload: UserRegisterRequest,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> tuple[UserRecord, str, datetime]:
    from app.models import UserLegalAcceptanceRecord
    from app.services.workspace_access import ensure_personal_workspace

    # Bot protection
    if payload.honeypot_field and payload.honeypot_field.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Petición rechazada por verificación de seguridad antibot.")

    # Passwords match
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Las contraseñas no coinciden.")

    validate_password_strength(payload.password)

    # Mandatory Legal Acceptances
    if not (payload.accept_terms and payload.accept_privacy and payload.accept_data_treatment):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debe aceptar expresamente los Términos y Condiciones, la Política de Tratamiento de Datos Personales y la Política de Privacidad.",
        )

    normalized_email = payload.email.strip().lower()
    existing = db.exec(select(UserRecord).where(UserRecord.email == normalized_email)).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="El correo electrónico ya se encuentra registrado en la plataforma. Por favor inicia sesión.",
        )

    user = UserRecord(
        email=normalized_email,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        email_verified=True,
        email_verified_at=utc_now(),
        consent_system_notifications=payload.consent_system_notifications,
        consent_commercial_promotions=payload.consent_commercial_promotions,
        consent_events_newsletters=payload.consent_events_newsletters,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Store Audit Evidence of Legal Acceptance
    now_ts = utc_now()
    doc_types = ["terms_and_conditions", "privacy_policy", "data_treatment_policy"]
    for doc in doc_types:
        acceptance = UserLegalAcceptanceRecord(
            user_id=user.id,
            document_type=doc,
            document_version="v1.0-2026-08",
            accepted=True,
            accepted_at=now_ts,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.add(acceptance)
    db.commit()

    # Create default personal workspace
    workspace_name = payload.workspace_name.strip() if payload.workspace_name and payload.workspace_name.strip() else f"Workspace de {user.full_name}"
    ensure_personal_workspace(db, user, default_name=workspace_name)

    token, expires_at = issue_access_token(db, user)
    return user, token, expires_at


def get_user_consents(user: UserRecord) -> UserConsentResponse:
    return UserConsentResponse(
        user_id=user.id,
        consent_system_notifications=user.consent_system_notifications,
        consent_commercial_promotions=user.consent_commercial_promotions,
        consent_events_newsletters=user.consent_events_newsletters,
        updated_at=user.updated_at,
    )


def update_user_consents(db: Session, user: UserRecord, payload: UserConsentUpdateRequest) -> UserConsentResponse:
    user.consent_system_notifications = payload.consent_system_notifications
    user.consent_commercial_promotions = payload.consent_commercial_promotions
    user.consent_events_newsletters = payload.consent_events_newsletters
    user.updated_at = utc_now()
    db.add(user)
    db.commit()
    db.refresh(user)
    return get_user_consents(user)


def update_user_language(db: Session, user: UserRecord, payload: UserLanguageUpdateRequest) -> UserLanguageResponse:
    lang = payload.preferred_language.strip().lower()
    if lang not in {"es", "en", "pt"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Idioma no soportado. Debe ser 'es', 'en' o 'pt'.")
    user.preferred_language = lang
    user.updated_at = utc_now()
    db.add(user)
    db.commit()
    db.refresh(user)
    return UserLanguageResponse(
        user_id=user.id,
        preferred_language=user.preferred_language,
        updated_at=user.updated_at,
    )
