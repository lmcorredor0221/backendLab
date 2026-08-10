from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session, select

from app.db import get_session
from app.models import (
    AuthUser,
    LoginRequest,
    LoginResponse,
    UserConsentResponse,
    UserConsentUpdateRequest,
    UserLanguageResponse,
    UserLanguageUpdateRequest,
    UserRecord,
    UserRegisterRequest,
    WorkspaceSelectionRequest,
)
from app.services.auth_service import (
    get_current_user,
    issue_access_token,
    revoke_access_token,
    verify_password,
)
from app.services.commerce_service import refresh_trm_on_login
from app.services.workspace_access import (
    WorkspaceAccessContext,
    build_auth_user,
    get_current_workspace_context,
    resolve_workspace_access,
)


router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_session)) -> LoginResponse:
    user = db.exec(select(UserRecord).where(UserRecord.email == payload.email)).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    refresh_trm_on_login()
    access_token, expires_at = issue_access_token(db, user)
    return LoginResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=build_auth_user(db, user),
    )


@router.get("/me", response_model=AuthUser)
def me(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> AuthUser:
    return build_auth_user(db, current_user, requested_workspace_id=workspace_context.workspace.id)


@router.post("/workspaces/select", response_model=AuthUser)
def select_workspace(
    payload: WorkspaceSelectionRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> AuthUser:
    resolve_workspace_access(db, current_user, requested_workspace_id=payload.workspace_id)
    return build_auth_user(db, current_user, requested_workspace_id=payload.workspace_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> None:
    del current_user
    if credentials is None:
        return
    revoke_access_token(db, credentials.credentials)


@router.post("/register", response_model=LoginResponse)
def register(
    payload: UserRegisterRequest,
    request: Request,
    db: Session = Depends(get_session),
) -> LoginResponse:
    from app.services.auth_service import register_user

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    user, access_token, expires_at = register_user(db, payload, ip_address=client_ip, user_agent=user_agent)
    return LoginResponse(
        access_token=access_token,
        expires_at=expires_at,
        user=build_auth_user(db, user),
    )


@router.get("/consents", response_model=UserConsentResponse)
def get_consents_route(
    current_user: UserRecord = Depends(get_current_user),
) -> UserConsentResponse:
    from app.services.auth_service import get_user_consents
    return get_user_consents(current_user)


@router.patch("/consents", response_model=UserConsentResponse)
def update_consents_route(
    payload: UserConsentUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> UserConsentResponse:
    from app.services.auth_service import update_user_consents
    return update_user_consents(db, current_user, payload)


@router.patch("/language", response_model=UserLanguageResponse)
def update_language_route(
    payload: UserLanguageUpdateRequest,
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
) -> UserLanguageResponse:
    from app.services.auth_service import update_user_language
    return update_user_language(db, current_user, payload)
