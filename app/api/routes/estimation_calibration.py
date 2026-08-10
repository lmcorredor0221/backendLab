from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db import get_session
from app.models import EstimationCalibrationDashboard, UserRecord
from app.services.auth_service import get_current_user
from app.services.estimation_calibration import build_estimation_calibration_dashboard
from app.services.stage5_service import FEATURE_FLAG_ESTIMATION, is_feature_flag_enabled
from app.services.workspace_access import WorkspaceAccessContext, get_current_workspace_context
from app.services.workspace_bootstrap import apply_workspace_bootstrap


router = APIRouter(prefix="/estimation", tags=["estimation"])


@router.get("/calibration", response_model=EstimationCalibrationDashboard)
def get_estimation_calibration_dashboard_route(
    db: Session = Depends(get_session),
    current_user: UserRecord = Depends(get_current_user),
    workspace_context: WorkspaceAccessContext = Depends(get_current_workspace_context),
) -> EstimationCalibrationDashboard:
    _ = current_user
    apply_workspace_bootstrap(db, workspace_context.workspace.id)
    if not is_feature_flag_enabled(db, FEATURE_FLAG_ESTIMATION, workspace_id=workspace_context.workspace.id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Comparative estimation feature flag is disabled")
    return build_estimation_calibration_dashboard(db, workspace_context.workspace.id)
