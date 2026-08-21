from uuid import uuid4

import pytest

from app.services.product_processing import (
    ProductBuildAction,
    ProductBuildActionState,
    ProductBuildAttentionItem,
    ProductBuildAttentionSeverity,
    ProductBuildAttentionSummary,
    ProductBuildDeliverableState,
    ProductBuildDeliverableStatus,
    ProductBuildEntitlement,
    ProductBuildLifecycle,
    ProductBuildProductKey,
    ProductBuildProgress,
    ProductBuildRecoverableError,
    ProductBuildStatus,
    ProductProcessingMode,
    calculate_product_build_percent,
)


def _status_for(lifecycle: ProductBuildLifecycle) -> ProductBuildStatus:
    workspace_id = uuid4()
    session_id = uuid4()
    return ProductBuildStatus(
        workspace_id=workspace_id,
        session_id=session_id,
        product_key=ProductBuildProductKey.blueprint_basic,
        product_mode=ProductProcessingMode.basic_free,
        product_label="Blueprint Basico",
        lifecycle=lifecycle,
        entitlement=ProductBuildEntitlement(
            access_state="allowed" if lifecycle != ProductBuildLifecycle.not_purchased else "preview",
            is_purchased=lifecycle != ProductBuildLifecycle.not_purchased,
            purchase_required=lifecycle == ProductBuildLifecycle.not_purchased,
        ),
        progress=ProductBuildProgress(
            percent=calculate_product_build_percent(3, 5),
            completed_units=3,
            total_units=5,
            label="3 de 5 entregables listos",
        ),
        deliverables=[
            ProductBuildDeliverableStatus(
                deliverable_key="diagram.architecture",
                title="Arquitectura propuesta",
                deliverable_type="diagram",
                state=ProductBuildDeliverableState.available,
                stage_key="design",
                href="/projects/session/blueprint",
            )
        ],
        attention=ProductBuildAttentionSummary(
            total=1,
            warning_count=1,
            items=[
                ProductBuildAttentionItem(
                    key="att-1",
                    title="Revision recomendada",
                    severity=ProductBuildAttentionSeverity.warning,
                    product_key="blueprint_basic",
                    run_id="run-1",
                    step_id="step-1",
                    source="uncertainty_backlog",
                    reason="Hay una observacion no bloqueante.",
                )
            ],
        ),
        actions=[
            ProductBuildAction(
                action_key="open_blueprint",
                label="Ver Blueprint",
                state=ProductBuildActionState.available,
                primary=True,
            )
        ],
    )


@pytest.mark.parametrize(
    "lifecycle",
    [
        ProductBuildLifecycle.not_purchased,
        ProductBuildLifecycle.payment_pending,
        ProductBuildLifecycle.running,
        ProductBuildLifecycle.requires_attention,
        ProductBuildLifecycle.completed,
        ProductBuildLifecycle.error,
    ],
)
def test_product_build_status_serializes_supported_lifecycle_states(lifecycle: ProductBuildLifecycle) -> None:
    payload = _status_for(lifecycle).model_dump(mode="json")

    assert payload["contract_version"] == "product-build-status.v1"
    assert payload["lifecycle"] == lifecycle.value
    assert payload["progress"]["percent"] == 60
    assert payload["attention"]["items"][0]["severity"] == "warning"
    assert payload["attention"]["items"][0]["run_id"] == "run-1"
    assert payload["attention"]["items"][0]["step_id"] == "step-1"


def test_product_build_status_contains_recoverable_error_shape() -> None:
    status = _status_for(ProductBuildLifecycle.error)
    status.last_error = ProductBuildRecoverableError(
        code="deliverable_generation_timeout",
        title="No se completo la generacion",
        message="El proceso puede reintentarse desde el checkpoint.",
        technical_message="timeout while generating diagram.architecture",
        retry_action_key="retry_product_build",
        trace_refs=["job:diagram.architecture"],
    )

    payload = status.model_dump(mode="json")

    assert payload["last_error"]["recoverable"] is True
    assert payload["last_error"]["retry_action_key"] == "retry_product_build"
    assert payload["last_error"]["trace_refs"] == ["job:diagram.architecture"]


def test_calculate_product_build_percent_clamps_and_handles_empty_totals() -> None:
    assert calculate_product_build_percent(1, 4) == 25
    assert calculate_product_build_percent(20, 10) == 100
    assert calculate_product_build_percent(-1, 10) == 0
    assert calculate_product_build_percent(1, 0) == 0
