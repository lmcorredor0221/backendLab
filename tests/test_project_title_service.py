from app.services.project_title_service import generate_commercial_project_title


def test_project_title_uses_commercial_aliases_and_max_length() -> None:
    title = generate_commercial_project_title(
        "Tenemos 6 personas revisando facturas en PDF y validándolas contra órdenes de compra.",
        language="es",
    )

    assert title == "Facturas IA"
    assert len(title) <= 30


def test_project_title_localizes_fallback_and_aliases() -> None:
    assert generate_commercial_project_title("Responder dudas repetitivas de clientes", language="en") == "Customer Support"
    assert generate_commercial_project_title("Auditar contratos e políticas internas", language="pt") == "Contratos IA"
    assert generate_commercial_project_title("", language="en") == "AI Project"


def test_project_title_compacts_generic_problem() -> None:
    title = generate_commercial_project_title(
        "Necesito automatizar la revisión diaria de reportes operativos internos con aprobaciones.",
        language="es",
    )

    assert 0 < len(title) <= 30
    assert title == "Revisión Diaria Reportes IA"
