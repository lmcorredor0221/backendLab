from __future__ import annotations

import hashlib
from datetime import timedelta
from uuid import UUID, uuid4

from dataclasses import dataclass

from sqlmodel import Session, select

from app.models import (
    AccessRequestCreateRequest,
    AccessRequestResponse,
    AccessRequestResolveRequest,
    CommercialAccessRequestRecord,
    CommercialAccessRequestStatus,
    CommercialAccessSnapshotV2,
    CommercialCapabilityDecisionEntry,
    CommercialCheckoutCompletionRequest,
    CommercialCheckoutSessionRequest,
    CommercialCheckoutSessionResponse,
    CommercialEntitlementRecord,
    CommercialEntitlementSource,
    CommercialEntitlementStatus,
    CommercialEntitlementSummary,
    CommercialEventRecord,
    CommercialOrderLineRecord,
    CommercialOrderLineResponse,
    CommercialOrderRecord,
    CommercialOrderResponse,
    CommercialOrderStatus,
    CommercialPaymentRecord,
    CommercialPaymentStatus,
    CommercialPriceStatus,
    CommercialProductStatus,
    CommercialProductType,
    CommercialTier,
    ProductCatalogRecord,
    ProductCatalogResponse,
    ProductPriceRecord,
    ProductPriceResponse,
    SessionRecord,
    UserRecord,
    WorkspaceMembershipRecord,
    WorkspaceRole,
    utc_now,
)
from app.services.commercial_event_catalog import enrich_commercial_event_metadata


@dataclass(frozen=True)
class EffectiveEntitlementState:
    tier: CommercialTier
    reason_code: str
    checkout_state: str
    purchase_refs: tuple[str, ...]
    entitlements: tuple[CommercialEntitlementSummary, ...]


TIER_RANKS: dict[CommercialTier, int] = {
    CommercialTier.blueprint: 1,
    CommercialTier.blueprint_pro: 2,
    CommercialTier.acp: 3,
}


def tier_rank(tier: CommercialTier) -> int:
    return TIER_RANKS[tier]


PRODUCT_SEED: tuple[dict, ...] = (
    {
        "product_key": "blueprint",
        "tier": CommercialTier.blueprint,
        "product_type": CommercialProductType.blueprint,
        "name": "Blueprint",
        "description": "Diseno integral visible dentro de Lean Agent Builder.",
        "scope": "project",
        "benefits": [
            "Visualizacion dentro de la plataforma.",
            "Arquitectura de muestra y narrativa de valor.",
            "Base para decidir si adquirir Blueprint Profesional o ACP.",
        ],
        "exclusions": ["Sin descarga, copia o exportacion externa."],
        "capabilities": ["blueprint.view", "acp.invite", "diagram.view.sample"],
        "metadata": {"commercial_stage": "free"},
    },
    {
        "product_key": "blueprint_pro",
        "tier": CommercialTier.blueprint_pro,
        "product_type": CommercialProductType.blueprint,
        "name": "Blueprint Profesional",
        "description": "Documento profesional descargable para decision, venta e implementacion guiada.",
        "scope": "project",
        "benefits": [
            "Descarga profesional del Blueprint.",
            "Arquitectura, alcance, reglas, herramientas, memoria, integraciones y roadmap.",
            "Estimacion de esfuerzo, tiempo y costo con comparativas comerciales.",
        ],
        "exclusions": [
            "No incluye Test Suite ACP.",
            "No incluye Prompt Pack ejecutable.",
            "No incluye paquete tecnico de construccion premium.",
        ],
        "capabilities": [
            "blueprint.download",
            "diagram.view.blueprint",
            "export_markdown",
            "export_json",
            "export_blueprint_core",
            "export_estimation_pack",
        ],
        "metadata": {"commercial_stage": "paid_blueprint"},
    },
    {
        "product_key": "acp",
        "tier": CommercialTier.acp,
        "product_type": CommercialProductType.acp,
        "name": "Agent Construction Package",
        "description": "Paquete tecnico portable para iniciar la construccion del sistema agentico.",
        "scope": "project",
        "benefits": [
            "Validacion, Test Suite, GAPs y preguntas de implementacion.",
            "Prompts, contratos, herramientas, memoria y artefactos tecnicos portables.",
            "Preparado para herramientas agenticas como Codex, Cursor, Claude Code o Copilot.",
        ],
        "exclusions": [
            "No ejecuta el despliegue final desde Lean Agent Builder.",
            "No instala IDEs, CLIs ni dependencias en el equipo del usuario.",
        ],
        "capabilities": [
            "acp.build",
            "acp.download",
            "diagram.view.acp",
            "export_construction_pack",
            "export_prompt_pack",
            "export_test_pack",
            "export_acp_zip",
            "library_workspace",
        ],
        "metadata": {"commercial_stage": "paid_acp"},
    },
)

import json
import urllib.request
from datetime import datetime
from app.models import BasePricesResponse, TRMResponse

TRM_SOURCE_LABEL = "Datos Abiertos Colombia / Superfinanciera"
TRM_DATASET_URL = "https://www.datos.gov.co/resource/32sa-8pi3.json?$limit=1&$order=vigenciahasta%20DESC"
TRM_REQUEST_TIMEOUT_SECONDS = 10

_TRM_CACHE: dict[str, Any] = {"rate": 3171.93, "date": "2026-08-06", "fetched_at": 0.0}

def get_today_trm_data(*, force_refresh: bool = False) -> dict[str, Any]:
    now_ts = datetime.utcnow().timestamp()
    if not force_refresh and _TRM_CACHE["fetched_at"] > 0 and (now_ts - _TRM_CACHE["fetched_at"]) < 3600:
        return _TRM_CACHE
    try:
        req = urllib.request.Request(TRM_DATASET_URL, headers={"User-Agent": "LeanAgentBuilder/1.0"})
        with urllib.request.urlopen(req, timeout=TRM_REQUEST_TIMEOUT_SECONDS) as response:
            if response.status == 200:
                raw = response.read().decode("utf-8")
                data = json.loads(raw)
                if data and isinstance(data, list) and len(data) > 0 and "valor" in data[0]:
                    rate = float(data[0]["valor"])
                    date_str = str(data[0].get("vigenciadesde", ""))[:10]
                    _TRM_CACHE["rate"] = rate
                    _TRM_CACHE["date"] = date_str
                    _TRM_CACHE["fetched_at"] = now_ts
                    return _TRM_CACHE
    except Exception:
        pass
    return _TRM_CACHE


def refresh_trm_on_login() -> dict[str, Any]:
    """Best-effort TRM refresh for successful logins.

    Login must stay available even if the public TRM source is temporarily down.
    """
    return get_today_trm_data(force_refresh=True)


PRICE_SEED: tuple[dict, ...] = (
    {
        "product_key": "blueprint",
        "price_code": "blueprint-free-v1",
        "currency": "USD",
        "unit_amount_cents": 0,
        "unit_amount_usd_cents": 0,
        "billing_period": "free",
    },
    {
        "product_key": "blueprint_pro",
        "price_code": "blueprint-pro-usd-v1",
        "currency": "USD",
        "unit_amount_cents": 6000,
        "unit_amount_usd_cents": 6000,
        "billing_period": "one_time",
    },
    {
        "product_key": "acp",
        "price_code": "acp-premium-usd-v1",
        "currency": "USD",
        "unit_amount_cents": 22000,
        "unit_amount_usd_cents": 22000,
        "billing_period": "one_time",
    },
)

REDACTED_KEYS = {"api_key", "token", "secret", "password", "content", "raw_prompt", "diagram_content"}


def ensure_commercial_seed(db: Session) -> None:
    for item in PRODUCT_SEED:
        product_payload = {**item, "metadata_payload": item.get("metadata", {})}
        product_payload.pop("metadata", None)
        existing = db.exec(
            select(ProductCatalogRecord).where(
                ProductCatalogRecord.product_key == item["product_key"],
                ProductCatalogRecord.version == 1,
            )
        ).first()
        if existing is None:
            db.add(ProductCatalogRecord(version=1, **product_payload))
            continue
        for key, value in product_payload.items():
            setattr(existing, key, value)
        existing.status = CommercialProductStatus.active
        existing.is_active = True
        existing.updated_at = utc_now()
        db.add(existing)

    for item in PRICE_SEED:
        price_payload = {**item, "metadata_payload": item.get("metadata", {})}
        price_payload.pop("metadata", None)
        existing = db.exec(select(ProductPriceRecord).where(ProductPriceRecord.product_key == item["product_key"])).first()
        if existing is None:
            db.add(ProductPriceRecord(version=1, **price_payload))
            continue
        # preserve existing custom USD amounts if updated by admin
        if existing.unit_amount_usd_cents <= 0 and item["unit_amount_usd_cents"] > 0:
            existing.unit_amount_usd_cents = item["unit_amount_usd_cents"]
            existing.unit_amount_cents = item["unit_amount_cents"]
        existing.status = CommercialPriceStatus.active
        existing.updated_at = utc_now()
        db.add(existing)


def list_catalog(db: Session) -> list[ProductCatalogResponse]:
    ensure_commercial_seed(db)
    products = db.exec(
        select(ProductCatalogRecord)
        .where(ProductCatalogRecord.is_active == True, ProductCatalogRecord.status == CommercialProductStatus.active)  # noqa: E712
        .order_by(ProductCatalogRecord.version, ProductCatalogRecord.product_key)
    ).all()
    return [serialize_product(db, item) for item in products]


def list_active_products(db: Session) -> list[ProductCatalogResponse]:
    return list_catalog(db)


def get_product(db: Session, product_key: str) -> ProductCatalogRecord:
    ensure_commercial_seed(db)
    product = db.exec(
        select(ProductCatalogRecord)
        .where(
            ProductCatalogRecord.product_key == product_key,
            ProductCatalogRecord.is_active == True,  # noqa: E712
            ProductCatalogRecord.status == CommercialProductStatus.active,
        )
        .order_by(ProductCatalogRecord.version.desc())
    ).first()
    if product is None:
        raise ValueError(f"Unknown commercial product: {product_key}")
    return product


def get_active_product(db: Session, product_key: str) -> ProductCatalogRecord:
    return get_product(db, product_key)


def tier_for_product(product_key: str) -> CommercialTier:
    for item in PRODUCT_SEED:
        if item["product_key"] == product_key:
            return item["tier"]
    raise ValueError(f"Unknown commercial product: {product_key}")


def get_price(db: Session, product_key: str, price_code: str = "") -> ProductPriceRecord:
    ensure_commercial_seed(db)
    statement = select(ProductPriceRecord).where(
        ProductPriceRecord.product_key == product_key,
        ProductPriceRecord.status == CommercialPriceStatus.active,
    )
    if price_code:
        statement = statement.where(ProductPriceRecord.price_code == price_code)
    price = db.exec(statement.order_by(ProductPriceRecord.version.desc())).first()
    if price is None:
        raise ValueError(f"No active price found for {product_key}")
    return price


def serialize_price(record: ProductPriceRecord) -> ProductPriceResponse:
    trm_info = get_today_trm_data()
    trm = trm_info["rate"]
    usd_cents = record.unit_amount_usd_cents if record.unit_amount_usd_cents > 0 else record.unit_amount_cents
    usd_val = usd_cents / 100.0
    cop_val = round(usd_val * trm)

    return ProductPriceResponse(
        price_code=record.price_code,
        currency="USD",
        unit_amount_cents=usd_cents,
        unit_amount_usd_cents=usd_cents,
        unit_amount_usd=usd_val,
        unit_amount_cop_calculated=cop_val,
        trm_applied=trm,
        billing_period=record.billing_period,
        version=record.version,
    )


def get_base_prices_summary(db: Session) -> BasePricesResponse:
    ensure_commercial_seed(db)
    pro = db.exec(select(ProductPriceRecord).where(ProductPriceRecord.product_key == "blueprint_pro")).first()
    acp = db.exec(select(ProductPriceRecord).where(ProductPriceRecord.product_key == "acp")).first()
    trm_info = get_today_trm_data()

    pro_val = (pro.unit_amount_usd_cents if pro and pro.unit_amount_usd_cents > 0 else 6000) / 100.0
    acp_val = (acp.unit_amount_usd_cents if acp and acp.unit_amount_usd_cents > 0 else 22000) / 100.0

    return BasePricesResponse(
        blueprint_free_usd=0.0,
        blueprint_pro_usd=pro_val,
        acp_premium_usd=acp_val,
        trm_cop=trm_info["rate"],
        updated_at=utc_now(),
    )


def update_base_prices_usd(db: Session, blueprint_pro_usd: float, acp_premium_usd: float) -> BasePricesResponse:
    ensure_commercial_seed(db)

    price_pro = db.exec(select(ProductPriceRecord).where(ProductPriceRecord.product_key == "blueprint_pro")).first()
    if price_pro:
        price_pro.unit_amount_usd_cents = int(round(blueprint_pro_usd * 100))
        price_pro.unit_amount_cents = price_pro.unit_amount_usd_cents
        price_pro.currency = "USD"
        price_pro.updated_at = utc_now()
        db.add(price_pro)

    price_acp = db.exec(select(ProductPriceRecord).where(ProductPriceRecord.product_key == "acp")).first()
    if price_acp:
        price_acp.unit_amount_usd_cents = int(round(acp_premium_usd * 100))
        price_acp.unit_amount_cents = price_acp.unit_amount_usd_cents
        price_acp.currency = "USD"
        price_acp.updated_at = utc_now()
        db.add(price_acp)

    db.commit()
    trm_info = get_today_trm_data()
    return BasePricesResponse(
        blueprint_free_usd=0.0,
        blueprint_pro_usd=blueprint_pro_usd,
        acp_premium_usd=acp_premium_usd,
        trm_cop=trm_info["rate"],
        updated_at=utc_now(),
    )


def serialize_product(db: Session, record: ProductCatalogRecord) -> ProductCatalogResponse:
    price = db.exec(
        select(ProductPriceRecord)
        .where(ProductPriceRecord.product_key == record.product_key, ProductPriceRecord.status == CommercialPriceStatus.active)
        .order_by(ProductPriceRecord.version.desc())
    ).first()
    return ProductCatalogResponse(
        product_key=record.product_key,
        tier=record.tier,
        product_type=record.product_type,
        name=record.name,
        description=record.description,
        scope=record.scope,
        benefits=list(record.benefits or []),
        exclusions=list(record.exclusions or []),
        capabilities=list(record.capabilities or []),
        price=serialize_price(price) if price is not None else None,
        version=record.version,
    )


def build_product_response(db: Session, record: ProductCatalogRecord) -> ProductCatalogResponse:
    return serialize_product(db, record)


def get_membership(db: Session, record: SessionRecord, user: UserRecord) -> WorkspaceMembershipRecord | None:
    return db.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == record.workspace_id,
            WorkspaceMembershipRecord.user_id == user.id,
            WorkspaceMembershipRecord.is_active == True,  # noqa: E712
        )
    ).first()


def role_for_user(db: Session, *, workspace_id: UUID, user_id: UUID) -> WorkspaceRole | None:
    membership = db.exec(
        select(WorkspaceMembershipRecord).where(
            WorkspaceMembershipRecord.workspace_id == workspace_id,
            WorkspaceMembershipRecord.user_id == user_id,
            WorkspaceMembershipRecord.is_active == True,  # noqa: E712
        )
    ).first()
    return membership.role if membership is not None else None


def is_active_entitlement(record: CommercialEntitlementRecord) -> bool:
    now = utc_now()
    return (
        record.status == CommercialEntitlementStatus.active
        and record.starts_at <= now
        and (record.ends_at is None or record.ends_at > now)
    )


def load_entitlements(db: Session, session_record: SessionRecord) -> list[CommercialEntitlementRecord]:
    return db.exec(
        select(CommercialEntitlementRecord)
        .where(
            CommercialEntitlementRecord.workspace_id == session_record.workspace_id,
            (CommercialEntitlementRecord.session_id == session_record.id)
            | (CommercialEntitlementRecord.session_id.is_(None)),
        )
        .order_by(CommercialEntitlementRecord.created_at.desc())
    ).all()


def serialize_entitlement(record: CommercialEntitlementRecord) -> CommercialEntitlementSummary:
    return CommercialEntitlementSummary(
        id=record.id,
        product_key=record.product_key,
        tier=record.tier,
        status=record.status,
        source=record.source,
        scope="workspace" if record.session_id is None else "project",
        starts_at=record.starts_at,
        ends_at=record.ends_at,
        purchase_ref=str(record.order_id or record.id),
        non_revenue=bool(record.metadata_payload.get("non_revenue")),
    )


def effective_tier_from_records(
    session_record: SessionRecord,
    entitlements: list[CommercialEntitlementRecord],
) -> tuple[CommercialTier, list[str], list[CommercialEntitlementRecord]]:
    active = [item for item in entitlements if is_active_entitlement(item)]
    effective_tier = session_record.commercial_tier
    purchase_refs: list[str] = []
    for entitlement in active:
        if tier_rank(entitlement.tier) > tier_rank(effective_tier):
            effective_tier = entitlement.tier
        purchase_refs.append(str(entitlement.id))
    if session_record.commercial_tier != CommercialTier.blueprint and not purchase_refs:
        purchase_refs.append(f"legacy-session:{session_record.id}:{session_record.commercial_tier.value}")
    return effective_tier, purchase_refs, active


def resolve_effective_entitlement_state(db: Session, session_record: SessionRecord) -> EffectiveEntitlementState:
    entitlements = load_entitlements(db, session_record)
    effective_tier, purchase_refs, active = effective_tier_from_records(session_record, entitlements)
    checkout_state = resolve_checkout_state(db, session_record)
    reason_code = "allowed" if effective_tier != CommercialTier.blueprint else "free_access"
    if checkout_state == "pending":
        reason_code = "checkout_pending"
    elif any(item.status == CommercialEntitlementStatus.suspended for item in entitlements):
        reason_code = "entitlement_suspended"
    elif any(item.status == CommercialEntitlementStatus.expired for item in entitlements):
        reason_code = "entitlement_expired"
    elif any(item.status == CommercialEntitlementStatus.revoked for item in entitlements):
        reason_code = "entitlement_revoked"
    elif any(item.status == CommercialEntitlementStatus.refunded for item in entitlements):
        reason_code = "entitlement_revoked"
    return EffectiveEntitlementState(
        tier=effective_tier,
        reason_code=reason_code,
        checkout_state=checkout_state,
        purchase_refs=tuple(purchase_refs),
        entitlements=tuple(serialize_entitlement(item) for item in entitlements),
    )


def map_reason(reason: str, policy_product: str, effective_tier: CommercialTier, role: WorkspaceRole | None) -> str:
    if reason == "allowed":
        return "allowed"
    if reason == "role":
        return "role_forbidden"
    if policy_product == "blueprint_pro" and tier_rank(effective_tier) < tier_rank(CommercialTier.blueprint_pro):
        return "purchase_required"
    if policy_product == "acp" and tier_rank(effective_tier) < tier_rank(CommercialTier.acp):
        return "purchase_required"
    if role is None:
        return "workspace_mismatch"
    return reason


def cta_for_reason(product: str, reason_code: str) -> str:
    if reason_code == "allowed":
        return ""
    if reason_code == "role_forbidden":
        return "Solicitar acceso"
    if product == "acp":
        return "Adquirir ACP"
    if product == "blueprint_pro":
        return "Adquirir Blueprint Profesional"
    return "Revisar acceso"


def build_commercial_access_v2(
    db: Session,
    session_record: SessionRecord,
    current_user: UserRecord,
    *,
    capabilities: tuple[str, ...] | None = None,
) -> CommercialAccessSnapshotV2:
    from app.services.commercial_access import CAPABILITY_POLICIES, TIER_LABELS, build_entitlement_context, resolve_capability_access

    membership = get_membership(db, session_record, current_user)
    entitlements = load_entitlements(db, session_record)
    effective_tier, purchase_refs, _ = effective_tier_from_records(session_record, entitlements)
    context = build_entitlement_context(
        tier=effective_tier,
        workspace_id=session_record.workspace_id,
        user_id=current_user.id,
        role=membership.role if membership else None,
        purchase_refs=tuple(purchase_refs),
    )
    selected_capabilities = capabilities or tuple(CAPABILITY_POLICIES.keys())
    decisions: list[CommercialCapabilityDecisionEntry] = []
    capability_reason_codes: list[str] = []
    for capability in selected_capabilities:
        decision = resolve_capability_access(context, capability)
        reason_code = map_reason(decision.reason, decision.product, effective_tier, context.role)
        if reason_code != "allowed":
            capability_reason_codes.append(reason_code)
        decisions.append(
            CommercialCapabilityDecisionEntry(
                capability=decision.capability,
                allowed=decision.allowed,
                current_tier=decision.current_tier,
                required_tier=decision.required_tier,
                product=decision.product,
                label=decision.label,
                reason_code=reason_code,
                cta_label=cta_for_reason(decision.product, reason_code),
            )
        )

    reason_code = "allowed" if all(item.allowed for item in decisions) else (capability_reason_codes[0] if capability_reason_codes else "free_access")
    return CommercialAccessSnapshotV2(
        workspace_id=session_record.workspace_id,
        session_id=session_record.id,
        user_id=current_user.id,
        role=context.role,
        tier=effective_tier,
        tier_label=TIER_LABELS[effective_tier],
        reason_code=reason_code,
        checkout_state=resolve_checkout_state(db, session_record),
        purchase_refs=purchase_refs,
        entitlements=[serialize_entitlement(item) for item in entitlements],
        capabilities=decisions,
    )


def resolve_checkout_state(db: Session, session_record: SessionRecord) -> str:
    latest_order = db.exec(
        select(CommercialOrderRecord)
        .where(CommercialOrderRecord.session_id == session_record.id)
        .order_by(CommercialOrderRecord.created_at.desc())
    ).first()
    if latest_order is None:
        return "not_started"
    if latest_order.status == CommercialOrderStatus.paid:
        return "confirmed"
    if latest_order.status == CommercialOrderStatus.pending:
        return "pending"
    return latest_order.status.value


def create_legacy_entitlement_if_needed(
    db: Session,
    session_record: SessionRecord,
    *,
    actor_user_id: UUID | None = None,
) -> CommercialEntitlementRecord | None:
    if session_record.commercial_tier == CommercialTier.blueprint:
        return None
    existing = db.exec(
        select(CommercialEntitlementRecord).where(
            CommercialEntitlementRecord.workspace_id == session_record.workspace_id,
            CommercialEntitlementRecord.session_id == session_record.id,
            CommercialEntitlementRecord.product_key == session_record.commercial_tier.value,
            CommercialEntitlementRecord.source.in_(
                (CommercialEntitlementSource.legacy_backfill, CommercialEntitlementSource.legacy_migration)
            ),
        )
    ).first()
    if existing is not None:
        return existing
    entitlement = CommercialEntitlementRecord(
        workspace_id=session_record.workspace_id,
        session_id=session_record.id,
        product_key=session_record.commercial_tier.value,
        tier=session_record.commercial_tier,
        status=CommercialEntitlementStatus.active,
        source=CommercialEntitlementSource.legacy_migration,
        granted_by_user_id=actor_user_id,
        metadata_payload={"non_revenue": True, "source": "session_commercial_tier"},
    )
    db.add(entitlement)
    return entitlement


def backfill_legacy_entitlements(db: Session) -> int:
    count = 0
    records = db.exec(select(SessionRecord).where(SessionRecord.commercial_tier != CommercialTier.blueprint)).all()
    for record in records:
        if create_legacy_entitlement_if_needed(db, record) is not None:
            count += 1
    return count


def sanitize_metadata(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if str(key).lower() in REDACTED_KEYS:
                sanitized[key] = "[redacted]"
            else:
                sanitized[key] = sanitize_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value[:20]]
    if isinstance(value, str) and len(value) > 240:
        return f"{value[:120]}...[truncated]"
    return value


def record_commercial_event(
    db: Session,
    *,
    workspace_id: UUID,
    session_id: UUID | None,
    user_id: UUID | None,
    event_key: str,
    product_key: str,
    source: str,
    metadata: dict | None = None,
    revenue_cents: int = 0,
    currency: str = "",
    correlation_id: str = "",
) -> CommercialEventRecord:
    event = CommercialEventRecord(
        workspace_id=workspace_id,
        session_id=session_id,
        user_id=user_id,
        event_key=event_key[:128],
        product_key=product_key[:128],
        source=source[:128],
        correlation_id=(correlation_id or str(uuid4()))[:128],
        revenue_cents=max(0, revenue_cents),
        currency=currency,
        metadata_payload=sanitize_metadata(enrich_commercial_event_metadata(event_key, metadata or {})),
    )
    db.add(event)
    return event


def ensure_buyer_can_checkout(db: Session, session_record: SessionRecord, user: UserRecord) -> WorkspaceRole:
    membership = get_membership(db, session_record, user)
    if membership is None:
        raise PermissionError("Workspace membership is required for checkout.")
    if membership.role not in {WorkspaceRole.owner, WorkspaceRole.admin}:
        raise PermissionError("Only workspace owners or admins can start checkout.")
    return membership.role


def calculate_project_upgrade_discount_cents(
    db: Session,
    session_id: UUID,
    target_product_key: str,
    base_amount_cents: int,
) -> tuple[int, int, bool]:
    """
    Verifica si en este proyecto específico (session_id) se adquirió previamente blueprint_pro.
    En caso afirmativo, retorna (descuento_cents, precio_neto_cents, es_upgrade).
    """
    if target_product_key != "acp":
        return 0, base_amount_cents, False

    entitlement = db.exec(
        select(CommercialEntitlementRecord).where(
            CommercialEntitlementRecord.session_id == session_id,
            CommercialEntitlementRecord.product_key == "blueprint_pro",
            CommercialEntitlementRecord.status == CommercialEntitlementStatus.active,
        )
    ).first()

    session_rec = db.get(SessionRecord, session_id) if session_id else None
    has_blueprint_pro = (entitlement is not None) or (
        session_rec is not None and session_rec.commercial_tier in (CommercialTier.blueprint_pro, CommercialTier.acp)
    )

    if not has_blueprint_pro:
        return 0, base_amount_cents, False

    try:
        bp_price = get_price(db, "blueprint_pro")
        bp_usd_cents = bp_price.unit_amount_usd_cents if bp_price.unit_amount_usd_cents > 0 else (bp_price.unit_amount_cents if bp_price.currency == "USD" else 4900)
    except Exception:
        bp_usd_cents = 4900

    if bp_usd_cents <= 0:
        bp_usd_cents = 4900

    discount_cents = min(bp_usd_cents, base_amount_cents)
    net_cents = max(0, base_amount_cents - discount_cents)
    return discount_cents, net_cents, True


def create_checkout_session(
    db: Session,
    *,
    request: CommercialCheckoutSessionRequest | None = None,
    payload: CommercialCheckoutSessionRequest | None = None,
    session_record: SessionRecord | None = None,
    record: SessionRecord | None = None,
    current_user: UserRecord,
    base_url: str = "",
) -> CommercialCheckoutSessionResponse:
    resolved_request = request or payload
    resolved_record = session_record or record
    if resolved_request is None or resolved_record is None:
        raise ValueError("Checkout request and project record are required.")
    ensure_buyer_can_checkout(db, resolved_record, current_user)
    product = get_product(db, resolved_request.product_key)
    if product.product_key == "blueprint":
        raise ValueError("Blueprint free does not require checkout.")
    price = get_price(db, product.product_key, resolved_request.price_code)

    discount_cents, net_total_cents, is_upgrade = calculate_project_upgrade_discount_cents(
        db, resolved_record.id, product.product_key, price.unit_amount_cents
    )

    idempotency_key = resolved_request.idempotency_key.strip() or checkout_idempotency_key(
        resolved_record.id,
        product.product_key,
        current_user.id,
    )
    existing = db.exec(
        select(CommercialOrderRecord).where(
            CommercialOrderRecord.workspace_id == resolved_record.workspace_id,
            CommercialOrderRecord.idempotency_key == idempotency_key,
        )
    ).first()
    if existing is not None:
        return serialize_checkout_response(db, existing)

    checkout_ref = f"sandbox_{uuid4().hex}"
    fallback_checkout_url = f"{base_url.rstrip('/')}/checkout/sandbox/{checkout_ref}" if base_url else f"/checkout/sandbox/{checkout_ref}"
    order = CommercialOrderRecord(
        workspace_id=resolved_record.workspace_id,
        session_id=resolved_record.id,
        buyer_user_id=current_user.id,
        currency=price.currency,
        subtotal_cents=price.unit_amount_cents,
        total_cents=net_total_cents,
        provider="sandbox",
        checkout_ref=checkout_ref,
        checkout_url=resolved_request.success_url or fallback_checkout_url,
        idempotency_key=idempotency_key,
        metadata_payload={
            "product_key": product.product_key,
            "price_code": price.price_code,
            "success_url": resolved_request.success_url,
            "cancel_url": resolved_request.cancel_url,
            "is_upgrade": is_upgrade,
            "upgrade_discount_cents": discount_cents,
            "base_product_cents": price.unit_amount_cents,
        },
    )
    db.add(order)
    db.flush()
    line = CommercialOrderLineRecord(
        order_id=order.id,
        product_key=product.product_key,
        price_code=price.price_code,
        quantity=1,
        unit_amount_cents=price.unit_amount_cents,
        total_amount_cents=price.unit_amount_cents,
        metadata_payload={"product_version": product.version, "price_version": price.version},
    )
    db.add(line)
    record_commercial_event(
        db,
        workspace_id=resolved_record.workspace_id,
        session_id=resolved_record.id,
        user_id=current_user.id,
        event_key="checkout_started",
        product_key=product.product_key,
        source="commerce_checkout",
        metadata={"order_id": str(order.id), "price_code": price.price_code},
        correlation_id=checkout_ref,
    )
    db.flush()
    return serialize_checkout_response(db, order)


def checkout_idempotency_key(session_id: UUID, product_key: str, user_id: UUID) -> str:
    digest = hashlib.sha256(f"{session_id}:{product_key}:{user_id}".encode("utf-8")).hexdigest()
    return digest[:64]


def serialize_checkout_response(db: Session, order: CommercialOrderRecord) -> CommercialCheckoutSessionResponse:
    line = db.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()
    entitlement = find_entitlement_for_order(db, order)
    return CommercialCheckoutSessionResponse(
        checkout_ref=order.checkout_ref,
        order_id=order.id,
        session_id=order.session_id or UUID(int=0),
        workspace_id=order.workspace_id,
        product_key=line.product_key if line else str(order.metadata_payload.get("product_key", "")),
        provider=order.provider,
        status=order.status,
        checkout_url=order.checkout_url,
        total_cents=order.total_cents,
        currency=order.currency,
        expires_at=order.created_at + timedelta(minutes=30),
        entitlement=serialize_entitlement(entitlement) if entitlement else None,
        next_action="refresh_access" if order.status == CommercialOrderStatus.paid else "open_checkout",
    )


def find_entitlement_for_order(db: Session, order: CommercialOrderRecord) -> CommercialEntitlementRecord | None:
    return db.exec(select(CommercialEntitlementRecord).where(CommercialEntitlementRecord.order_id == order.id)).first()


def complete_checkout_session(
    db: Session,
    *,
    checkout_ref: str,
    request: CommercialCheckoutCompletionRequest,
    current_user: UserRecord,
) -> CommercialCheckoutSessionResponse:
    order = db.exec(select(CommercialOrderRecord).where(CommercialOrderRecord.checkout_ref == checkout_ref)).first()
    if order is None:
        raise ValueError("Checkout session not found.")
    session_record = db.get(SessionRecord, order.session_id) if order.session_id else None
    if session_record is None:
        raise ValueError("Checkout session is not attached to a project.")
    ensure_buyer_can_checkout(db, session_record, current_user)

    if request.outcome == "cancel":
        order.status = CommercialOrderStatus.canceled
        order.updated_at = utc_now()
        db.add(order)
        record_commercial_event(
            db,
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            user_id=current_user.id,
            event_key="checkout_canceled",
            product_key=str(order.metadata_payload.get("product_key", "")),
            source="commerce_checkout",
            correlation_id=checkout_ref,
        )
        return serialize_checkout_response(db, order)
    if request.outcome == "failure":
        order.status = CommercialOrderStatus.failed
        order.updated_at = utc_now()
        db.add(order)
        record_commercial_event(
            db,
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            user_id=current_user.id,
            event_key="payment_failed",
            product_key=str(order.metadata_payload.get("product_key", "")),
            source="commerce_checkout",
            correlation_id=checkout_ref,
        )
        return serialize_checkout_response(db, order)

    if order.status == CommercialOrderStatus.paid:
        return serialize_checkout_response(db, order)

    line = db.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).first()
    if line is None:
        raise ValueError("Checkout order has no order line.")
    product = get_product(db, line.product_key)
    payment_id = request.provider_payment_id.strip() or f"sandbox_pay_{checkout_ref}"
    payment = db.exec(
        select(CommercialPaymentRecord).where(
            CommercialPaymentRecord.provider == order.provider,
            CommercialPaymentRecord.provider_payment_id == payment_id,
        )
    ).first()
    if payment is None:
        payment = CommercialPaymentRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            order_id=order.id,
            provider=order.provider,
            provider_payment_id=payment_id,
            provider_checkout_ref=checkout_ref,
            status=CommercialPaymentStatus.succeeded,
            amount_cents=order.total_cents,
            currency=order.currency,
            idempotency_key=f"{checkout_ref}:success",
            metadata_payload={"sandbox": True},
        )
        db.add(payment)
        db.flush()

    entitlement = find_entitlement_for_order(db, order)
    if entitlement is None:
        entitlement = CommercialEntitlementRecord(
            workspace_id=order.workspace_id,
            session_id=order.session_id,
            product_key=product.product_key,
            tier=product.tier,
            status=CommercialEntitlementStatus.active,
            source=CommercialEntitlementSource.checkout,
            order_id=order.id,
            order_line_id=line.id,
            payment_id=payment.id,
            granted_by_user_id=current_user.id,
            metadata_payload={"non_revenue": False, "provider": "sandbox"},
        )
        db.add(entitlement)

    if tier_rank(product.tier) > tier_rank(session_record.commercial_tier):
        session_record.commercial_tier = product.tier
        session_record.updated_at = utc_now()
        db.add(session_record)

    order.status = CommercialOrderStatus.paid
    order.paid_at = utc_now()
    order.updated_at = utc_now()
    db.add(order)
    record_commercial_event(
        db,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        user_id=current_user.id,
        event_key="payment_confirmed",
        product_key=product.product_key,
        source="commerce_checkout",
        revenue_cents=order.total_cents,
        currency=order.currency,
        metadata={"order_id": str(order.id), "entitlement_id": str(entitlement.id)},
        correlation_id=checkout_ref,
    )
    return serialize_checkout_response(db, order)


def complete_sandbox_checkout(
    db: Session,
    *,
    checkout_ref: str,
    payload: CommercialCheckoutCompletionRequest,
    current_user: UserRecord,
) -> CommercialOrderResponse:
    try:
        checkout = complete_checkout_session(
            db,
            checkout_ref=checkout_ref,
            request=payload,
            current_user=current_user,
        )
    except ValueError as exc:
        raise LookupError(str(exc)) from exc
    order = db.get(CommercialOrderRecord, checkout.order_id)
    if order is None:
        raise LookupError("Order not found after sandbox checkout.")
    return serialize_order(db, order)


def serialize_order(db: Session, order: CommercialOrderRecord) -> CommercialOrderResponse:
    lines = db.exec(select(CommercialOrderLineRecord).where(CommercialOrderLineRecord.order_id == order.id)).all()
    entitlement = find_entitlement_for_order(db, order)
    return CommercialOrderResponse(
        id=order.id,
        workspace_id=order.workspace_id,
        session_id=order.session_id,
        buyer_user_id=order.buyer_user_id,
        status=order.status,
        provider=order.provider,
        checkout_ref=order.checkout_ref,
        checkout_url=order.checkout_url,
        currency=order.currency,
        total_cents=order.total_cents,
        lines=[
            CommercialOrderLineResponse(
                product_key=line.product_key,
                price_code=line.price_code,
                quantity=line.quantity,
                total_amount_cents=line.total_amount_cents,
            )
            for line in lines
        ],
        entitlement=serialize_entitlement(entitlement) if entitlement else None,
        created_at=order.created_at,
        updated_at=order.updated_at,
    )


def build_order_response(db: Session, order: CommercialOrderRecord) -> CommercialOrderResponse:
    return serialize_order(db, order)


def create_access_request(
    db: Session,
    *,
    workspace_id: UUID,
    request: AccessRequestCreateRequest,
    current_user: UserRecord,
) -> AccessRequestResponse:
    from app.services.commercial_access import CAPABILITY_POLICIES

    session_record = db.get(SessionRecord, request.session_id)
    if session_record is None or session_record.workspace_id != workspace_id:
        raise ValueError("Project is not available in this workspace.")
    policy = CAPABILITY_POLICIES.get(request.capability)
    if policy is None:
        raise ValueError("Unknown capability.")
    membership = get_membership(db, session_record, current_user)
    if membership is None:
        raise PermissionError("Workspace membership is required.")
    existing = db.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.workspace_id == workspace_id,
            CommercialAccessRequestRecord.session_id == request.session_id,
            CommercialAccessRequestRecord.requester_user_id == current_user.id,
            CommercialAccessRequestRecord.capability == request.capability,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
        )
    ).first()
    if existing is not None:
        return serialize_access_request(existing)
    record = CommercialAccessRequestRecord(
        workspace_id=workspace_id,
        session_id=request.session_id,
        requester_user_id=current_user.id,
        capability=request.capability,
        product_key=policy.product,
        target_tier=policy.required_tier,
        reason=request.reason,
    )
    db.add(record)
    db.flush()
    record_commercial_event(
        db,
        workspace_id=workspace_id,
        session_id=request.session_id,
        user_id=current_user.id,
        event_key="access_request_created",
        product_key=policy.product,
        source="access_request",
        metadata={"capability": request.capability},
    )
    return serialize_access_request(record)


def request_access(
    db: Session,
    *,
    payload: AccessRequestCreateRequest,
    record: SessionRecord,
    current_user: UserRecord,
    product_key: str,
    target_tier: CommercialTier,
) -> AccessRequestResponse:
    existing = db.exec(
        select(CommercialAccessRequestRecord).where(
            CommercialAccessRequestRecord.workspace_id == record.workspace_id,
            CommercialAccessRequestRecord.session_id == record.id,
            CommercialAccessRequestRecord.requester_user_id == current_user.id,
            CommercialAccessRequestRecord.capability == payload.capability,
            CommercialAccessRequestRecord.status == CommercialAccessRequestStatus.pending,
        )
    ).first()
    if existing is not None:
        return serialize_access_request(existing)
    access_request = CommercialAccessRequestRecord(
        workspace_id=record.workspace_id,
        session_id=record.id,
        requester_user_id=current_user.id,
        capability=payload.capability,
        product_key=product_key,
        target_tier=target_tier,
        reason=payload.reason,
    )
    db.add(access_request)
    db.flush()
    record_commercial_event(
        db,
        workspace_id=record.workspace_id,
        session_id=record.id,
        user_id=current_user.id,
        event_key="access_request_created",
        product_key=product_key,
        source="access_request",
        metadata={"capability": payload.capability},
    )
    return serialize_access_request(access_request)


def resolve_access_request_by_id(
    db: Session,
    *,
    workspace_id: UUID,
    request_id: UUID,
    request: AccessRequestResolveRequest,
    current_user: UserRecord,
) -> AccessRequestResponse:
    record = db.get(CommercialAccessRequestRecord, request_id)
    if record is None or record.workspace_id != workspace_id:
        raise ValueError("Access request not found.")
    session_record = db.get(SessionRecord, record.session_id)
    if session_record is None:
        raise ValueError("Project is not available.")
    membership = get_membership(db, session_record, current_user)
    if membership is None or membership.role not in {WorkspaceRole.owner, WorkspaceRole.admin}:
        raise PermissionError("Only workspace owners or admins can resolve access requests.")
    if record.status != CommercialAccessRequestStatus.pending:
        return serialize_access_request(record)
    record.status = CommercialAccessRequestStatus(request.decision)
    record.resolver_user_id = current_user.id
    record.resolution_note = request.resolution_note
    record.resolved_at = utc_now()
    record.updated_at = utc_now()
    db.add(record)
    record_commercial_event(
        db,
        workspace_id=workspace_id,
        session_id=record.session_id,
        user_id=current_user.id,
        event_key=f"access_request_{record.status.value}",
        product_key=record.product_key,
        source="access_request",
        metadata={"capability": record.capability, "request_id": str(record.id)},
    )
    return serialize_access_request(record)


def resolve_access_request(
    db: Session,
    *,
    access_request: CommercialAccessRequestRecord,
    payload: AccessRequestResolveRequest,
    current_user: UserRecord,
) -> AccessRequestResponse:
    session_record = db.get(SessionRecord, access_request.session_id)
    if session_record is None:
        raise ValueError("Project is not available.")
    membership = get_membership(db, session_record, current_user)
    if membership is None or membership.role not in {WorkspaceRole.owner, WorkspaceRole.admin}:
        raise PermissionError("Only workspace owners or admins can resolve access requests.")
    if access_request.status != CommercialAccessRequestStatus.pending:
        return serialize_access_request(access_request)
    access_request.status = CommercialAccessRequestStatus(payload.decision)
    access_request.resolver_user_id = current_user.id
    access_request.resolution_note = payload.resolution_note
    access_request.resolved_at = utc_now()
    access_request.updated_at = utc_now()
    db.add(access_request)
    record_commercial_event(
        db,
        workspace_id=access_request.workspace_id,
        session_id=access_request.session_id,
        user_id=current_user.id,
        event_key=f"access_request_{access_request.status.value}",
        product_key=access_request.product_key,
        source="access_request",
        metadata={"capability": access_request.capability, "request_id": str(access_request.id)},
    )
    return serialize_access_request(access_request)


def serialize_access_request(record: CommercialAccessRequestRecord) -> AccessRequestResponse:
    return AccessRequestResponse(
        id=record.id,
        workspace_id=record.workspace_id,
        session_id=record.session_id,
        requester_user_id=record.requester_user_id,
        capability=record.capability,
        product_key=record.product_key,
        target_tier=record.target_tier,
        status=record.status,
        reason=record.reason,
        resolution_note=record.resolution_note,
        created_at=record.created_at,
        updated_at=record.updated_at,
        resolved_at=record.resolved_at,
    )


def build_access_request_response(record: CommercialAccessRequestRecord) -> AccessRequestResponse:
    return serialize_access_request(record)


def status_for_product(access: CommercialAccessSnapshotV2, product_key: str) -> str:
    if product_key == "blueprint":
        return "active"
    if any(item.product_key == product_key and item.status == CommercialEntitlementStatus.active for item in access.entitlements):
        return "active"
    if access.checkout_state == "pending":
        return "checkout_pending"
    return "available"


def can_checkout_product(access: CommercialAccessSnapshotV2, product_key: str) -> tuple[bool, str]:
    if product_key == "blueprint":
        return False, "El Blueprint gratuito no requiere checkout."
    if any(item.product_key == product_key and item.status == CommercialEntitlementStatus.active for item in access.entitlements):
        return False, "El producto ya esta activo para este proyecto."
    if access.role not in {WorkspaceRole.owner, WorkspaceRole.admin}:
        return False, "Tu rol debe ser owner o admin para iniciar checkout."
    return True, ""
