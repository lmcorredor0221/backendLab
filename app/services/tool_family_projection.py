from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
import unicodedata


WHATSAPP_CANONICAL_CONNECTOR_KEY = "whatsapp_cloud_api"
WHATSAPP_CANONICAL_TOOL_NAME = "whatsapp_business_messaging"

WHATSAPP_CONNECTOR_ALIASES = {
    WHATSAPP_CANONICAL_CONNECTOR_KEY: WHATSAPP_CANONICAL_CONNECTOR_KEY,
    WHATSAPP_CANONICAL_TOOL_NAME: WHATSAPP_CANONICAL_CONNECTOR_KEY,
    "whatsapp_cloud_api_connector": WHATSAPP_CANONICAL_CONNECTOR_KEY,
    "whatsapp_business_api": WHATSAPP_CANONICAL_CONNECTOR_KEY,
    "whatsapp_business": WHATSAPP_CANONICAL_CONNECTOR_KEY,
    "whatsapp_messaging": WHATSAPP_CANONICAL_CONNECTOR_KEY,
    "whatsapp": WHATSAPP_CANONICAL_CONNECTOR_KEY,
}

GOOGLE_WORKSPACE_CONNECTOR_KEYS = {
    "google_drive_file_picker",
    "google_sheets_read_table",
    "google_calendar_availability_reader",
    "google_calendar_event_creator",
    "gmail_draft_creator",
    "gmail_send_message",
}

GOOGLE_WORKSPACE_CONNECTOR_ALIASES = {
    **{key: key for key in GOOGLE_WORKSPACE_CONNECTOR_KEYS},
    "google_drive_api": "google_drive_file_picker",
    "drive_api": "google_drive_file_picker",
    "drive_documents_api": "google_drive_file_picker",
    "google_drive": "google_drive_file_picker",
    "google_sheets_api": "google_sheets_read_table",
    "sheets_api": "google_sheets_read_table",
    "google_sheets": "google_sheets_read_table",
    "gsheets_api": "google_sheets_read_table",
    "google_calendar_api": "google_calendar_availability_reader",
    "calendar_api": "google_calendar_availability_reader",
    "google_calendar_reader": "google_calendar_availability_reader",
    "google_calendar_create_api": "google_calendar_event_creator",
    "google_calendar_event_api": "google_calendar_event_creator",
    "gmail_api": "gmail_draft_creator",
    "google_gmail_api": "gmail_draft_creator",
    "gmail_drafts_api": "gmail_draft_creator",
    "gmail_send_api": "gmail_send_message",
}

ODOO_CANONICAL_CONNECTOR_KEYS = {
    "odoo_partner_read",
    "odoo_crm_lead_read",
    "odoo_sale_order_read",
    "odoo_sale_quote_create",
    "odoo_activity_create",
    "odoo_crm_lead_update",
}

ODOO_LEGACY_CONNECTOR_ALIASES = {
    **{key: key for key in ODOO_CANONICAL_CONNECTOR_KEYS},
    "odoo_crm_api": "odoo_partner_read",
    "odoo_write_api": "odoo_crm_lead_update",
    "odoo_partner_api": "odoo_partner_read",
    "odoo_sales_api": "odoo_sale_order_read",
    "odoo_quote_api": "odoo_sale_quote_create",
}

ODOO_QUOTE_SIGNAL_TERMS = (
    "cotizacion",
    "cotizaciones",
    "quote",
    "quotes",
    "propuesta comercial",
    "propuestas comerciales",
    "sale.order",
    "sale order",
    "sales order",
    "pedido de venta",
    "pedidos de venta",
)

SIDE_EFFECT_CONNECTOR_KEYS = {
    "google_calendar_event_creator",
    "gmail_draft_creator",
    "gmail_send_message",
    "odoo_sale_quote_create",
    "odoo_activity_create",
    "odoo_crm_lead_update",
    WHATSAPP_CANONICAL_CONNECTOR_KEY,
}


def normalize_tool_identifier(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _tool_identity_values(tool: Any) -> set[str]:
    return {
        normalize_tool_identifier(getattr(tool, "connector_key", "")),
        normalize_tool_identifier(getattr(tool, "registered_api_ref", "")),
        normalize_tool_identifier(getattr(tool, "name", "")),
    } - {""}


def _clone_tool(tool: Any, **overrides: Any) -> SimpleNamespace:
    if hasattr(tool, "model_dump"):
        payload = tool.model_dump(mode="json")
    elif isinstance(tool, dict):
        payload = dict(tool)
    else:
        payload = dict(getattr(tool, "__dict__", {}) or {})
    payload.update(overrides)
    return SimpleNamespace(**payload)


def resolve_whatsapp_connector_key(tool: Any) -> str:
    for value in _tool_identity_values(tool):
        key = WHATSAPP_CONNECTOR_ALIASES.get(value)
        if key:
            return key
    return ""


def resolve_google_workspace_connector_key(tool: Any) -> str:
    for value in _tool_identity_values(tool):
        key = GOOGLE_WORKSPACE_CONNECTOR_ALIASES.get(value)
        if key:
            return key
    return ""


def resolve_odoo_connector_key(tool: Any) -> str:
    for value in _tool_identity_values(tool):
        key = ODOO_LEGACY_CONNECTOR_ALIASES.get(value)
        if key:
            return key
    return ""


def project_tool_for_construction(tool: Any) -> Any:
    if resolve_whatsapp_connector_key(tool):
        return _clone_tool(
            tool,
            name=WHATSAPP_CANONICAL_TOOL_NAME,
            archetype=getattr(tool, "archetype", "") or "messaging_gateway",
            integration_kind=getattr(tool, "integration_kind", "") or "webhook_plus_rest_api",
            connector_key=WHATSAPP_CANONICAL_CONNECTOR_KEY,
            registered_api_ref=WHATSAPP_CANONICAL_CONNECTOR_KEY,
            has_side_effects=True,
            tool_type=getattr(tool, "tool_type", "") or "external",
            contract_review_state=getattr(tool, "contract_review_state", "") or "connector-detected",
        )

    google_key = resolve_google_workspace_connector_key(tool)
    if google_key:
        return _clone_tool(
            tool,
            name=google_key,
            connector_key=google_key,
            registered_api_ref=google_key,
            has_side_effects=bool(getattr(tool, "has_side_effects", False) or google_key in SIDE_EFFECT_CONNECTOR_KEYS),
            tool_type=getattr(tool, "tool_type", "") or "external",
            contract_review_state=getattr(tool, "contract_review_state", "") or "connector-detected",
        )

    odoo_key = resolve_odoo_connector_key(tool)
    if odoo_key:
        return _clone_tool(
            tool,
            name=odoo_key,
            connector_key=odoo_key,
            registered_api_ref=odoo_key,
            integration_kind=getattr(tool, "integration_kind", "") or "versioned_rpc_api",
            has_side_effects=bool(getattr(tool, "has_side_effects", False) or odoo_key in SIDE_EFFECT_CONNECTOR_KEYS),
            requires_approval=bool(getattr(tool, "requires_approval", False) or odoo_key in SIDE_EFFECT_CONNECTOR_KEYS),
            tool_type=getattr(tool, "tool_type", "") or "external",
            contract_review_state=getattr(tool, "contract_review_state", "") or "connector-detected",
        )

    return tool


def project_tools_for_construction(tools: list[Any]) -> list[Any]:
    projected: list[Any] = []
    seen_family_connectors: set[str] = set()
    for tool in tools:
        item = project_tool_for_construction(tool)
        family_key = (
            resolve_whatsapp_connector_key(item)
            or resolve_google_workspace_connector_key(item)
            or resolve_odoo_connector_key(item)
        )
        if family_key:
            if family_key in seen_family_connectors:
                continue
            seen_family_connectors.add(family_key)
        projected.append(item)
    return projected


def project_blueprint_tools_for_construction(snapshot: Any) -> list[Any]:
    blueprint = getattr(snapshot, "blueprint", None)
    if blueprint is None:
        return []
    return project_tools_for_construction(list(getattr(blueprint, "tools", []) or []))


def _fold_search_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def snapshot_has_odoo_quote_signal(snapshot: Any) -> bool:
    try:
        payload = snapshot.model_dump(mode="json")
    except Exception:
        payload = {}
    search_text = _fold_search_text(json.dumps(payload, ensure_ascii=False, default=str))
    return any(term in search_text for term in ODOO_QUOTE_SIGNAL_TERMS)


def odoo_connector_keys_for_snapshot(snapshot: Any) -> set[str]:
    keys = {
        key
        for key in (resolve_odoo_connector_key(tool) for tool in project_blueprint_tools_for_construction(snapshot))
        if key
    }
    if keys:
        keys.update({"odoo_partner_read", "odoo_crm_lead_read"})
    if keys and snapshot_has_odoo_quote_signal(snapshot):
        keys.update({"odoo_sale_order_read", "odoo_sale_quote_create"})
    return keys
