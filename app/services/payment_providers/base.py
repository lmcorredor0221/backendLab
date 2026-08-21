from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from app.models import (
    CommercialOrderStatus,
    ProductCatalogRecord,
    ProductPriceRecord,
    SessionRecord,
    UserRecord,
)


@dataclass(frozen=True)
class CheckoutProviderContext:
    workspace_id: UUID
    session_record: SessionRecord
    current_user: UserRecord
    product: ProductCatalogRecord
    price: ProductPriceRecord
    subtotal_cents: int
    discount_cents: int
    total_cents: int
    currency: str
    is_upgrade: bool
    idempotency_key: str
    success_url: str = ""
    cancel_url: str = ""
    base_url: str = ""


@dataclass(frozen=True)
class CheckoutProviderDraft:
    provider: str
    checkout_ref: str
    checkout_url: str
    status: CommercialOrderStatus = CommercialOrderStatus.pending
    metadata: dict[str, object] = field(default_factory=dict)


class CommercePaymentProvider(Protocol):
    provider_key: str

    def create_checkout_draft(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        """Build the provider-specific checkout draft without committing an order."""

