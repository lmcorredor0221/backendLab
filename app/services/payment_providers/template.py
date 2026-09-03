from __future__ import annotations

from dataclasses import dataclass, field

from sqlmodel import Session

from app.models import CommercialOrderRecord, CommercialOrderStatus
from app.services.payment_providers.base import CheckoutProviderContext, CheckoutProviderDraft


@dataclass(frozen=True)
class CheckoutProviderFinalizeResult:
    checkout_url: str = ""
    status: CommercialOrderStatus | None = None
    provider_checkout_id: str = ""
    provider_payment_link_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def noop(cls) -> "CheckoutProviderFinalizeResult":
        return cls()


class TemplateCommercePaymentProvider:
    provider_key = ""
    display_name = ""

    def build_checkout_seed(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        raise NotImplementedError

    def create_checkout_draft(self, context: CheckoutProviderContext) -> CheckoutProviderDraft:
        return self.build_checkout_seed(context)

    def finalize_checkout(
        self,
        session: Session,
        *,
        order: CommercialOrderRecord,
        context: CheckoutProviderContext,
    ) -> CheckoutProviderFinalizeResult:
        return CheckoutProviderFinalizeResult.noop()

    def build_next_action(self, order: CommercialOrderRecord) -> str:
        if order.status == CommercialOrderStatus.paid:
            return "refresh_access"
        if order.status != CommercialOrderStatus.pending:
            return "open_checkout"
        if not order.checkout_url and bool(order.metadata_payload.get("requires_payment_link")):
            return "await_payment_link"
        return "open_checkout"
