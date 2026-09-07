from __future__ import annotations

import hashlib
import hmac
from decimal import Decimal, InvalidOperation
from typing import Mapping


def format_payu_amount_from_cents(amount_cents: int) -> str:
    amount = Decimal(max(0, amount_cents)) / Decimal(100)
    return f"{amount:.2f}"


def format_payu_confirmation_value(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "0.0"
    normalized = text.replace(",", ".")
    integer, dot, decimal = normalized.partition(".")
    if not dot:
        return f"{integer}.0"
    if len(decimal) > 1 and decimal[1] != "0":
        return f"{integer}.{decimal[:2]}"
    first_decimal = decimal[0] if decimal else "0"
    return f"{integer}.{first_decimal}"


def sign_payu_payment_form(
    *,
    api_key: str,
    merchant_id: str,
    reference_code: str,
    amount: str,
    currency: str,
    algorithm: str = "MD5",
    hmac_secret: str = "",
    payment_methods: str = "",
    iin: str = "",
    pse_banks: str = "",
) -> str:
    parts = [
        api_key.strip(),
        merchant_id.strip(),
        reference_code.strip(),
        amount.strip(),
        currency.strip().upper(),
    ]
    if payment_methods or iin or pse_banks:
        parts.extend([payment_methods.strip(), iin.strip(), pse_banks.strip()])
    return _digest("~".join(parts), algorithm=algorithm, hmac_secret=hmac_secret or api_key)


def verify_payu_confirmation_signature(
    payload: Mapping[str, object],
    *,
    api_key: str,
    hmac_secret: str = "",
) -> bool:
    received = _payload_value(payload, "sign", "signature")
    if not received or not api_key:
        return False
    merchant_id = _payload_value(payload, "merchant_id", "merchantId")
    reference_sale = _payload_value(payload, "reference_sale", "referenceCode")
    value = _payload_value(payload, "value", "TX_VALUE")
    currency = _payload_value(payload, "currency")
    state_pol = _payload_value(payload, "state_pol", "transactionState", "polTransactionState")
    if not all([merchant_id, reference_sale, value, currency, state_pol]):
        return False
    formatted_value = format_payu_confirmation_value(value)
    base = "~".join(
        [
            api_key.strip(),
            merchant_id.strip(),
            reference_sale.strip(),
            formatted_value,
            currency.strip().upper(),
            state_pol.strip(),
        ]
    )
    candidates = {
        _digest(base, algorithm="MD5"),
        _digest(base, algorithm="SHA1"),
        _digest(base, algorithm="SHA256"),
        _digest(base, algorithm="HMAC-SHA256", hmac_secret=api_key),
    }
    if hmac_secret:
        candidates.add(_digest(base, algorithm="HMAC-SHA256", hmac_secret=hmac_secret))
    normalized_received = received.strip().lower()
    return any(hmac.compare_digest(normalized_received, candidate.lower()) for candidate in candidates)


def amount_cents_from_payu_value(value: str, *, fallback_cents: int = 0) -> int:
    text = str(value or "").strip().replace(",", ".")
    try:
        amount = Decimal(text)
    except (InvalidOperation, ValueError):
        return max(0, fallback_cents)
    return max(0, int(amount * Decimal(100)))


def _payload_value(payload: Mapping[str, object], *keys: str) -> str:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _digest(value: str, *, algorithm: str, hmac_secret: str = "") -> str:
    normalized = algorithm.strip().upper().replace("_", "-")
    if normalized == "SHA":
        normalized = "SHA1"
    data = value.encode("utf-8")
    if normalized == "HMAC-SHA256":
        return hmac.new((hmac_secret or "").encode("utf-8"), data, hashlib.sha256).hexdigest()
    if normalized == "SHA1":
        return hashlib.sha1(data).hexdigest()
    if normalized == "SHA256":
        return hashlib.sha256(data).hexdigest()
    return hashlib.md5(data).hexdigest()
