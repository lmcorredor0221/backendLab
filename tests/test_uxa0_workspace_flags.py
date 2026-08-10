from app.services.workspace_bootstrap import (
    DEFAULT_FEATURE_FLAGS,
    FEATURE_FLAG_ATTENTION_V2,
    FEATURE_FLAG_PRODUCT_EXPERIENCE_V2,
)


def _flag_payload(flag_key: str) -> dict[str, object]:
    matches = [item for item in DEFAULT_FEATURE_FLAGS if item["key"] == flag_key]
    assert len(matches) == 1
    return matches[0]


def test_uxa0_adds_full_experience_and_attention_contract_flags_disabled_by_default() -> None:
    product_experience = _flag_payload(FEATURE_FLAG_PRODUCT_EXPERIENCE_V2)
    attention_v2 = _flag_payload(FEATURE_FLAG_ATTENTION_V2)

    assert product_experience["enabled"] is False
    assert product_experience["stage_hint"] == "workspace"
    assert "completa" in str(product_experience["description"]).lower()

    assert attention_v2["enabled"] is False
    assert attention_v2["stage_hint"] == "workspace"
