from app.services.acp_serialization import (
    build_content_hash,
    normalize_text_document,
    serialize_json_document,
    serialize_markdown_document,
    serialize_yaml_document,
)


def test_serialize_yaml_document_is_deterministic_and_preserves_order() -> None:
    payload = {
        "metadata": {
            "name": "Lean Agent Builder",
            "version": "1.0.0",
        },
        "business": {
            "objective": "Disenar un ACP util",
        },
    }

    result = serialize_yaml_document(payload)

    assert result == serialize_yaml_document(payload)
    assert result.splitlines()[0] == "metadata:"
    assert "business:" in result
    assert "\r" not in result
    assert result.endswith("\n")


def test_serialize_json_document_sorts_keys_and_normalizes_newline() -> None:
    payload = {"zeta": 1, "alpha": {"delta": 4, "beta": 2}}

    result = serialize_json_document(payload)

    assert result.startswith("{\n  \"alpha\"")
    assert "\"zeta\": 1" in result
    assert "\r" not in result
    assert result.endswith("\n")


def test_markdown_and_hash_normalization_ignore_line_endings() -> None:
    content = "# Titulo  \r\n\r\nLinea uno  \r\nLinea dos\r\n"

    normalized = serialize_markdown_document(content)

    assert normalized == "# Titulo\n\nLinea uno\nLinea dos\n"
    assert normalize_text_document(content) == normalized
    assert build_content_hash(content) == build_content_hash(normalized)
