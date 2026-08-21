from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.services.llm_finops.contracts import NormalizedLLMUsage


def empty_usage() -> NormalizedLLMUsage:
    return NormalizedLLMUsage()


def normalize_openai_usage(raw_usage: Any) -> NormalizedLLMUsage:
    payload = _usage_to_dict(raw_usage)
    input_tokens = _first_int(payload, "input_tokens", "prompt_tokens")
    output_tokens = _first_int(payload, "output_tokens", "completion_tokens")
    total_tokens = _first_int(payload, "total_tokens") or input_tokens + output_tokens
    cached_input_tokens = _first_int(
        payload,
        "cached_input_tokens",
        "input_cached_tokens",
        "prompt_cached_tokens",
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
        "cache_read_input_tokens",
    )
    reasoning_tokens = _first_int(
        payload,
        "reasoning_tokens",
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    )
    accepted_prediction_tokens = _first_int(
        payload,
        "accepted_prediction_tokens",
        ("output_tokens_details", "accepted_prediction_tokens"),
        ("completion_tokens_details", "accepted_prediction_tokens"),
    )
    rejected_prediction_tokens = _first_int(
        payload,
        "rejected_prediction_tokens",
        ("output_tokens_details", "rejected_prediction_tokens"),
        ("completion_tokens_details", "rejected_prediction_tokens"),
    )
    provider_metrics = _collect_known_provider_metrics(
        payload,
        extra={
            "cached_input_tokens": cached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "accepted_prediction_tokens": accepted_prediction_tokens,
            "rejected_prediction_tokens": rejected_prediction_tokens,
        },
    )
    return NormalizedLLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        accepted_prediction_tokens=accepted_prediction_tokens,
        rejected_prediction_tokens=rejected_prediction_tokens,
        tool_call_count=_first_int(payload, "tool_call_count", "tool_calls"),
        provider_metrics=provider_metrics,
        raw_usage=payload,
    )


def normalize_deepseek_usage(raw_usage: Any) -> NormalizedLLMUsage:
    payload = _usage_to_dict(raw_usage)
    input_tokens = _first_int(payload, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(payload, "completion_tokens", "output_tokens")
    total_tokens = _first_int(payload, "total_tokens") or input_tokens + output_tokens
    cache_hit_tokens = _first_int(
        payload,
        "prompt_cache_hit_tokens",
        "input_cache_hit_tokens",
        ("prompt_tokens_details", "cached_tokens"),
    )
    cache_miss_tokens = _first_int(
        payload,
        "prompt_cache_miss_tokens",
        "input_cache_miss_tokens",
    )
    reasoning_tokens = _first_int(
        payload,
        "reasoning_tokens",
        ("completion_tokens_details", "reasoning_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
    )
    provider_metrics = _collect_known_provider_metrics(
        payload,
        extra={
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": cache_miss_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
    )
    return NormalizedLLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cache_hit_tokens,
        reasoning_tokens=reasoning_tokens,
        tool_call_count=_first_int(payload, "tool_call_count", "tool_calls"),
        provider_metrics=provider_metrics,
        raw_usage=payload,
    )


def normalize_cli_usage(
    audit: Any,
    *,
    prompt_text: str | None = None,
    output_text: str | None = None,
) -> NormalizedLLMUsage:
    payload = _usage_to_dict(audit)
    input_tokens = _estimate_tokens(prompt_text or "")
    output_tokens = _estimate_tokens(output_text or "")
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    attempted_models = payload.get("attempted_models", [])
    attempts = payload.get("attempts", [])
    provider_metrics = {
        "run_id": payload.get("run_id", ""),
        "status": payload.get("status", ""),
        "selected_model": payload.get("selected_model", ""),
        "requested_model": payload.get("requested_model", ""),
        "attempted_models": attempted_models if isinstance(attempted_models, list) else [],
        "fallback_used": bool(payload.get("fallback_used", False)),
        "duration_ms": _first_int(payload, ("metrics", "duration_ms"), "duration_ms"),
        "queue_wait_ms": _first_int(payload, ("metrics", "queue_wait_ms"), "queue_wait_ms"),
        "output_size_bytes": _first_int(payload, ("metrics", "output_size_bytes"), "output_size_bytes"),
        "stdout_bytes": _first_int(payload, ("metrics", "stdout_bytes"), "stdout_bytes"),
        "stderr_bytes": _first_int(payload, ("metrics", "stderr_bytes"), "stderr_bytes"),
        "exit_code": _first_int(payload, "exit_code", "returncode", ("metrics", "exit_code")),
        "attempt_count": len(attempts) if isinstance(attempts, list) else _first_int(payload, "attempt_count", ("metadata", "attempt_count")),
    }
    if not provider_metrics["output_size_bytes"] and output_text:
        provider_metrics["output_size_bytes"] = len(output_text.encode("utf-8"))
    if not input_tokens:
        input_tokens = _first_int(metrics, "prompt_estimated_tokens")
    if not output_tokens:
        output_tokens = _first_int(metrics, "output_estimated_tokens")
    return NormalizedLLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        provider_metrics={key: value for key, value in provider_metrics.items() if _has_metric_value(value)},
        raw_usage=payload,
        usage_is_estimated=True,
    )


def _usage_to_dict(raw_usage: Any) -> dict[str, Any]:
    if raw_usage is None:
        return {}
    if isinstance(raw_usage, dict):
        return {str(key): _usage_to_dict(value) if _is_structured(value) else value for key, value in raw_usage.items()}
    if is_dataclass(raw_usage):
        return _usage_to_dict(asdict(raw_usage))
    model_dump = getattr(raw_usage, "model_dump", None)
    if callable(model_dump):
        try:
            return _usage_to_dict(model_dump(mode="json"))
        except TypeError:
            return _usage_to_dict(model_dump())
    if hasattr(raw_usage, "__dict__"):
        return {
            str(key): _usage_to_dict(value) if _is_structured(value) else value
            for key, value in vars(raw_usage).items()
            if not str(key).startswith("_")
        }
    return {}


def _is_structured(value: Any) -> bool:
    return isinstance(value, dict) or is_dataclass(value) or hasattr(value, "model_dump") or hasattr(value, "__dict__")


def _has_metric_value(value: Any) -> bool:
    return value is not None and value != ""


def _first_int(payload: dict[str, Any], *keys: str | tuple[str, ...]) -> int:
    for key in keys:
        value = _get_value(payload, key)
        if value in (None, ""):
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _get_value(payload: dict[str, Any], key: str | tuple[str, ...]) -> Any:
    if isinstance(key, tuple):
        current: Any = payload
        for segment in key:
            if not isinstance(current, dict):
                return None
            current = current.get(segment)
        return current
    return payload.get(key)


def _collect_known_provider_metrics(payload: dict[str, Any], *, extra: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        key: value
        for key, value in extra.items()
        if value not in (None, "", 0)
    }
    for key, value in _flatten_scalars(payload).items():
        if key in {"input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens"}:
            continue
        metrics.setdefault(key, value)
    return metrics


def _flatten_scalars(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_scalars(value, next_prefix))
        return flattened
    if isinstance(payload, list):
        return {}
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return {prefix or "value": payload}
    return {}


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
