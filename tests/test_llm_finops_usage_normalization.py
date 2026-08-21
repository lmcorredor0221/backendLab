from dataclasses import dataclass

from app.services.llm_finops.usage_normalization import (
    normalize_cli_usage,
    normalize_deepseek_usage,
    normalize_openai_usage,
)


@dataclass
class UsageDetails:
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


@dataclass
class OpenAIUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: UsageDetails
    output_tokens_details: UsageDetails


def test_normalize_openai_usage_preserves_core_and_special_tokens() -> None:
    usage = normalize_openai_usage(
        OpenAIUsage(
            input_tokens=200,
            output_tokens=80,
            total_tokens=280,
            input_tokens_details=UsageDetails(cached_tokens=50),
            output_tokens_details=UsageDetails(reasoning_tokens=25, accepted_prediction_tokens=3),
        )
    )

    assert usage.input_tokens == 200
    assert usage.output_tokens == 80
    assert usage.total_tokens == 280
    assert usage.cached_input_tokens == 50
    assert usage.reasoning_tokens == 25
    assert usage.accepted_prediction_tokens == 3
    assert usage.compatibility_token_usage()["prompt_tokens"] == 200


def test_normalize_deepseek_usage_supports_cache_hit_and_miss_tokens() -> None:
    usage = normalize_deepseek_usage(
        {
            "prompt_tokens": 150,
            "completion_tokens": 45,
            "prompt_cache_hit_tokens": 60,
            "prompt_cache_miss_tokens": 90,
        }
    )

    assert usage.input_tokens == 150
    assert usage.output_tokens == 45
    assert usage.total_tokens == 195
    assert usage.cached_input_tokens == 60
    assert usage.provider_metrics["cache_miss_tokens"] == 90


def test_normalize_cli_usage_marks_estimated_usage_and_keeps_runtime_metrics() -> None:
    usage = normalize_cli_usage(
        {
            "run_id": "run-123",
            "status": "succeeded",
            "selected_model": "gpt-test",
            "attempts": [{"attempt_number": 1}],
            "metrics": {
                "duration_ms": 1200,
                "queue_wait_ms": 25,
                "output_size_bytes": 300,
                "exit_code": 0,
            },
        },
        prompt_text="abcd" * 20,
        output_text="efgh" * 8,
    )

    assert usage.usage_is_estimated is True
    assert usage.input_tokens == 20
    assert usage.output_tokens == 8
    assert usage.total_tokens == 28
    assert usage.provider_metrics["duration_ms"] == 1200
    assert usage.provider_metrics["attempt_count"] == 1


def test_normalizers_fill_total_tokens_when_provider_omits_total() -> None:
    openai_usage = normalize_openai_usage({"input_tokens": 10, "output_tokens": 7})
    deepseek_usage = normalize_deepseek_usage({"prompt_tokens": 11, "completion_tokens": 9})

    assert openai_usage.total_tokens == 17
    assert deepseek_usage.total_tokens == 20
