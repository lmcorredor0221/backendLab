from __future__ import annotations

import json

from scripts.run_llm_finops_release_gate import main, run_llm_finops_release_gate


def test_llm_finops_release_gate_passes_all_checks() -> None:
    summary = run_llm_finops_release_gate()
    checks = {item["name"]: item for item in summary["checks"]}

    assert summary["ok"] is True
    assert set(checks) == {
        "api_summary",
        "ledger",
        "migrations",
        "normalization",
        "pricing",
        "prompt_response_storage",
    }
    assert all(item["ok"] for item in checks.values())
    assert checks["prompt_response_storage"]["evidence"]["stored_hashes"] is True


def test_llm_finops_release_gate_returns_nonzero_with_clear_failure(capsys) -> None:
    exit_code = main(["--force-fail", "ledger"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    failed = [item for item in payload["checks"] if not item["ok"]]

    assert exit_code == 1
    assert payload["ok"] is False
    assert len(failed) == 1
    assert failed[0]["name"] == "ledger"
    assert "Forced failure" in failed[0]["detail"]
