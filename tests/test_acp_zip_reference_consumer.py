from __future__ import annotations

import sys
from pathlib import Path

from app.services.acp_export_profiles import apply_acp_export_profile
from app.services.acp_generator import generate_acp_preview
from app.services.acp_zip_export import build_acp_zip
from tests.test_acp_generator import build_ready_snapshot


REFERENCE_CONSUMER_PATH = Path(__file__).resolve().parents[2] / "shared_specs" / "reference_consumers" / "python"
sys.path.insert(0, str(REFERENCE_CONSUMER_PATH))

from acp_zip_reference_consumer import validate_acp_zip  # noqa: E402


def test_external_zip_consumer_validates_all_commercial_profiles(tmp_path) -> None:
    preview = generate_acp_preview(build_ready_snapshot())

    for profile in ("blueprint-professional", "acp-portable", "acp-full"):
        profiled_preview = apply_acp_export_profile(preview, profile)
        zip_path = tmp_path / f"{profile}.zip"
        zip_path.write_bytes(build_acp_zip(profiled_preview))

        assert validate_acp_zip(zip_path, profile) == []


def test_external_zip_consumer_keeps_legacy_aliases_compatible(tmp_path) -> None:
    preview = generate_acp_preview(build_ready_snapshot())

    for alias in ("design-only", "extended"):
        profiled_preview = apply_acp_export_profile(preview, alias)
        zip_path = tmp_path / f"{alias}.zip"
        zip_path.write_bytes(build_acp_zip(profiled_preview))

        assert validate_acp_zip(zip_path, alias) == []
