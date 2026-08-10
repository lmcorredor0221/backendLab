from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from app.models import ACPPreview
from app.services.acp_serialization import normalize_text_document


ZIP_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def build_acp_zip(preview: ACPPreview) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for item in sorted(preview.files, key=lambda entry: entry.path):
            zip_info = ZipInfo(filename=item.path, date_time=ZIP_FIXED_TIMESTAMP)
            zip_info.compress_type = ZIP_DEFLATED
            archive.writestr(zip_info, normalize_text_document(item.content_text).encode("utf-8"))
    return buffer.getvalue()
