from __future__ import annotations

from sqlmodel import Session

from app.db import engine
from app.services.marketing_analytics_service import process_pending_marketing_events


def main() -> None:
    with Session(engine) as db:
        summary = process_pending_marketing_events(db)
    print(summary)


if __name__ == "__main__":
    main()
