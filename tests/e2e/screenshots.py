"""
Timestamped screenshot helper for the e2e tests.

Fixed-name screenshots in the repo root overwrote the previous
run's evidence and were committable; every screenshot now goes
to artifacts/screenshots/<name>_<timestamp>.png (gitignored via
artifacts/), so each run's evidence survives.
"""

from datetime import datetime
from pathlib import Path


SCREENSHOT_DIR = Path("artifacts/screenshots")


async def save_screenshot(page, name: str) -> Path | None:
    """
    Save a full-page screenshot under artifacts/screenshots
    with a timestamp. Never raises: a screenshot failure must
    not mask the original test failure.
    """

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = (
        SCREENSHOT_DIR
        / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.png"
    )

    try:
        await page.screenshot(path=str(path), full_page=True)
    except Exception as error:
        print(f"Screenshot failed ({name}): {error}")
        return None

    print(f"Screenshot saved: {path}")
    return path
