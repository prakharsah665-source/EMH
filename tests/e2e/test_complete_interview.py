import re
from pathlib import Path

import pytest
from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from evaluation.redaction import redact_pii
from tests.e2e.screenshots import save_screenshot

from config.interview_session import (
    InterviewSessionError,
    require_fresh_tests_url,
)


# ============================================================
# Configuration
# ============================================================

DEBUG_DIR = Path("artifacts/debug")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

async def debug_page(page: Page, label: str) -> None:
    """
    Print useful page information and capture a screenshot.
    """

    print()
    print("=" * 80)
    print(f"DEBUG: {label}")
    print("=" * 80)

    print(f"URL: {page.url}")

    try:
        print(f"Title: {await page.title()}")
    except Exception:
        print("Title: <unavailable>")

    print()
    print("BUTTONS:")

    buttons = page.get_by_role("button")
    count = await buttons.count()

    print(f"Button count: {count}")

    for index in range(count):
        button = buttons.nth(index)

        try:
            text = (await button.inner_text()).strip()
        except Exception:
            text = "<unable to read text>"

        try:
            visible = await button.is_visible()
        except Exception:
            visible = False

        try:
            enabled = await button.is_enabled()
        except Exception:
            enabled = False

        try:
            aria_label = await button.get_attribute("aria-label")
        except Exception:
            aria_label = None

        print(
            f"[{index}] "
            f"text={text!r} "
            f"aria-label={aria_label!r} "
            f"visible={visible} "
            f"enabled={enabled}"
        )

    print()
    print("VISIBLE TEXT PREVIEW:")

    try:
        body_text = await page.locator("body").inner_text()

        print(redact_pii(body_text[:5000]))

    except Exception as error:
        print(f"Could not read body text: {error}")

    # Timestamped so successive runs never overwrite evidence.
    await save_screenshot(
        page, label.lower().replace(" ", "_")
    )

    print("=" * 80)
    print()


async def find_continue_button(page: Page):
    """
    Find the interview entry button using several safe
    accessibility-based strategies.

    A fresh session shows "Start interview"; a previously
    started session shows "Continue interview". Both are
    valid entry points into the interview flow.

    Returns the locator or None.
    """

    candidates = [
        page.get_by_role(
            "button",
            name="Continue interview",
            exact=True,
        ),
        page.get_by_role(
            "button",
            name="Start interview",
            exact=True,
        ),
        page.get_by_role(
            "button",
            name="Continue",
            exact=True,
        ),
        page.get_by_role(
            "button",
            name="Continue interview",
        ),
        page.get_by_role(
            "button",
            name="Start interview",
        ),
    ]

    for candidate in candidates:

        try:
            if await candidate.count() > 0:

                for index in range(
                    await candidate.count()
                ):

                    button = candidate.nth(index)

                    if await button.is_visible():
                        return button

        except Exception:
            continue

    return None


async def wait_for_continue_button(
    page: Page,
    timeout: int = 15000,
):
    """
    Wait for the Continue button while producing useful
    diagnostics if it does not appear.
    """

    # Deterministic application-state wait: the button only
    # exists once the React app has finished rendering the
    # interview screen, so wait on the locator itself for the
    # full budget instead of checking a momentary snapshot.

    primary = page.get_by_role(
        "button",
        name=re.compile(
            r"^(Continue|Start) interview$",
            re.IGNORECASE,
        ),
    ).first

    try:

        await primary.wait_for(
            state="visible",
            timeout=timeout,
        )

        return primary

    except PlaywrightTimeoutError:

        # Try the alternative accessible names before
        # reporting diagnostics.

        button = await find_continue_button(page)

        if button is not None:
            return button

        await debug_page(
            page,
            "continue_button_timeout",
        )

        raise


# ============================================================
# Main E2E test
# ============================================================

@pytest.mark.asyncio
async def test_complete_interview():
    """
    Complete-interview smoke test.

    This version intentionally includes detailed diagnostics
    around the Continue Interview step.

    The browser is managed directly with async_playwright,
    matching every other E2E test in this repo (there is no
    pytest-playwright plugin providing a `page` fixture).
    """

    # Same primary session as the full-interview evaluation.
    try:
        interview_url, _claims = require_fresh_tests_url()
    except InterviewSessionError as error:
        pytest.fail(str(error))

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=False
        )

        context = await browser.new_context(
            permissions=[
                "microphone",
                "camera",
            ]
        )

        page = await context.new_page()

        try:

            print()
            print("=" * 80)
            print("EMH COMPLETE INTERVIEW TEST")
            print("=" * 80)

            # ------------------------------------------------
            # 1. Open interview
            # ------------------------------------------------

            print()
            print("Opening interview URL...")

            await page.goto(
                interview_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            print(
                f"Interview page loaded: {page.url}"
            )

            # ------------------------------------------------
            # 2. Initial page diagnostics
            # ------------------------------------------------

            await debug_page(
                page,
                "initial_page",
            )

            # ------------------------------------------------
            # 3. Wait for Continue Interview
            # ------------------------------------------------

            print(
                "Looking for Continue/Start Interview button..."
            )

            continue_button = await wait_for_continue_button(
                page,
                timeout=15000,
            )

            if continue_button is None:
                pytest.fail(
                    "Continue Interview button was not found."
                )

            print(
                "Continue Interview button found."
            )

            print(
                f"Button text: "
                f"{await continue_button.inner_text()!r}"
            )

            print(
                f"Button enabled: "
                f"{await continue_button.is_enabled()}"
            )

            # ------------------------------------------------
            # 4. Click Continue Interview
            # ------------------------------------------------

            if not await continue_button.is_enabled():
                await debug_page(
                    page,
                    "continue_button_disabled",
                )

                pytest.fail(
                    "Continue Interview button is visible "
                    "but disabled."
                )

            await continue_button.click()

            print(
                "Continue Interview clicked."
            )

            # ------------------------------------------------
            # 5. Wait for the next UI state
            # ------------------------------------------------

            await page.wait_for_load_state(
                "domcontentloaded"
            )

            await page.wait_for_timeout(1000)

            print()
            print(
                f"After Continue URL: {page.url}"
            )

            # ------------------------------------------------
            # 6. Capture post-continue state
            # ------------------------------------------------

            await debug_page(
                page,
                "after_continue",
            )

            print()
            print("=" * 80)
            print("COMPLETE INTERVIEW TEST PASSED")
            print("=" * 80)

        finally:

            await context.close()
            await browser.close()