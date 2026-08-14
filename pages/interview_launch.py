"""
Shared interview launch flow (page object).

Ported from the working launch logic in
tests/e2e/test_bot_responsiveness.py so every E2E test drives
the interview the same way:

  * accepts BOTH "Start interview" and "Continue interview",
  * survives the "Interview Already Open" one-session lock by
    reloading (bounded),
  * dismisses the react-joyride guided-tour overlay that
    intercepts clicks on a fresh session,
  * handles system configuration, speaker test, recording
    consent and "Continue to Interview",
  * enforces a FRESH (non-stale, non-expired) interview session
    up front, so downstream LiveKit/audio/recording/config
    failures caused by a dead room fail loudly at the source.

The candidate microphone energy step is optional: it only runs
when the page has the injected fake mic (window.__emhSpeak), so
tests that do not inject audio can reuse this helper unchanged.
"""

import re
from typing import Awaitable, Callable

from config.interview_session import require_fresh_interview_url


LAUNCH_BUTTON_RE = re.compile(
    r"^(start|continue) interview$", re.IGNORECASE
)


def _noop(_message: str) -> None:
    pass


async def _dismiss_guided_tour(page, log) -> None:
    for _ in range(12):
        tour_active = await page.evaluate(
            """
            () => {
                const portal = document.getElementById(
                    'react-joyride-portal'
                );
                return !!(portal && portal.childElementCount > 0);
            }
            """
        )
        if not tour_active:
            return
        dismissed = False
        for selector in (
            '[data-test-id="button-skip"]',
            '[data-test-id="button-close"]',
            '[aria-label="Skip"]',
            '[aria-label="Close"]',
        ):
            candidate = page.locator(selector)
            if await candidate.count() > 0:
                try:
                    await candidate.first.click(timeout=2_000)
                    dismissed = True
                    break
                except Exception:
                    pass
        if not dismissed:
            button = page.get_by_role(
                "button",
                name=re.compile(
                    r"^(skip( tour)?|close|got it|done|finish|last"
                    r"|next)",
                    re.IGNORECASE,
                ),
            )
            if await button.count() > 0:
                try:
                    await button.first.click(timeout=2_000)
                    dismissed = True
                except Exception:
                    pass
        if not dismissed:
            await page.evaluate(
                """
                () => {
                    const portal = document.getElementById(
                        'react-joyride-portal'
                    );
                    if (portal) portal.remove();
                }
                """
            )
        await page.wait_for_timeout(500)
    log(
        "[WARNING] Guided-tour overlay still present after 12 "
        "dismiss attempts."
    )


async def _feed_mic_check(page, log) -> None:
    """
    If the page injected a fake mic (window.__emhSpeak), play a
    short clip so the setup-screen mic check sees real energy.
    Silently skipped for tests without audio injection.
    """

    has_speak = await page.evaluate(
        "typeof window.__emhSpeak === 'function'"
    )
    if not has_speak:
        return

    # Imported lazily so non-audio tests need no audio fixtures.
    # Best-effort: a fixture/import problem must not abort launch.
    try:
        from tests.e2e.test_bot_responsiveness import (
            ensure_answer_fixtures,
            fixture_base64,
        )

        clip = ensure_answer_fixtures(1)[0]
        await page.evaluate(
            "b => window.__emhSpeak(b)", fixture_base64(clip)
        )
        log("Microphone check clip played (setup screen only).")
    except Exception as error:  # fixtures unavailable / synth failed
        log(f"[WARNING] Skipped fake-mic check clip: {error}")


async def launch_into_interview_room(
    page,
    *,
    log: Callable[[str], None] | None = None,
    interview_url: str | None = None,
    enforce_fresh: bool = True,
) -> str:
    """
    Drive a page from the interview URL into the live interview
    room. Returns the resolved interview URL. Raises
    InterviewSessionError (via require_fresh_interview_url) when
    the session is missing/stale/expired and enforce_fresh.
    """

    log = log or _noop

    if interview_url is None:
        if enforce_fresh:
            interview_url, _claims = require_fresh_interview_url()
        else:
            from config.interview_session import get_interview_url

            interview_url = get_interview_url()

    await page.goto(
        interview_url, wait_until="domcontentloaded", timeout=30_000
    )
    log("Interview page loaded (URL withheld - it contains the JWT).")

    # One-session lock: reload until the launch button appears.
    lock_screen = page.get_by_text("Interview Already Open", exact=False)
    for _ in range(12):
        await page.wait_for_function(
            """
            () => {
                const bodyText =
                    document.body ? document.body.innerText : "";
                if (bodyText.includes("Interview Already Open")) {
                    return true;
                }
                return Array.from(
                    document.querySelectorAll("button")
                ).some(button =>
                    /^(start|continue) interview$/i.test(
                        button.innerText.trim()
                    )
                );
            }
            """,
            timeout=30_000,
        )
        if await lock_screen.count() == 0:
            break
        log("Interview locked by a previous session - reloading...")
        await page.wait_for_timeout(5_000)
        await page.reload(wait_until="domcontentloaded")

    # Accept BOTH "Start interview" and "Continue interview".
    launch_button = page.get_by_role(
        "button", name=LAUNCH_BUTTON_RE
    ).first
    await launch_button.wait_for(state="visible", timeout=30_000)
    log(f"Launch button: {await launch_button.inner_text()}")
    await launch_button.click()

    # System Configuration screen.
    await page.get_by_text(
        "System Configuration", exact=True
    ).wait_for(state="visible", timeout=15_000)
    log("System Configuration loaded.")

    await _dismiss_guided_tour(page, log)
    log("Guided-tour overlay handled (if present).")

    await _feed_mic_check(page, log)

    # Speaker test (optional button).
    speaker_test = page.get_by_role(
        "button", name="Test Speaker Before Interview"
    )
    if await speaker_test.count() > 0:
        await speaker_test.first.wait_for(state="visible", timeout=15_000)
        await speaker_test.first.click()
        await page.wait_for_timeout(3_000)
        log("Speaker test completed.")

    # Recording consent (sr-only checkbox - DOM click).
    consent = page.locator('input[type="checkbox"]').last
    await consent.wait_for(state="attached", timeout=15_000)
    if not await consent.is_checked():
        await consent.evaluate("element => element.click()")
        await page.wait_for_function(
            """
            () => {
                const boxes = document.querySelectorAll(
                    'input[type="checkbox"]'
                );
                const box = boxes[boxes.length - 1];
                return box && box.checked === true;
            }
            """,
            timeout=5_000,
        )
    log("Recording consent accepted.")

    # Continue to Interview.
    continue_button = page.get_by_role(
        "button", name="Continue to Interview"
    )
    await continue_button.wait_for(state="visible", timeout=15_000)
    await page.wait_for_function(
        """
        () => {
            const button = Array.from(
                document.querySelectorAll("button")
            ).find(el =>
                el.innerText.trim() === "Continue to Interview"
            );
            return button && !button.disabled;
        }
        """,
        timeout=15_000,
    )
    await continue_button.click()
    log("Entered interview room.")

    return interview_url