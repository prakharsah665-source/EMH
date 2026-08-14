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

# The app gates the React Joyride guided tour on this
# localStorage flag (verified live 2026-08-14): when "true"
# the tour never renders, so it cannot intercept clicks
# (Speaker Test, consent). Pre-seeding it is tour PREVENTION,
# not the bypass of a real UI blocker - the tour is first-run
# onboarding, not a functional gate.
TOUR_COMPLETE_KEY = "interviewRoomTourComplete"

TOUR_SUPPRESS_JS = (
    "try { localStorage.setItem("
    f"'{TOUR_COMPLETE_KEY}', 'true'"
    ") } catch (e) {}"
)


class InterviewLaunchError(RuntimeError):
    """
    A launch STEP failed its post-condition. Carries the exact
    broken stage plus the failure-taxonomy layer so downstream
    reporting never mislabels a launch problem as a bot
    failure.
    """

    def __init__(self, step: str, layer: str, detail: str):
        self.step = step
        self.layer = layer
        super().__init__(
            f"LAUNCH STEP FAILED: {step}\n"
            f"Layer: {layer} (NOT a bot failure)\n"
            f"{detail}"
        )


def _noop(_message: str) -> None:
    pass


async def _page_hint(page) -> str:
    """Short redacted page summary for step-failure messages."""

    try:
        from evaluation.redaction import redact_pii

        body = await page.locator("body").inner_text()
        return (
            f"URL: {page.url.split('?')[0].rsplit('/', 1)[0]}/…\n"
            "Page text (first 500 chars, redacted): "
            + redact_pii(" ".join(body.split())[:500])
        )
    except Exception as error:
        return f"(page state unreadable: {error})"


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

    # Prevent the guided tour BEFORE any app code can render
    # it: on this page load (evaluate) and on any later
    # navigation (init script).
    try:
        await page.add_init_script(TOUR_SUPPRESS_JS)
    except Exception:
        pass  # already-created pages on some drivers

    try:
        await page.goto(
            interview_url,
            wait_until="domcontentloaded",
            timeout=30_000,
        )
    except Exception as error:
        raise InterviewLaunchError(
            "open interview URL",
            "environment/network",
            f"Navigation failed: {error}",
        ) from error

    await page.evaluate(TOUR_SUPPRESS_JS)
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
    try:
        await launch_button.wait_for(state="visible", timeout=30_000)
    except Exception as error:
        raise InterviewLaunchError(
            "find Start/Continue launch button",
            "session/environment",
            "No launch button within 30s - the session may be "
            "expired, locked or already finished.\n"
            + await _page_hint(page),
        ) from error
    log(f"Launch button: {await launch_button.inner_text()}")
    await launch_button.click()

    # Post-condition: the launch click must reach the System
    # Configuration screen.
    try:
        await page.get_by_text(
            "System Configuration", exact=True
        ).wait_for(state="visible", timeout=15_000)
    except Exception as error:
        raise InterviewLaunchError(
            "reach System Configuration after launch click",
            "harness/app-launch",
            "Clicked the launch button but the System "
            "Configuration screen never rendered.\n"
            + await _page_hint(page),
        ) from error
    log("System Configuration loaded.")

    await _dismiss_guided_tour(page, log)

    # Post-condition: the tour must actually be gone - a live
    # overlay intercepts every later click (Speaker Test,
    # consent) and would surface as fake step failures.
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
    if tour_active:
        raise InterviewLaunchError(
            "dismiss guided-tour overlay",
            "harness/ui-overlay",
            "The React Joyride overlay is still rendered after "
            "prevention (localStorage "
            f"{TOUR_COMPLETE_KEY}=true) and 12 dismissal "
            "attempts - later clicks would be intercepted.",
        )
    log("Guided tour prevented/dismissed (post-condition met).")

    await _feed_mic_check(page, log)

    # Speaker test with a post-condition. The button must be
    # WAITED for, not sampled once - it renders after the mic
    # section settles, and silently skipping it leaves the
    # speaker check "required", which blocks the room
    # transition later (found live: the enabled state of
    # 'Continue to Interview' does NOT imply the speaker test
    # passed, so it is not a completion signal).
    _SPEAKER_DONE_JS = """
        () => {
            const body = (
                document.body.innerText || ""
            ).toLowerCase();
            return body.includes("all tests completed")
                || (!body.includes("speaker test required")
                    && !body.includes("pending"));
        }
    """

    speaker_test = page.get_by_role(
        "button", name="Test Speaker Before Interview"
    ).first
    speaker_button_present = True
    try:
        await speaker_test.wait_for(
            state="visible", timeout=15_000
        )
    except Exception:
        speaker_button_present = False

    if speaker_button_present:
        await speaker_test.click()
        try:
            await page.wait_for_function(
                _SPEAKER_DONE_JS, timeout=20_000
            )
        except Exception as error:
            raise InterviewLaunchError(
                "complete speaker test",
                "audio/environment",
                "Clicked 'Test Speaker Before Interview' but "
                "the speaker check never left the required/"
                "pending state within 20s (speaker playback "
                "may be silent in this environment).\n"
                + await _page_hint(page),
            ) from error
        log("Speaker test completed (post-condition met).")
    else:
        # No button - acceptable ONLY if the speaker check is
        # already satisfied.
        already_done = await page.evaluate(_SPEAKER_DONE_JS)
        if not already_done:
            raise InterviewLaunchError(
                "find speaker test button",
                "harness/app-launch",
                "The speaker check is still required but the "
                "'Test Speaker Before Interview' button never "
                "appeared within 15s.\n" + await _page_hint(page),
            )
        log("Speaker check already satisfied - no test needed.")

    # Recording consent (sr-only checkbox - DOM click) with a
    # checked post-condition.
    try:
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
    except Exception as error:
        raise InterviewLaunchError(
            "accept recording consent",
            "harness/app-launch",
            "The consent checkbox never reached checked state.\n"
            + await _page_hint(page),
        ) from error
    log("Recording consent accepted (post-condition met).")

    # Continue to Interview.
    try:
        continue_button = page.get_by_role(
            "button", name="Continue to Interview"
        )
        await continue_button.wait_for(
            state="visible", timeout=15_000
        )
        await page.wait_for_function(
            """
            () => {
                const button = Array.from(
                    document.querySelectorAll("button")
                ).find(el =>
                    el.innerText.trim() ===
                        "Continue to Interview"
                );
                return button && !button.disabled;
            }
            """,
            timeout=15_000,
        )
    except Exception as error:
        raise InterviewLaunchError(
            "enable Continue to Interview",
            "harness/app-launch",
            "The 'Continue to Interview' button never became "
            "enabled (a setup step above may not really have "
            "completed).\n" + await _page_hint(page),
        ) from error

    await continue_button.click()

    # Post-condition: the interview ROOM must actually render -
    # System Configuration gone and a room control present.
    # Only then may this helper claim the room was entered.
    try:
        await page.wait_for_function(
            """
            () => {
                const body = (
                    document.body.innerText || ""
                ).toLowerCase();
                if (body.includes("system configuration")) {
                    return false;
                }
                return ["end interview", "leave interview",
                        "mute", "recording", "elapsed time"]
                    .some(marker => body.includes(marker));
            }
            """,
            timeout=30_000,
        )
    except Exception as error:
        raise InterviewLaunchError(
            "enter interview room",
            "harness/app-launch",
            "Clicked 'Continue to Interview' but the interview "
            "room never rendered (no room controls appeared "
            "within 30s).\n" + await _page_hint(page),
        ) from error

    log("Entered interview room (post-condition met).")

    return interview_url