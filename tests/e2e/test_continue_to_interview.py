"""
Continue-to-Interview E2E test.

Runs against the run's SINGLE shared EMH_TESTS_URL room join
(tests/e2e/shared_room.py): the session budget is one joinable
room per run (INTERVIEW_URL is reserved for the full-interview
evaluation and a room join consumes the session), so this test,
test_interview_room and test_livekit_connection share one join
instead of each consuming/skipping.

As the first room-join consumer in the suite order, this test
normally PERFORMS the shared join - the proven launch flow from
pages.interview_launch (Start OR Continue, joyride prevention,
one-tab lock, system configuration, speaker test, recording
consent, "Continue to Interview") with per-step post-conditions,
on a fresh unconsumed session enforced by the unchanged session
policy. Its own responsibility is then to verify the browser
permissions and that the real interview room actually loaded.
"""

import pytest

from evaluation.redaction import redact_pii
from tests.e2e.screenshots import save_screenshot


# ============================================================
# Diagnostics
# ============================================================

async def print_page_state(page, title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("URL:", page.url)
    try:
        body = await page.locator("body").inner_text()
        print()
        print("PAGE TEXT:")
        print(redact_pii(body[:10000]))
    except Exception as error:
        print("Could not read page text:", error)
    print("=" * 70)


# ============================================================
# Assertions
# ============================================================

async def verify_permissions(page):
    print("Checking camera and microphone permissions...")

    camera_permission = await page.evaluate(
        """
        async () => (
            await navigator.permissions.query({ name: "camera" })
        ).state
        """
    )
    microphone_permission = await page.evaluate(
        """
        async () => (
            await navigator.permissions.query({ name: "microphone" })
        ).state
        """
    )

    print("Camera permission:", camera_permission)
    print("Microphone permission:", microphone_permission)

    assert camera_permission == "granted", (
        f"Camera permission is {camera_permission}"
    )
    assert microphone_permission == "granted", (
        f"Microphone permission is {microphone_permission}"
    )
    print("Camera and microphone permissions are GRANTED.")


async def verify_interview_room(page):
    print()
    print("Verifying actual interview room...")
    await page.wait_for_timeout(2000)

    body = (await page.locator("body").inner_text()).lower()
    print("Current URL:", page.url)

    strong_indicators = ["end interview", "leave interview", "livekit"]
    strong_matches = [i for i in strong_indicators if i in body]
    if strong_matches:
        print("[PASS] Strong interview-room indicators:", strong_matches)
        return

    system_config_visible = "system configuration" in body
    if "interview" in page.url.lower() and not system_config_visible:
        print(
            "[PASS] Interview page loaded and System Configuration "
            "is gone."
        )
        return

    await print_page_state(page, "INTERVIEW ROOM NOT VERIFIED")
    pytest.fail("Could not verify that the actual interview room loaded.")


# ============================================================
# Main Test
# ============================================================

@pytest.mark.asyncio(loop_scope="session")
async def test_continue_to_interview(shared_room):
    print()
    print("=" * 70)
    print("CONTINUE TO INTERVIEW TEST")
    print("=" * 70)

    # One shared room join per run: the first consumer joins
    # through the unchanged session policy (fresh, unconsumed
    # EMH_TESTS_URL; the join is ledgered), later consumers
    # attach to the same live page - tests/e2e/shared_room.py.
    page = await shared_room.ensure_joined("test_continue_to_interview")

    try:
        await verify_permissions(page)
        await verify_interview_room(page)

        await save_screenshot(page, "continue_to_interview")

        print()
        print("=" * 70)
        print("CONTINUE TO INTERVIEW TEST PASSED")
        print("=" * 70)

    except Exception:
        await print_page_state(
            page, "CONTINUE TO INTERVIEW TEST FAILED"
        )
        try:
            await save_screenshot(
                page, "continue_to_interview_failed"
            )
        except Exception:
            pass
        raise
