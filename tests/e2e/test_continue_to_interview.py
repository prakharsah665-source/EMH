"""
Continue-to-Interview E2E test.

Root cause of the previous failures: the bespoke launch here
only matched "Start interview" / "Begin interview" (never
"Continue interview"), did not dismiss the react-joyride tour
overlay, did not survive the "Interview Already Open" lock, and
ran against a stale interview session. That broke this test and
cascaded into the downstream LiveKit/audio/recording/system-
configuration failures.

Fix: reuse the proven launch flow from the bot-responsiveness
test via the shared page object (pages.interview_launch), which
accepts Start OR Continue, handles the tour and the lock, walks
system configuration + speaker test + recording consent, clicks
"Continue to Interview", and enforces a FRESH interview session.
This test then verifies the browser permissions and that the
real interview room actually loaded.
"""

import pytest
from playwright.async_api import async_playwright

from pages.interview_launch import launch_into_interview_room
from tests.e2e.session_policy import (
    mark_room_joined,
    resolve_room_session,
)


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
        print(body[:10000])
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

@pytest.mark.asyncio
async def test_continue_to_interview():
    # Fresh, non-stale, non-expired AND unconsumed session:
    # this test JOINS the interview room, so it skips (or uses
    # EMH_ROOM_TESTS_URL) when the primary session's room was
    # already joined - see tests/e2e/session_policy.py.
    room_url, room_claims = resolve_room_session(
        "test_continue_to_interview"
    )

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        context = await browser.new_context(
            permissions=["camera", "microphone"],
        )
        page = await context.new_page()

        try:
            print()
            print("=" * 70)
            print("CONTINUE TO INTERVIEW TEST")
            print("=" * 70)

            # Shared launch: Start OR Continue, joyride, lock,
            # system config, speaker test, consent, Continue.
            await launch_into_interview_room(
                page, log=print, interview_url=room_url
            )
            mark_room_joined(
                room_claims, "test_continue_to_interview"
            )

            await verify_permissions(page)
            await verify_interview_room(page)

            await page.screenshot(
                path="continue_to_interview.png", full_page=True
            )
            print("Screenshot saved: continue_to_interview.png")

            print()
            print("=" * 70)
            print("CONTINUE TO INTERVIEW TEST PASSED")
            print("=" * 70)

        except Exception:
            await print_page_state(
                page, "CONTINUE TO INTERVIEW TEST FAILED"
            )
            try:
                await page.screenshot(
                    path="continue_to_interview_failed.png",
                    full_page=True,
                )
                print(
                    "Failure screenshot saved: "
                    "continue_to_interview_failed.png"
                )
            except Exception:
                pass
            raise

        finally:
            await context.close()
            await browser.close()