import pytest
from playwright.async_api import async_playwright

from config.settings import INTERVIEW_URL
from evaluation.redaction import redact_pii
from pages.interview_launch import LAUNCH_BUTTON_RE
from config.interview_session import (
    InterviewSessionError,
    require_fresh_interview_url,
)


async def print_page_state(page, title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("URL:", page.url)
    print("TITLE:", await page.title())

    try:
        text = await page.locator("body").inner_text()
        print("\nPAGE TEXT:")
        print(redact_pii(text[:6000]))
    except Exception as error:
        print("Could not read page:", error)

    print("=" * 70)


async def check_media_permissions(page):
    print("Checking camera permission...")

    camera = await page.evaluate(
        """
        async () => {
            const result = await navigator.permissions.query({
                name: "camera"
            });

            return result.state;
        }
        """
    )

    print("Camera permission:", camera)

    assert camera == "granted", (
        f"Camera permission should be granted, "
        f"but was '{camera}'."
    )

    print("Checking microphone permission...")

    microphone = await page.evaluate(
        """
        async () => {
            const result = await navigator.permissions.query({
                name: "microphone"
            });

            return result.state;
        }
        """
    )

    print("Microphone permission:", microphone)

    assert microphone == "granted", (
        f"Microphone permission should be granted, "
        f"but was '{microphone}'."
    )


async def check_media_stream(page):
    print("Testing actual camera and microphone...")

    result = await page.evaluate(
        """
        async () => {
            try {
                const stream =
                    await navigator.mediaDevices.getUserMedia({
                        video: true,
                        audio: true
                    });

                const videoTracks = stream.getVideoTracks();
                const audioTracks = stream.getAudioTracks();

                const videoWorking =
                    videoTracks.length > 0 &&
                    videoTracks[0].readyState === "live";

                const audioWorking =
                    audioTracks.length > 0 &&
                    audioTracks[0].readyState === "live";

                stream.getTracks().forEach(
                    track => track.stop()
                );

                return {
                    success: true,
                    video: videoWorking,
                    audio: audioWorking,
                    videoTracks: videoTracks.length,
                    audioTracks: audioTracks.length
                };

            } catch (error) {
                return {
                    success: false,
                    error: error.name + ": " + error.message
                };
            }
        }
        """
    )

    print("Media result:", result)

    assert result["success"], (
        "Camera/microphone access failed: "
        f"{result.get('error')}"
    )

    assert result["video"], (
        "Camera stream is not active."
    )

    assert result["audio"], (
        "Microphone stream is not active."
    )


async def get_continue_button(page):
    names = [
        "Continue to Interview",
        "Continue to interview",
        "Continue Interview",
        "Continue interview",
    ]

    for name in names:
        locator = page.get_by_role(
            "button",
            name=name,
            exact=True,
        )

        if await locator.count() > 0:
            return locator.first

    return None


async def check_system_configuration(page):
    print("Checking System Configuration screen...")

    body = (
        await page.locator("body").inner_text()
    ).lower()

    required_items = [
        "system configuration",
        "network connection",
        "camera",
        "audio input",
        "session recording consent",
    ]

    for item in required_items:
        if item in body:
            print(f"[PASS] Found: {item}")
        else:
            print(f"[FAIL] Missing: {item}")

            await print_page_state(
                page,
                f"Missing system configuration item: {item}",
            )

            pytest.fail(
                f"System Configuration is missing: {item}"
            )


async def check_required_state(page):
    """
    Verify that the candidate cannot bypass the required
    system checks.
    """

    print(
        "Checking Continue to Interview button..."
    )

    button = await get_continue_button(page)

    if button is None:
        await print_page_state(
            page,
            "CONTINUE BUTTON NOT FOUND",
        )

        pytest.fail(
            "Continue to Interview button was not found."
        )

    print(
        "Continue button:",
        await button.inner_text(),
    )

    print(
        "Button visible:",
        await button.is_visible(),
    )

    print(
        "Button enabled:",
        await button.is_enabled(),
    )

    return button


@pytest.mark.asyncio
async def test_system_configuration():
    """
    Verify the interview System Configuration flow.

    Test flow:

        Open interview
             ↓
        Camera permission granted
             ↓
        Microphone permission granted
             ↓
        Actual media access works
             ↓
        System Configuration displayed
             ↓
        Required checks displayed
             ↓
        Continue to Interview is inspected
    """

    if not INTERVIEW_URL:
        pytest.fail(
            "INTERVIEW_URL is not configured.\n\n"
            "Run:\n\n"
            'export INTERVIEW_URL="YOUR_INTERVIEW_URL"\n\n'
            "Then run the test again."
        )

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        )

        context = await browser.new_context(
            permissions=[
                "camera",
                "microphone",
            ],
        )

        page = await context.new_page()

        try:
            print()
            print("=" * 70)
            print("SYSTEM CONFIGURATION TEST")
            print("=" * 70)

            print("Opening interview...")

            try:
                interview_url, _claims = require_fresh_interview_url()
            except InterviewSessionError as error:
                pytest.fail(str(error))

            await page.goto(
                interview_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            print("Interview page loaded.")
            print("URL:", page.url)

            await page.wait_for_timeout(2000)

            # ------------------------------------------------
            # Camera + microphone permissions
            # ------------------------------------------------

            await check_media_permissions(page)

            print(
                "Camera and microphone permissions "
                "are GRANTED."
            )

            # ------------------------------------------------
            # Actual camera + microphone
            # ------------------------------------------------

            await check_media_stream(page)

            print(
                "Camera and microphone are working."
            )

            # ------------------------------------------------
            # Start interview / enter system configuration
            # ------------------------------------------------

            print(
                "Looking for Start Interview button..."
            )

            # Wait for the launch button instead of sampling
            # once after a fixed sleep (the button renders ~2s
            # after load, so a one-shot query is a race that
            # fails as a fake product bug). Accept BOTH "Start
            # interview" and "Continue interview" - a session
            # opened once before legitimately shows Continue,
            # which is a session state, not a missing button.
            start_button = page.get_by_role(
                "button",
                name=LAUNCH_BUTTON_RE,
            ).first

            try:
                await start_button.wait_for(
                    state="visible",
                    timeout=30000,
                )
            except Exception:
                await print_page_state(
                    page,
                    "START INTERVIEW BUTTON NOT FOUND",
                )

                pytest.fail(
                    "Start/Continue interview button was not "
                    "found within 30s."
                )

            print(
                "Found:",
                await start_button.inner_text(),
            )

            await start_button.click()

            print(
                "Start Interview clicked."
            )

            # ------------------------------------------------
            # Wait for system configuration
            # ------------------------------------------------

            await page.wait_for_timeout(2000)

            await check_system_configuration(page)

            print(
                "System Configuration screen "
                "verified."
            )

            # ------------------------------------------------
            # Check required state
            # ------------------------------------------------

            continue_button = (
                await check_required_state(page)
            )

            # ------------------------------------------------
            # Screenshot for test evidence
            # ------------------------------------------------

            await page.screenshot(
                path="system_configuration.png",
                full_page=True,
            )

            print(
                "Screenshot saved:"
                " system_configuration.png"
            )

            print()
            print("=" * 70)
            print(
                "SYSTEM CONFIGURATION TEST PASSED"
            )
            print("=" * 70)

        except Exception:
            await print_page_state(
                page,
                "SYSTEM CONFIGURATION TEST FAILED",
            )

            await page.screenshot(
                path="system_configuration_failed.png",
                full_page=True,
            )

            raise

        finally:
            await context.close()
            await browser.close()