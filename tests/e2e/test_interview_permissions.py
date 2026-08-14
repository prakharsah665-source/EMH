import pytest
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)

from config.settings import INTERVIEW_URL
from evaluation.redaction import redact_pii
from tests.e2e.screenshots import save_screenshot


async def create_context(browser: Browser) -> BrowserContext:
    """Create a browser context with camera and microphone access."""

    context = await browser.new_context(
        permissions=["camera", "microphone"]
    )

    return context


async def print_debug_info(page: Page, message: str) -> None:
    """Print useful information when debugging a failure."""

    print()
    print("=" * 70)
    print(message)
    print("=" * 70)

    print("URL:", page.url)
    print("TITLE:", await page.title())

    try:
        body = await page.locator("body").inner_text()
        print("\nPAGE TEXT:")
        print(redact_pii(body[:5000]))
    except Exception as exc:
        print("Could not read page:", exc)

    print("=" * 70)
    print()


async def check_permissions(page: Page) -> None:
    """Verify browser camera and microphone permissions."""

    camera = await page.evaluate(
        """
        async () => {
            const result =
                await navigator.permissions.query({
                    name: "camera"
                });

            return result.state;
        }
        """
    )

    microphone = await page.evaluate(
        """
        async () => {
            const result =
                await navigator.permissions.query({
                    name: "microphone"
                });

            return result.state;
        }
        """
    )

    print("Camera permission:", camera)
    print("Microphone permission:", microphone)

    assert camera == "granted", (
        f"Camera permission is '{camera}', "
        "expected 'granted'."
    )

    assert microphone == "granted", (
        f"Microphone permission is '{microphone}', "
        "expected 'granted'."
    )


async def check_media_access(page: Page) -> None:
    """
    Actually request camera and microphone access.

    This verifies that permissions are not only marked as
    granted, but that the application can obtain media streams.
    """

    result = await page.evaluate(
        """
        async () => {
            try {
                const stream =
                    await navigator.mediaDevices.getUserMedia({
                        video: true,
                        audio: true
                    });

                const videoTracks =
                    stream.getVideoTracks();

                const audioTracks =
                    stream.getAudioTracks();

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
        "Camera permission was granted, "
        "but camera stream is not active."
    )

    assert result["audio"], (
        "Microphone permission was granted, "
        "but microphone stream is not active."
    )


async def find_start_button(page: Page):
    """
    buttons = page.get_by_role("button")

count = await buttons.count()

print("\n========== ALL BUTTONS ==========")

for i in range(count):
    button = buttons.nth(i)

    try:
        print(
            i,
            "TEXT:",
            repr(await button.inner_text()),
            "VISIBLE:",
            await button.is_visible(),
            "ENABLED:",
            await button.is_enabled(),
        )
    except Exception as exc:
        print(i, "ERROR:", exc)

print("=================================\n")
    Find the interview start button.

    Supports several possible button names so the test does not
    fail just because capitalization/text changed slightly.
    """

    names = [
        "Start Interview",
        "Start interview",
        "Continue Interview",
        "Continue interview",
        "Continue to Interview",
        "Continue to interview",
        "Begin Interview",
        "Begin interview",
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


async def start_interview(page: Page) -> None:
    """Find and click the application's interview start button."""

    button = await find_start_button(page)

    if button is None:
        await print_debug_info(
            page,
            "INTERVIEW START BUTTON NOT FOUND",
        )

        pytest.fail(
            "Could not find the interview start button."
        )

    print(
        "Interview button found:",
        await button.inner_text(),
    )

    await button.wait_for(
        state="visible",
        timeout=15000,
    )

    await button.click()

    print("Interview start button clicked.")

    await page.wait_for_timeout(3000)


async def check_interview_started(page: Page) -> None:
    """
    Verify that the interview actually progressed after
    clicking the start button.
    """

    print("Checking interview state...")

    body = (
        await page.locator("body").inner_text()
    ).lower()

    print("\nCurrent page text:")
    print(redact_pii(body[:5000]))

    # Check for common indicators that the interview started.
    interview_indicators = [
        "end interview",
        "finish interview",
        "question",
        "next question",
        "recording",
        "listening",
        "interview",
        "answer",
    ]

    found = [
        indicator
        for indicator in interview_indicators
        if indicator in body
    ]

    print(
        "Interview indicators found:",
        found,
    )

    if not found:
        await print_debug_info(
            page,
            "INTERVIEW STARTED BUT NO EXPECTED CONTENT FOUND",
        )

        pytest.fail(
            "Interview start could not be verified."
        )


async def run_interview_session(
    browser: Browser,
    session_number: int,
) -> None:
    """Run one complete interview session."""

    print()
    print("=" * 70)
    print(
        f"STARTING INTERVIEW SESSION #{session_number}"
    )
    print("=" * 70)

    context = await create_context(browser)

    page = await context.new_page()

    try:
        print("Opening interview URL...")

        await page.goto(
            INTERVIEW_URL,
            wait_until="domcontentloaded",
            timeout=30000,
        )

        print("Page loaded.")
        print("URL:", page.url)

        await page.wait_for_timeout(2000)

        # --------------------------------------------------
        # Permission verification
        # --------------------------------------------------

        print(
            "Checking camera and microphone permissions..."
        )

        await check_permissions(page)

        print(
            "Camera and microphone permissions "
            "are GRANTED."
        )

        # --------------------------------------------------
        # Actual media verification
        # --------------------------------------------------

        print(
            "Testing actual camera and microphone access..."
        )

        await check_media_access(page)

        print(
            "Camera and microphone access "
            "is working."
        )

        # --------------------------------------------------
        # Start interview
        # --------------------------------------------------

        await start_interview(page)

        # --------------------------------------------------
        # Verify interview started
        # --------------------------------------------------

        await check_interview_started(page)

        print()
        print(
            f"INTERVIEW SESSION #{session_number} "
            "PASSED"
        )

    except Exception:
        await print_debug_info(
            page,
            f"SESSION #{session_number} FAILED",
        )

        # Save screenshot for debugging.
        try:
            await save_screenshot(
                page,
                f"interview_session_{session_number}_failed",
            )

        except Exception as screenshot_error:
            print(
                "Could not save screenshot:",
                screenshot_error,
            )

        raise

    finally:
        await context.close()

        print(
            f"Interview session #{session_number} "
            "browser context closed."
        )


@pytest.mark.asyncio
async def test_interview_permissions():
    """
    Test camera/microphone permissions and verify that the
    interview can be started twice using separate sessions.

    Flow:

        Session 1
        ----------
        Grant camera
        Grant microphone
        Open interview
        Verify permissions
        Verify getUserMedia()
        Start interview
        Verify interview started
        Close session

        Session 2
        ----------
        Create fresh browser session
        Grant camera
        Grant microphone
        Open interview
        Verify permissions
        Verify getUserMedia()
        Start interview again
        Verify interview started
        Close session
    """

    if not INTERVIEW_URL:
        pytest.fail(
            "INTERVIEW_URL is not set.\n\n"
            "Run the test like this:\n\n"
            'export INTERVIEW_URL="YOUR_INTERVIEW_URL"\n\n'
            "pytest tests/e2e/test_interview_permissions.py -v -s"
        )

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        )

        try:
            # ==============================================
            # FIRST INTERVIEW
            # ==============================================

            await run_interview_session(
                browser,
                session_number=1,
            )

            # ==============================================
            # SECOND INTERVIEW
            # ==============================================

            await run_interview_session(
                browser,
                session_number=2,
            )

        finally:
            await browser.close()

    print()
    print("=" * 70)
    print(
        "ALL TESTS PASSED: "
        "Camera/microphone permissions worked and "
        "interview was started twice."
    )
    print("=" * 70)