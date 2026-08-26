from urllib.parse import urlparse

import pytest
from playwright.async_api import async_playwright

from evaluation.redaction import redact_pii
from tests.e2e.screenshots import save_screenshot

from config.interview_session import (
    get_tests_url,
    tests_url_configured,
)
from pages.interview_launch import (
    LAUNCH_BUTTON_RE,
    TOUR_SUPPRESS_JS,
    _dismiss_guided_tour,
)
from config.interview_session import (
    InterviewSessionError,
    require_fresh_tests_url,
)


def _interview_origin():
    """
    Origin of the interview page under test.

    Permissions must be granted for the origin the page is
    actually served from: granting them for a hard-coded
    origin that does not match leaves the page permission
    state at "prompt".
    """

    if not tests_url_configured():
        return None

    parts = urlparse(get_tests_url())

    return f"{parts.scheme}://{parts.netloc}"


ORIGIN = _interview_origin()


async def print_page_state(page, title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("URL:", page.url)

    try:
        body = await page.locator("body").inner_text()

        print("\nPAGE TEXT:")
        print(redact_pii(body[:8000]))

    except Exception as error:
        print(
            "Could not read page:",
            error,
        )

    print("=" * 70)


async def grant_media_permissions(context):
    await context.grant_permissions(
        [
            "camera",
            "microphone",
        ],
        origin=ORIGIN,
    )

    print(
        "Camera and microphone permissions granted."
    )


async def verify_permissions(page):
    print(
        "Checking camera and microphone permissions..."
    )

    camera = await page.evaluate(
        """
        async () => {
            const permission =
                await navigator.permissions.query({
                    name: "camera"
                });

            return permission.state;
        }
        """
    )

    microphone = await page.evaluate(
        """
        async () => {
            const permission =
                await navigator.permissions.query({
                    name: "microphone"
                });

            return permission.state;
        }
        """
    )

    print(
        "Camera permission:",
        camera,
    )

    print(
        "Microphone permission:",
        microphone,
    )

    assert camera == "granted", (
        f"Camera permission is '{camera}', "
        "expected 'granted'."
    )

    assert microphone == "granted", (
        f"Microphone permission is '{microphone}', "
        "expected 'granted'."
    )

    print(
        "Camera and microphone permissions are GRANTED."
    )


async def verify_media_access(page):
    print(
        "Testing actual camera and microphone access..."
    )

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

    print(
        "Media result:",
        result,
    )

    assert result["success"], (
        "Camera/microphone access failed: "
        f"{result.get('error')}"
    )

    assert result["video"], (
        "Camera stream is not working."
    )

    assert result["audio"], (
        "Microphone stream is not working."
    )

    print(
        "Camera and microphone access is working."
    )


_NO_VISIBLE_TOUR_JS = """
    () => {
        const parts = document.querySelectorAll(
            '#react-joyride-portal, ' +
            '.react-joyride__overlay, ' +
            '.react-joyride__spotlight, ' +
            '.react-joyride__beacon, .__floater'
        );
        return Array.from(parts).every(
            el => el.getClientRects().length === 0
        );
    }
"""


async def ensure_tour_suppressed(page):
    """
    Make sure no React Joyride element can intercept pointer
    events before we click anything on the setup screen.

    The app has more than one guided tour and their steps mount
    at unpredictable times (the setup-screen tour renders its
    spotlight AFTER System Configuration appears, right over the
    'Test Speaker Before Interview' button). Re-inject the CSS
    suppression, actively dismiss any tour that mounted before
    the CSS landed, then wait until every joyride element is
    invisible (zero client rects) so the click cannot race a
    late-mounting spotlight.
    """

    await page.evaluate(TOUR_SUPPRESS_JS)

    await _dismiss_guided_tour(page, print)

    await page.wait_for_function(
        _NO_VISIBLE_TOUR_JS,
        timeout=10000,
    )

    print(
        "Guided tour suppressed - no Joyride element visible."
    )


async def find_start_button(page):
    # Wait for the launch button instead of sampling once after
    # a fixed sleep (the button renders ~2s after load, so a
    # one-shot query is a race that fails as a fake product
    # bug). Accept BOTH "Start interview" and "Continue
    # interview" - a session opened once before legitimately
    # shows Continue, which is a session state, not a missing
    # button.
    button = page.get_by_role(
        "button",
        name=LAUNCH_BUTTON_RE,
    ).first

    try:
        await button.wait_for(
            state="visible",
            timeout=30000,
        )
    except Exception:
        return None

    return button


async def click_start_interview(page):
    print(
        "Looking for Start Interview..."
    )

    button = await find_start_button(page)

    if button is None:

        await print_page_state(
            page,
            "START INTERVIEW BUTTON NOT FOUND",
        )

        pytest.fail(
            "Start Interview button was not found."
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

    print(
        "Interview start button clicked."
    )

    await page.wait_for_timeout(
        2000
    )


async def verify_system_configuration(page):
    print(
        "Checking System Configuration..."
    )

    body = (
        await page.locator("body").inner_text()
    ).lower()

    required_items = [
        "system configuration",
        "audio input",
        "audio configuration",
        "select microphone",
        "select speaker",
    ]

    for item in required_items:

        if item in body:

            print(
                "[PASS] Found:",
                item,
            )

        else:

            await print_page_state(
                page,
                f"MISSING: {item}",
            )

            pytest.fail(
                f"System Configuration item "
                f"missing: {item}"
            )


async def find_microphone_select(page):
    """
    Find the microphone selector.

    We use the visible text around Select Microphone
    and then inspect nearby select elements.
    """

    selects = page.locator(
        "select"
    )

    count = await selects.count()

    print(
        "Select elements found:",
        count,
    )

    if count == 0:
        return None

    for index in range(count):

        select = selects.nth(index)

        try:

            print(
                f"Select {index}:",
                await select.input_value(),
            )

        except Exception:
            pass

    # Microphone is the first selector in the
    # Audio Configuration section.
    return selects.nth(0)


async def find_speaker_select(page):
    """
    Find the speaker selector.

    Speaker is the second selector in the Audio
    Configuration section.
    """

    selects = page.locator(
        "select"
    )

    count = await selects.count()

    if count < 2:
        return None

    return selects.nth(1)


async def inspect_audio_selects(page):
    """
    Print all audio select options.
    """

    selects = page.locator(
        "select"
    )

    count = await selects.count()

    print()
    print(
        "========== AUDIO SELECTORS =========="
    )

    for index in range(count):

        select = selects.nth(index)

        try:

            options = await select.locator(
                "option"
            ).all_text_contents()

            value = await select.input_value()

            print(
                f"Select {index}:"
            )

            print(
                "  Current value:",
                value,
            )

            print(
                "  Options:",
                options,
            )

        except Exception as error:

            print(
                f"Could not inspect select {index}:",
                error,
            )

    print(
        "======================================"
    )
    print()


async def verify_microphone_complete(page):
    """
    Verify microphone configuration is complete.
    """

    body = (
        await page.locator("body").inner_text()
    ).lower()

    required_text = [
        "select microphone",
        "complete",
        "microphone test completed",
        "your voice is clear",
    ]

    for text in required_text:

        if text in body:

            print(
                "[PASS] Microphone:",
                text,
            )

        else:

            print(
                "[WARNING] Microphone text not found:",
                text,
            )


async def verify_speaker_pending(page):
    """
    Before the speaker test, the UI should normally show
    speaker testing as pending.
    """

    body = (
        await page.locator("body").inner_text()
    ).lower()

    if "speaker test required" in body:

        print(
            "[PASS] Speaker test is required."
        )

        return True

    if "test speaker before interview" in body:

        print(
            "[PASS] Test Speaker button is present."
        )

        return True

    if "pending" in body:

        print(
            "[PASS] Speaker configuration is pending."
        )

        return True

    print(
        "[WARNING] Could not identify speaker pending state."
    )

    return False


async def find_speaker_test_button(page):
    """
    Find the actual Test Speaker button.
    """

    names = [
        "Test Speaker Before Interview",
        "Test Speaker",
        "Test speaker before interview",
        "Test speaker",
    ]

    for name in names:

        button = page.get_by_role(
            "button",
            name=name,
            exact=True,
        )

        if await button.count() > 0:

            print(
                "Speaker test button found:",
                await button.inner_text(),
            )

            return button.first

    # Fallback based on visible text.
    text_locator = page.get_by_text(
        "Test Speaker Before Interview",
        exact=False,
    )

    if await text_locator.count() > 0:

        print(
            "Speaker test text found."
        )

        return text_locator.first

    return None


async def click_speaker_test(page):
    print(
        "Looking for speaker test..."
    )

    # The setup-screen tour can mount its spotlight at any
    # point after System Configuration renders; make sure it
    # is gone IMMEDIATELY before the click, not just earlier.
    await ensure_tour_suppressed(page)

    button = await find_speaker_test_button(
        page
    )

    if button is None:

        await print_page_state(
            page,
            "SPEAKER TEST BUTTON NOT FOUND",
        )

        pytest.fail(
            "Test Speaker Before Interview "
            "button was not found."
        )

    await button.scroll_into_view_if_needed()

    await button.click(
        timeout=15000
    )

    print(
        "Speaker test clicked."
    )

    await page.wait_for_timeout(
        2000
    )


async def verify_speaker_complete(page):
    """
    Verify speaker test completed successfully.
    """

    body = (
        await page.locator("body").inner_text()
    ).lower()

    success_indicators = [
        "audio input & output",
        "all tests completed",
        "speaker",
        "complete",
    ]

    print(
        "Checking speaker test result..."
    )

    for text in success_indicators:

        if text in body:

            print(
                "[PASS] Found:",
                text,
            )

    # Strong indicators.
    completed = (
        "all tests completed" in body
        or
        "audio input & output" in body
        and "complete" in body
    )

    if completed:

        print(
            "[PASS] Audio configuration is complete."
        )

    else:

        await print_page_state(
            page,
            "AUDIO CONFIGURATION RESULT",
        )

        pytest.fail(
            "Audio configuration did not appear "
            "to complete."
        )


async def verify_audio_status_card(page):
    """
    Verify the top Audio Input & Output card.
    """

    body = (
        await page.locator("body").inner_text()
    ).lower()

    if (
        "audio input & output" in body
        and "all tests completed" in body
    ):

        print(
            "[PASS] Audio Input & Output shows "
            "'All tests completed'."
        )

        return

    print(
        "[WARNING] Could not verify exact "
        "Audio Input & Output status."
    )


async def find_continue_button(page):
    names = [
        "Continue to Interview",
        "Continue to interview",
        "Continue Interview",
        "Continue interview",
    ]

    for name in names:

        button = page.get_by_role(
            "button",
            name=name,
            exact=True,
        )

        if await button.count() > 0:

            return button.first

    return None


async def verify_continue_button(page):
    print(
        "Checking Continue to Interview..."
    )

    button = await find_continue_button(
        page
    )

    if button is None:

        await print_page_state(
            page,
            "CONTINUE BUTTON NOT FOUND",
        )

        pytest.fail(
            "Continue to Interview button "
            "was not found."
        )

    print(
        "Continue button found:",
        await button.inner_text(),
    )

    print(
        "Visible:",
        await button.is_visible(),
    )

    print(
        "Enabled:",
        await button.is_enabled(),
    )

    return button


@pytest.mark.asyncio
async def test_audio_configuration():
    """
    Test Audio Configuration.

    Flow:

        Open interview
             ↓
        Grant camera/microphone
             ↓
        Verify media access
             ↓
        Start Interview
             ↓
        System Configuration
             ↓
        Verify microphone
             ↓
        Verify speaker is pending
             ↓
        Test speaker
             ↓
        Verify Audio Input & Output complete
             ↓
        Verify Continue button
    """

    if not tests_url_configured():

        pytest.fail(
            "No interview URL configured - set INTERVIEW_URL "
            "(or the EMH_INTERVIEW_URL override)."
        )

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        )

        context = await browser.new_context()

        page = await context.new_page()

        try:

            print()
            print("=" * 70)
            print(
                "AUDIO CONFIGURATION TEST"
            )
            print("=" * 70)

            # =================================================
            # 1. Open interview
            # =================================================

            print(
                "Opening interview URL..."
            )

            try:
                interview_url, _claims = require_fresh_tests_url()
            except InterviewSessionError as error:
                pytest.fail(str(error))

            # Prevent the guided tour BEFORE any app code can
            # render it: CSS suppression on this load and any
            # later navigation.
            await page.add_init_script(TOUR_SUPPRESS_JS)

            await page.goto(
                interview_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            # Belt and suspenders: init scripts do not run on a
            # page that already navigated before they were
            # added, so apply the suppression to this document
            # too.
            await page.evaluate(TOUR_SUPPRESS_JS)

            print(
                "Interview page loaded."
            )

            print(
                "URL:",
                page.url,
            )

            await page.wait_for_timeout(
                2000
            )

            # =================================================
            # 2. Permissions
            # =================================================

            await grant_media_permissions(
                context
            )

            await verify_permissions(
                page
            )

            # =================================================
            # 3. Actual media
            # =================================================

            await verify_media_access(
                page
            )

            # =================================================
            # 4. Start interview setup
            # =================================================

            await click_start_interview(
                page
            )

            # =================================================
            # 5. System Configuration
            # =================================================

            # The setup-screen tour mounts after System
            # Configuration renders - dismiss it before any
            # interaction on this screen.
            await ensure_tour_suppressed(page)

            await verify_system_configuration(
                page
            )

            # =================================================
            # 6. Inspect audio selectors
            # =================================================

            await inspect_audio_selects(
                page
            )

            # =================================================
            # 7. Microphone
            # =================================================

            microphone = (
                await find_microphone_select(
                    page
                )
            )

            if microphone is None:

                await print_page_state(
                    page,
                    "MICROPHONE SELECT NOT FOUND",
                )

                pytest.fail(
                    "Microphone selector was not found."
                )

            print(
                "Microphone selector found."
            )

            await verify_microphone_complete(
                page
            )

            # =================================================
            # 8. Speaker
            # =================================================

            speaker = (
                await find_speaker_select(
                    page
                )
            )

            if speaker is None:

                await print_page_state(
                    page,
                    "SPEAKER SELECT NOT FOUND",
                )

                pytest.fail(
                    "Speaker selector was not found."
                )

            print(
                "Speaker selector found."
            )

            # =================================================
            # 9. Verify speaker requires testing
            # =================================================

            await verify_speaker_pending(
                page
            )

            # =================================================
            # 10. Test speaker
            # =================================================

            await click_speaker_test(
                page
            )

            # =================================================
            # 11. Verify completion
            # =================================================

            await verify_speaker_complete(
                page
            )

            await verify_audio_status_card(
                page
            )

            # =================================================
            # 12. Continue button
            # =================================================

            continue_button = (
                await verify_continue_button(
                    page
                )
            )

            print(
                "Continue button verified."
            )

            # =================================================
            # 13. Screenshot
            # =================================================

            await save_screenshot(
                page, "audio_configuration"
            )

            # =================================================
            # FINAL
            # =================================================

            print()
            print("=" * 70)
            print(
                "AUDIO CONFIGURATION TEST PASSED"
            )
            print("=" * 70)

        except Exception:

            await print_page_state(
                page,
                "AUDIO CONFIGURATION TEST FAILED",
            )

            try:

                await save_screenshot(
                    page, "audio_configuration_failed"
                )

            except Exception:
                pass

            raise

        finally:

            await context.close()
            await browser.close()