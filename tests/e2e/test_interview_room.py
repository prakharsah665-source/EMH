from urllib.parse import urlparse

import pytest
from playwright.async_api import async_playwright

from evaluation.redaction import redact_pii

from config.settings import INTERVIEW_URL
from pages.interview_launch import LAUNCH_BUTTON_RE
from tests.e2e.session_policy import (
    mark_room_joined,
    resolve_room_session,
)


def _interview_origin():
    """
    Origin of the interview page under test.

    Permissions must be granted for the origin the page is
    actually served from: granting them for a hard-coded
    origin that does not match leaves the page permission
    state at "prompt".
    """

    if not INTERVIEW_URL:
        return None

    parts = urlparse(INTERVIEW_URL)

    return f"{parts.scheme}://{parts.netloc}"


ORIGIN = _interview_origin()


# ============================================================
# DEBUG / UTILITY
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
        print(redact_pii(body[:12000]))

    except Exception as error:
        print(
            "Could not read page text:",
            error,
        )

    print("=" * 70)


# ============================================================
# PERMISSIONS
# ============================================================

async def grant_permissions(context):
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
        f"Camera permission is '{camera}'."
    )

    assert microphone == "granted", (
        f"Microphone permission is '{microphone}'."
    )

    print(
        "Camera and microphone permissions are GRANTED."
    )


# ============================================================
# MEDIA ACCESS
# ============================================================

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

                const result = {
                    success: true,

                    video:
                        videoTracks.length > 0 &&
                        videoTracks[0].readyState === "live",

                    audio:
                        audioTracks.length > 0 &&
                        audioTracks[0].readyState === "live",

                    videoTracks:
                        videoTracks.length,

                    audioTracks:
                        audioTracks.length
                };

                stream.getTracks().forEach(
                    track => track.stop()
                );

                return result;

            } catch (error) {
                return {
                    success: false,
                    video: false,
                    audio: false,
                    videoTracks: 0,
                    audioTracks: 0,
                    error:
                        error.name +
                        ": " +
                        error.message
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
        "Camera access is not working."
    )

    assert result["audio"], (
        "Microphone access is not working."
    )

    print(
        "Camera and microphone access is working."
    )


# ============================================================
# START INTERVIEW
# ============================================================

async def click_start_interview(page):
    print(
        "Looking for Start Interview..."
    )

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

        await print_page_state(
            page,
            "START INTERVIEW BUTTON NOT FOUND",
        )

        pytest.fail(
            "Start/Continue interview button was not found "
            "within 30s."
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


# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

async def verify_system_configuration(page):
    """
    Verify that the System Configuration screen
    has loaded and contains the expected sections.
    """

    print(
        "Checking System Configuration..."
    )

    # Wait for the main heading first.
    try:

        await page.get_by_text(
            "System Configuration",
            exact=False,
        ).first.wait_for(
            state="visible",
            timeout=10000,
        )

    except Exception:

        await print_page_state(
            page,
            "SYSTEM CONFIGURATION NOT FOUND",
        )

        pytest.fail(
            "System Configuration screen "
            "was not found."
        )

    body = (
        await page.locator("body")
        .inner_text()
    ).lower()

    required_items = [
        "system configuration",
        "audio configuration",
        "select microphone",
        "select speaker",
        "session recording consent",
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
                f"SYSTEM CONFIGURATION ITEM MISSING: {item}",
            )

            pytest.fail(
                f"System Configuration item missing: "
                f"{item}"
            )

    print(
        "System Configuration verified."
    )


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

async def complete_audio_configuration(page):
    print()
    print(
        "Checking Audio Configuration..."
    )

    body = (
        await page.locator("body")
        .inner_text()
    ).lower()

    assert "audio configuration" in body, (
        "Audio Configuration section not found."
    )

    assert "select microphone" in body, (
        "Select Microphone section not found."
    )

    assert "select speaker" in body, (
        "Select Speaker section not found."
    )

    print(
        "[PASS] Audio Configuration section found."
    )

    # --------------------------------------------------------
    # Microphone
    # --------------------------------------------------------

    if "microphone test completed" in body:

        print(
            "[PASS] Microphone test completed."
        )

    else:

        print(
            "[WARNING] Microphone test completion "
            "text was not found."
        )

    # --------------------------------------------------------
    # Speaker
    # --------------------------------------------------------

    speaker_button = page.get_by_role(
        "button",
        name="Test Speaker Before Interview",
        exact=True,
    )

    if await speaker_button.count() > 0:

        print(
            "Speaker test button found:",
            await speaker_button.inner_text(),
        )

        await speaker_button.click()

        print(
            "Speaker test clicked."
        )

        await page.wait_for_timeout(
            1500
        )

    else:

        body = (
            await page.locator("body")
            .inner_text()
        ).lower()

        if "all tests completed" not in body:

            await print_page_state(
                page,
                "SPEAKER TEST BUTTON NOT FOUND",
            )

            pytest.fail(
                "Speaker test button was not found "
                "and audio tests are not complete."
            )

        print(
            "Speaker test already completed."
        )

    # --------------------------------------------------------
    # Verify audio completion
    # --------------------------------------------------------

    try:

        await page.get_by_text(
            "All tests completed",
            exact=True,
        ).wait_for(
            state="visible",
            timeout=10000,
        )

        print(
            "[PASS] Audio Input & Output:"
            " All tests completed."
        )

    except Exception:

        body = (
            await page.locator("body")
            .inner_text()
        ).lower()

        if "all tests completed" not in body:

            await print_page_state(
                page,
                "AUDIO CONFIGURATION NOT COMPLETE",
            )

            pytest.fail(
                "Audio configuration did not reach "
                "'All tests completed'."
            )

        print(
            "[PASS] Audio configuration completed."
        )


# ============================================================
# RECORDING CONSENT
# ============================================================

async def accept_recording_consent(page):
    print()
    print(
        "Checking Recording Consent..."
    )

    checkbox = page.locator(
        'input[type="checkbox"]'
    ).last

    if await checkbox.count() == 0:

        await print_page_state(
            page,
            "RECORDING CONSENT CHECKBOX NOT FOUND",
        )

        pytest.fail(
            "Recording consent checkbox "
            "was not found."
        )

    initial_state = (
        await checkbox.is_checked()
    )

    print(
        "Initial recording consent:",
        initial_state,
    )

    if initial_state:

        print(
            "[PASS] Recording consent already accepted."
        )

        return

    print(
        "Accepting recording consent..."
    )

    # The application uses a custom checkbox:
    #
    # <input type="checkbox" class="peer sr-only">
    #
    # Playwright .check() can be intercepted by
    # the custom visual div.
    #
    # DOM click is the same method that worked
    # in test_recording_consent.py.

    await checkbox.evaluate(
        """
        element => element.click()
        """
    )

    # Wait for React state to update.
    #
    # IMPORTANT:
    # Do not pass an element handle as a positional
    # argument to wait_for_function().
    #
    # We inspect the checkbox directly from the DOM.

    await page.wait_for_function(
        """
        () => {
            const checkbox =
                document.querySelector(
                    'input[type="checkbox"]'
                );

            return (
                checkbox !== null &&
                checkbox.checked === true
            );
        }
        """,
        timeout=5000,
    )

    final_state = (
        await checkbox.is_checked()
    )

    print(
        "Recording consent after click:",
        final_state,
    )

    assert final_state is True, (
        "Recording consent did not become checked."
    )

    print(
        "[PASS] Recording consent accepted."
    )


# ============================================================
# CONTINUE TO INTERVIEW
# ============================================================

async def find_continue_button(page):
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


async def click_continue_to_interview(page):
    print()
    print(
        "Looking for Continue to Interview..."
    )

    button = await find_continue_button(
        page
    )

    if button is None:

        await print_page_state(
            page,
            "CONTINUE TO INTERVIEW BUTTON NOT FOUND",
        )

        pytest.fail(
            "Continue to Interview button "
            "was not found."
        )

    await button.wait_for(
        state="visible",
        timeout=15000,
    )

    print(
        "Continue button found:",
        await button.inner_text(),
    )

    is_enabled = await button.is_enabled()

    print(
        "Continue button enabled:",
        is_enabled,
    )

    assert is_enabled, (
        "Continue to Interview button "
        "is disabled."
    )

    print(
        "[PASS] Continue button is enabled."
    )

    url_before = page.url

    await button.click()

    print(
        "Continue to Interview clicked."
    )

    await page.wait_for_timeout(
        4000
    )

    print(
        "URL before:",
        url_before,
    )

    print(
        "URL after:",
        page.url,
    )


# ============================================================
# INTERVIEW ROOM
# ============================================================

async def verify_interview_room(page):
    print()
    print(
        "Verifying actual interview room..."
    )

    # Give the application time to establish the
    # interview session.
    await page.wait_for_timeout(
        2000
    )

    current_url = page.url

    print(
        "Current URL:",
        current_url,
    )

    body = await page.locator(
        "body"
    ).inner_text()

    body_lower = body.lower()

    print()
    print(
        "Interview room page text:"
    )

    print(
        body[:10000]
    )

    # --------------------------------------------------------
    # System Configuration must be gone
    # --------------------------------------------------------

    system_configuration_visible = (
        "system configuration" in body_lower
    )

    print(
        "System Configuration visible:",
        system_configuration_visible,
    )

    assert not system_configuration_visible, (
        "System Configuration is still visible "
        "after Continue to Interview."
    )

    print(
        "[PASS] System Configuration is gone."
    )

    # --------------------------------------------------------
    # Interview indicators
    # --------------------------------------------------------

    indicators = [
        "interview",
        "recording",
        "camera",
        "microphone",
        "mute",
        "question",
        "livekit",
        "end interview",
        "leave interview",
    ]

    found_indicators = []

    for indicator in indicators:

        if indicator in body_lower:

            found_indicators.append(
                indicator
            )

    print(
        "Interview indicators found:",
        found_indicators,
    )

    assert len(found_indicators) > 0, (
        "No interview-room indicators were found."
    )

    print(
        "[PASS] Interview room indicators found."
    )


# ============================================================
# VIDEO
# ============================================================

async def verify_video_element(page):
    print()
    print(
        "Checking video elements..."
    )

    videos = page.locator(
        "video"
    )

    count = await videos.count()

    print(
        "Video elements found:",
        count,
    )

    if count == 0:

        print(
            "[WARNING] No video element found."
        )

        return

    visible_count = 0

    for index in range(count):

        video = videos.nth(index)

        try:

            visible = await video.is_visible()

            print(
                f"Video {index} visible:",
                visible,
            )

            if visible:
                visible_count += 1

        except Exception:
            pass

    print(
        "Visible video elements:",
        visible_count,
    )

    if visible_count > 0:

        print(
            "[PASS] Video element is visible."
        )

    else:

        print(
            "[WARNING] Video element exists but "
            "is not visible."
        )


# ============================================================
# AUDIO
# ============================================================

async def verify_audio_elements(page):
    print()
    print(
        "Checking audio elements..."
    )

    audios = page.locator(
        "audio"
    )

    count = await audios.count()

    print(
        "Audio elements found:",
        count,
    )

    if count > 0:

        print(
            "[PASS] Audio element exists."
        )

    else:

        print(
            "[INFO] No HTML audio element found."
        )


# ============================================================
# MEDIA DEVICES
# ============================================================

async def inspect_media_devices(page):
    print()
    print(
        "Inspecting media devices..."
    )

    result = await page.evaluate(
        """
        async () => {
            try {

                const devices =
                    await navigator.mediaDevices
                        .enumerateDevices();

                return devices.map(
                    device => ({
                        kind: device.kind,
                        label: device.label,
                        deviceId:
                            device.deviceId
                                ? "present"
                                : "missing"
                    })
                );

            } catch (error) {

                return {
                    error:
                        error.name +
                        ": " +
                        error.message
                };
            }
        }
        """
    )

    print(
        "Media devices:",
        result,
    )


# ============================================================
# MAIN TEST
# ============================================================

@pytest.mark.asyncio
async def test_interview_room():

    # This test JOINS the interview room, so it needs an
    # unconsumed session (skips or uses EMH_ROOM_TESTS_URL
    # when the primary session's room was already joined).
    room_url, room_claims = resolve_room_session(
        "test_interview_room"
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
                "INTERVIEW ROOM TEST"
            )
            print("=" * 70)

            # =================================================
            # 1. Open Interview
            # =================================================

            print(
                "Opening interview URL..."
            )

            await page.goto(
                room_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

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

            await grant_permissions(
                context
            )

            await verify_permissions(
                page
            )

            # =================================================
            # 3. Actual Media
            # =================================================

            await verify_media_access(
                page
            )

            # =================================================
            # 4. Start Interview
            # =================================================

            await click_start_interview(
                page
            )

            # =================================================
            # 5. System Configuration
            # =================================================

            await verify_system_configuration(
                page
            )

            # =================================================
            # 6. Audio Configuration
            # =================================================

            await complete_audio_configuration(
                page
            )

            # =================================================
            # 7. Recording Consent
            # =================================================

            await accept_recording_consent(
                page
            )

            # =================================================
            # 8. Continue
            # =================================================

            await click_continue_to_interview(
                page
            )
            mark_room_joined(
                room_claims, "test_interview_room"
            )

            # =================================================
            # 9. Interview Room
            # =================================================

            await verify_interview_room(
                page
            )

            # =================================================
            # 10. Video
            # =================================================

            await verify_video_element(
                page
            )

            # =================================================
            # 11. Audio
            # =================================================

            await verify_audio_elements(
                page
            )

            # =================================================
            # 12. Media Devices
            # =================================================

            await inspect_media_devices(
                page
            )

            # =================================================
            # 13. Screenshot
            # =================================================

            await page.screenshot(
                path="interview_room.png",
                full_page=True,
            )

            print(
                "Screenshot saved:"
                " interview_room.png"
            )

            # =================================================
            # FINAL PASS
            # =================================================

            print()
            print("=" * 70)
            print(
                "INTERVIEW ROOM TEST PASSED"
            )
            print("=" * 70)

        except Exception:

            await print_page_state(
                page,
                "INTERVIEW ROOM TEST FAILED",
            )

            try:

                await page.screenshot(
                    path="interview_room_failed.png",
                    full_page=True,
                )

                print(
                    "Failure screenshot saved:"
                    " interview_room_failed.png"
                )

            except Exception:
                pass

            raise

        finally:

            await context.close()
            await browser.close()