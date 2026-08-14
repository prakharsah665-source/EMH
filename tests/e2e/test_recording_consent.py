from urllib.parse import urlparse

import pytest
from playwright.async_api import async_playwright

from evaluation.redaction import redact_pii
from tests.e2e.screenshots import save_screenshot

from config.settings import INTERVIEW_URL
from pages.interview_launch import LAUNCH_BUTTON_RE
from config.interview_session import (
    InterviewSessionError,
    require_fresh_interview_url,
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


async def print_page_state(page, title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print("URL:", page.url)
    print("TITLE:", await page.title())

    try:
        body = await page.locator("body").inner_text()

        print("\nPAGE TEXT:")
        print(redact_pii(body[:7000]))

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

    await page.wait_for_timeout(2000)


async def verify_system_configuration(page):
    print(
        "Checking System Configuration..."
    )

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

            print(
                "[PASS] Found:",
                item,
            )

        else:

            await print_page_state(
                page,
                f"MISSING SYSTEM CONFIGURATION: {item}",
            )

            pytest.fail(
                f"System Configuration item missing: "
                f"{item}"
            )

    print(
        "System Configuration verified."
    )


async def find_consent_checkbox(page):
    """
    Find the actual recording consent checkbox.

    The application uses:

        input[type="checkbox"].peer.sr-only

    The input is visually hidden, but it is the real
    checkbox whose checked state is controlled by React.
    """

    checkboxes = page.locator(
        'input[type="checkbox"]'
    )

    count = await checkboxes.count()

    print(
        "Checkbox count:",
        count,
    )

    if count == 0:
        return None

    for index in range(count):

        checkbox = checkboxes.nth(index)

        print(
            f"Checkbox {index}:",
            "checked=",
            await checkbox.is_checked(),
            "class=",
            await checkbox.get_attribute("class"),
        )

    return checkboxes.last


async def inspect_consent_checkbox(page, checkbox):
    """
    Print the actual DOM structure for debugging.
    """

    html = await checkbox.evaluate(
        """
        element => {
            const label =
                element.closest("label");

            return label
                ? label.outerHTML
                : element.outerHTML;
        }
        """
    )

    print()
    print(
        "========== CONSENT CHECKBOX DOM =========="
    )
    print(html)
    print(
        "==========================================="
    )
    print()


async def click_consent_checkbox(
    page,
    checkbox,
    expected_state,
):
    """
    Toggle the recording consent checkbox using the
    native input's DOM click.

    expected_state:
        True  -> checkbox should become checked
        False -> checkbox should become unchecked
    """

    current_state = await checkbox.is_checked()

    print(
        "Current consent state:",
        current_state,
    )

    print(
        "Expected consent state:",
        expected_state,
    )

    # Already in the desired state.
    if current_state == expected_state:

        print(
            "Consent is already in expected state."
        )

        return

    print(
        "Clicking native checkbox DOM..."
    )

    # The input is sr-only, so Playwright's normal
    # click/check action can be blocked by the custom div.
    #
    # element.click() triggers the actual browser click
    # event on the checkbox and lets React update state.
    await checkbox.evaluate(
        """
        element => {
            element.click();
        }
        """
    )

    # Wait for React to update the checkbox state.
    try:

        await page.wait_for_function(
            """
            ({ element, expected }) =>
                element.checked === expected
            """,
            arg={
                "element": await checkbox.element_handle(),
                "expected": expected_state,
            },
            timeout=5000,
        )

    except Exception:

        # Read state manually for a useful failure message.
        actual_state = await checkbox.is_checked()

        print(
            "Checkbox state after DOM click:",
            actual_state,
        )

        await print_page_state(
            page,
            "CONSENT STATE DID NOT UPDATE",
        )

        pytest.fail(
            "Recording consent did not change to "
            f"{expected_state}. "
            f"Actual state: {actual_state}"
        )

    final_state = await checkbox.is_checked()

    print(
        "Consent state after click:",
        final_state,
    )

    assert final_state == expected_state, (
        f"Expected consent state {expected_state}, "
        f"but got {final_state}."
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
        "Looking for Continue to Interview..."
    )

    button = await find_continue_button(page)

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
        "Continue button visible:",
        await button.is_visible(),
    )

    print(
        "Continue button enabled:",
        await button.is_enabled(),
    )

    return button


@pytest.mark.asyncio
async def test_recording_consent():
    """
    Test recording consent behavior.

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
        Consent initially unchecked
             ↓
        Check consent
             ↓
        Verify checked
             ↓
        Uncheck consent
             ↓
        Verify unchecked
             ↓
        Check consent again
             ↓
        Verify checked
             ↓
        Verify Continue button
    """

    if not INTERVIEW_URL:

        pytest.fail(
            "INTERVIEW_URL is not set.\n\n"
            "Run:\n\n"
            'export INTERVIEW_URL="YOUR_INTERVIEW_URL"\n\n'
            "Then run the test."
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
                "RECORDING CONSENT TEST"
            )
            print("=" * 70)

            # =================================================
            # 1. Open interview
            # =================================================

            print(
                "Opening interview URL..."
            )

            try:
                interview_url, _claims = require_fresh_interview_url()
            except InterviewSessionError as error:
                pytest.fail(str(error))

            await page.goto(
                interview_url,
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
            # 2. Grant permissions
            # =================================================

            await grant_media_permissions(
                context
            )

            # =================================================
            # 3. Verify permissions
            # =================================================

            await verify_permissions(
                page
            )

            # =================================================
            # 4. Verify actual media
            # =================================================

            await verify_media_access(
                page
            )

            # =================================================
            # 5. Start interview setup
            # =================================================

            await click_start_interview(
                page
            )

            # =================================================
            # 6. Verify system configuration
            # =================================================

            await verify_system_configuration(
                page
            )

            # =================================================
            # 7. Find consent checkbox
            # =================================================

            checkbox = await find_consent_checkbox(
                page
            )

            if checkbox is None:

                await print_page_state(
                    page,
                    "CONSENT CHECKBOX NOT FOUND",
                )

                pytest.fail(
                    "Recording consent checkbox "
                    "was not found."
                )

            # =================================================
            # 8. Inspect actual DOM
            # =================================================

            await inspect_consent_checkbox(
                page,
                checkbox
            )

            # =================================================
            # 9. Verify initial state
            # =================================================

            initial_state = await checkbox.is_checked()

            print(
                "Initial consent state:",
                initial_state,
            )

            if initial_state:

                print(
                    "[WARNING] Consent was already checked."
                )

            else:

                print(
                    "[PASS] Consent initially unchecked."
                )

            # =================================================
            # 10. Accept consent
            # =================================================

            print()
            print(
                "STEP 1: Accept recording consent"
            )

            await click_consent_checkbox(
                page,
                checkbox,
                True,
            )

            checked_state = await checkbox.is_checked()

            print(
                "Consent after accepting:",
                checked_state,
            )

            assert checked_state is True

            print(
                "[PASS] Recording consent accepted."
            )

            # =================================================
            # 11. Remove consent
            # =================================================

            print()
            print(
                "STEP 2: Remove recording consent"
            )

            await click_consent_checkbox(
                page,
                checkbox,
                False,
            )

            unchecked_state = await checkbox.is_checked()

            print(
                "Consent after removing:",
                unchecked_state,
            )

            assert unchecked_state is False

            print(
                "[PASS] Recording consent removed."
            )

            # =================================================
            # 12. Accept again
            # =================================================

            print()
            print(
                "STEP 3: Accept recording consent again"
            )

            await click_consent_checkbox(
                page,
                checkbox,
                True,
            )

            final_state = await checkbox.is_checked()

            print(
                "Final consent state:",
                final_state,
            )

            assert final_state is True

            print(
                "[PASS] Recording consent accepted again."
            )

            # =================================================
            # 13. Continue button
            # =================================================

            print()
            print(
                "STEP 4: Verify Continue to Interview"
            )

            continue_button = (
                await verify_continue_button(
                    page
                )
            )

            assert await continue_button.is_visible()

            print(
                "[PASS] Continue to Interview exists."
            )

            # Do not require enabled=True here.
            #
            # Speaker testing is also required in your
            # application, so consent alone does not mean
            # the candidate is ready.

            # =================================================
            # 14. Screenshot
            # =================================================

            await save_screenshot(
                page, "recording_consent"
            )

            # =================================================
            # FINAL RESULT
            # =================================================

            print()
            print("=" * 70)
            print(
                "RECORDING CONSENT TEST PASSED"
            )
            print("=" * 70)

        except Exception:

            await print_page_state(
                page,
                "RECORDING CONSENT TEST FAILED",
            )

            try:

                await save_screenshot(
                    page, "recording_consent_failed"
                )

            except Exception:
                pass

            raise

        finally:

            await context.close()
            await browser.close()