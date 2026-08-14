import asyncio
import re

import pytest
from playwright.async_api import async_playwright

from config.settings import INTERVIEW_URL
from config.interview_session import (
    InterviewSessionError,
    require_fresh_interview_url,
)


@pytest.mark.asyncio
async def test_livekit_connection():

    livekit_connections = []

    # Deterministic readiness signals:
    #   livekit_opened - the LiveKit WebSocket exists
    #   livekit_active - LiveKit frames were exchanged, so
    #                    the room connection is actually live
    livekit_opened = asyncio.Event()
    livekit_active = asyncio.Event()

    async with async_playwright() as playwright:

        # ==================================================
        # 1. BROWSER SETUP
        # ==================================================

        print("\n========================================")
        print("LIVEKIT CONNECTION TEST")
        print("========================================")

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
            ]
        )

        page = await context.new_page()

        # ==================================================
        # 2. CAPTURE WEBSOCKET CONNECTIONS
        # ==================================================

        def handle_websocket(websocket):

            url = websocket.url

            # Never print the query string.
            # It contains the LiveKit access token.

            safe_url = url.split("?")[0]

            print(
                f"\nWebSocket opened:\n{safe_url}"
            )

            if "livekit.cloud" in url:

                print(
                    ">>> LIVEKIT WEBSOCKET DETECTED"
                )

                livekit_connections.append(
                    url
                )

                livekit_opened.set()

                websocket.on(
                    "framesent",
                    lambda payload: livekit_active.set(),
                )

                websocket.on(
                    "framereceived",
                    lambda payload: livekit_active.set(),
                )

        page.on(
            "websocket",
            handle_websocket,
        )

        # ==================================================
        # 3. OPEN INTERVIEW
        # ==================================================

        print(
            "\n[1] Opening interview..."
        )

        try:
            interview_url, _claims = require_fresh_interview_url()
        except InterviewSessionError as error:
            pytest.fail(str(error))

        await page.goto(
            interview_url,
            wait_until="domcontentloaded",
        )

        print(
            "Interview page loaded."
        )

        # ==================================================
        # 3b. VERIFY MEDIA PERMISSIONS
        #
        # The LiveKit room needs microphone and camera
        # access. Verify the granted browser permissions
        # deterministically instead of discovering the
        # problem later as a missing WebSocket.
        # ==================================================

        for permission_name in ("microphone", "camera"):

            permission_state = await page.evaluate(
                f"""
                navigator.permissions
                    .query({{ name: "{permission_name}" }})
                    .then(result => result.state)
                """
            )

            print(
                f"Permission '{permission_name}': "
                f"{permission_state}"
            )

            assert permission_state == "granted", (
                f"Browser permission '{permission_name}' is "
                f"'{permission_state}', expected 'granted'. "
                "LiveKit cannot publish media without it."
            )

        # ==================================================
        # 4. LAUNCH INTERVIEW SETUP
        #
        # A fresh interview session shows "Start interview";
        # a resumed session shows "Continue interview".
        # Accept either so the test does not depend on
        # whether a previous session already exists.
        # ==================================================

        print(
            "\n[2] Clicking Start/Continue interview..."
        )

        launch_interview = page.get_by_role(
            "button",
            name=re.compile(
                r"^(start|continue) interview$",
                re.IGNORECASE,
            ),
        )

        # The backend allows the interview in only one
        # tab/device at a time. A previous E2E test's
        # socket can linger for a few seconds after its
        # browser closes, which shows an "Interview
        # Already Open" lock screen instead of the launch
        # button. The lock screen renders only after the
        # page's socket handshake is rejected, so wait for
        # the page to settle into either state, and reload
        # until the server releases the lock (bounded,
        # condition-driven).
        lock_screen = page.get_by_text(
            "Interview Already Open",
            exact=False,
        )

        for _ in range(12):

            await page.wait_for_function(
                """
                () => {
                    const bodyText =
                        document.body
                            ? document.body.innerText
                            : "";

                    if (
                        bodyText.includes(
                            "Interview Already Open"
                        )
                    ) {
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
                timeout=30000,
            )

            if await lock_screen.count() == 0:
                break

            print(
                "Interview locked by a previous "
                "session - reloading..."
            )

            await page.wait_for_timeout(5000)

            await page.reload(
                wait_until="domcontentloaded",
            )

        try:

            await launch_interview.first.wait_for(
                state="visible",
                timeout=30000,
            )

        except Exception:

            # A previous E2E test on the same interview
            # session can leave the backend in a transient
            # state. Dump the page so the failure is
            # diagnosable.
            print(
                "\nLaunch button not found. Page state:"
            )

            print("URL:", page.url)
            print("Title:", await page.title())

            print(
                "Body:\n",
                (
                    await page.locator("body").inner_text()
                )[:4000],
            )

            raise

        assert await launch_interview.count() == 1, (
            "Start/Continue interview button was not found."
        )

        print(
            "Launch button:",
            await launch_interview.first.inner_text(),
        )

        await launch_interview.first.click()

        print(
            "Interview launch button clicked."
        )

        # ==================================================
        # 5. SYSTEM CONFIGURATION
        # ==================================================

        print(
            "\n[3] Waiting for System Configuration..."
        )

        system_configuration = page.get_by_text(
            "System Configuration",
            exact=True,
        )

        await system_configuration.wait_for(
            state="visible",
            timeout=15000,
        )

        print(
            "System Configuration loaded."
        )

        # ==================================================
        # 6. WAIT FOR AUDIO SELECTORS
        # ==================================================

        print(
            "\n[4] Waiting for audio selectors..."
        )

        await page.wait_for_function(
            """
            () =>
                document.querySelectorAll("select").length >= 2
            """,
            timeout=15000,
        )

        selects = page.locator(
            "select"
        )

        assert await selects.count() >= 2, (
            "Microphone and speaker selectors "
            "were not loaded."
        )

        microphone = selects.nth(0)
        speaker = selects.nth(1)

        print(
            "Audio selectors loaded."
        )

        # ==================================================
        # 7. SELECT MICROPHONE
        # ==================================================

        print(
            "\n[5] Configuring microphone..."
        )

        microphone_options = await microphone.locator(
            "option"
        ).all_inner_texts()

        preferred_microphone = (
            "MacBook Pro Microphone (Built-in)"
        )

        print(
            "Available microphones:"
        )

        for option in microphone_options:
            print(
                f"  - {option}"
            )

        if preferred_microphone in microphone_options:

            await microphone.select_option(
                label=preferred_microphone
            )

            print(
                f"Selected: {preferred_microphone}"
            )

        else:

            print(
                "Preferred microphone not found."
            )

            print(
                "Using application default."
            )

        # ==================================================
        # 8. SELECT SPEAKER
        # ==================================================

        print(
            "\n[6] Configuring speaker..."
        )

        speaker_options = await speaker.locator(
            "option"
        ).all_inner_texts()

        preferred_speaker = (
            "MacBook Pro Speakers (Built-in)"
        )

        print(
            "Available speakers:"
        )

        for option in speaker_options:
            print(
                f"  - {option}"
            )

        if preferred_speaker in speaker_options:

            await speaker.select_option(
                label=preferred_speaker
            )

            print(
                f"Selected: {preferred_speaker}"
            )

        else:

            print(
                "Preferred speaker not found."
            )

            print(
                "Using application default."
            )

        # ==================================================
        # 9. TEST SPEAKER
        # ==================================================

        print(
            "\n[7] Testing speaker..."
        )

        speaker_test = page.get_by_role(
            "button",
            name="Test Speaker Before Interview",
        )

        await speaker_test.wait_for(
            state="visible",
            timeout=15000,
        )

        assert await speaker_test.count() == 1, (
            "Test Speaker Before Interview "
            "button was not found."
        )

        await speaker_test.click()

        print(
            "Speaker test started."
        )

        await page.wait_for_timeout(3000)

        print(
            "Speaker test completed."
        )

        # ==================================================
        # 10. RECORDING CONSENT
        # ==================================================

        print(
            "\n[8] Accepting recording consent..."
        )

        consent = page.locator(
            'input[type="checkbox"]'
        ).last

        await consent.wait_for(
            state="attached",
            timeout=15000,
        )

        # The input is sr-only, so Playwright's normal
        # click/check action can be blocked by the custom
        # div. element.click() triggers the actual browser
        # click event and lets React update state.
        await consent.evaluate(
            """
            element => {
                element.click();
            }
            """
        )

        await page.wait_for_function(
            """
            element => element.checked === true
            """,
            arg=await consent.element_handle(),
            timeout=5000,
        )

        print(
            "Recording consent accepted."
        )

        # ==================================================
        # 11. VERIFY CONSENT
        # ==================================================

        assert await consent.is_checked(), (
            "Recording consent was not accepted."
        )

        # ==================================================
        # 12. CONTINUE TO INTERVIEW
        # ==================================================

        print(
            "\n[9] Waiting for Continue to Interview..."
        )

        continue_button = page.get_by_role(
            "button",
            name="Continue to Interview",
        )

        await continue_button.wait_for(
            state="visible",
            timeout=15000,
        )

        assert await continue_button.count() == 1, (
            "Continue to Interview button "
            "was not found."
        )

        # Wait for React/frontend state to update.

        await page.wait_for_function(
            """
            () => {
                const buttons =
                    Array.from(
                        document.querySelectorAll("button")
                    );

                const button = buttons.find(
                    element =>
                        element.innerText.trim() ===
                        "Continue to Interview"
                );

                return (
                    button &&
                    !button.disabled
                );
            }
            """,
            timeout=15000,
        )

        print(
            "Continue to Interview is enabled."
        )

        # ==================================================
        # 13. ENTER INTERVIEW
        # ==================================================

        print(
            "\n[10] Entering interview..."
        )

        await continue_button.click()

        print(
            "Continue to Interview clicked."
        )

        # ==================================================
        # 14. WAIT FOR LIVEKIT
        # ==================================================

        print(
            "\n[11] Waiting for LiveKit WebSocket..."
        )

        # LiveKit should connect after the interview
        # room starts. Wait on the connection event itself
        # instead of polling with fixed sleeps.

        try:

            await asyncio.wait_for(
                livekit_opened.wait(),
                timeout=30,
            )

            print(
                "\n>>> LIVEKIT CONNECTION DETECTED"
            )

        except asyncio.TimeoutError:
            # The assertions below report the failure with
            # full debug information.
            print(
                "\nLiveKit WebSocket was not detected "
                "within 30 seconds."
            )

        if livekit_connections:

            print(
                "Waiting for LiveKit signalling traffic..."
            )

            try:

                await asyncio.wait_for(
                    livekit_active.wait(),
                    timeout=15,
                )

                print(
                    ">>> LiveKit frames exchanged - "
                    "room connection is live."
                )

            except asyncio.TimeoutError:
                print(
                    "LiveKit socket opened but no frames "
                    "were observed."
                )

        # ==================================================
        # 15. DEBUG INFORMATION
        # ==================================================

        print(
            "\n========================================"
        )

        print(
            "LIVEKIT DEBUG INFORMATION"
        )

        print(
            "========================================"
        )

        print(
            "\nCurrent URL:"
        )

        print(
            page.url
        )

        print(
            "\nPage title:"
        )

        print(
            await page.title()
        )

        print(
            "\nLiveKit connections captured:"
        )

        print(
            len(livekit_connections)
        )

        for websocket_url in livekit_connections:

            # Only print the safe endpoint.
            print(
                websocket_url.split("?")[0]
            )

        print(
            "\nCurrent page text:"
        )

        body = await page.locator(
            "body"
        ).inner_text()

        print(
            body
        )

        # ==================================================
        # 16. VERIFY LIVEKIT CONNECTION
        # ==================================================

        assert len(livekit_connections) > 0, (
            "LiveKit WebSocket was not established "
            "after entering the interview."
        )

        assert livekit_active.is_set(), (
            "The LiveKit WebSocket opened but exchanged no "
            "frames - the room join never completed. Check "
            "the LiveKit access token permissions and the "
            "room join response."
        )

        # ==================================================
        # 17. VERIFY LIVEKIT ENDPOINT
        # ==================================================

        livekit_url = livekit_connections[0]

        expected_endpoint = (
            "wss://easemyhiring-m2nfeq60.livekit.cloud/rtc/v1"
        )

        assert livekit_url.startswith(
            expected_endpoint
        ), (
            "Unexpected LiveKit WebSocket endpoint."
        )

        print(
            "\nLiveKit endpoint verified:"
        )

        print(
            expected_endpoint
        )

        # ==================================================
        # 18. SCREENSHOT
        # ==================================================

        screenshot_path = (
            "artifacts/screenshots/"
            "livekit_connection.png"
        )

        await page.screenshot(
            path=screenshot_path,
            full_page=True,
        )

        print(
            "\nScreenshot saved:"
        )

        print(
            screenshot_path
        )

        # ==================================================
        # 19. FINAL RESULT
        # ==================================================

        print(
            "\n========================================"
        )

        print(
            "LIVEKIT CONNECTION TEST PASSED"
        )

        print(
            "========================================"
        )

        await browser.close()