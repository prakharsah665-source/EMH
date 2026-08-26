import asyncio
import os
from urllib.parse import urlsplit, parse_qs, urlencode, urlunsplit

import pytest
from playwright.async_api import async_playwright

from config.interview_session import (
    InterviewSessionError,
    get_tests_url,
)
from pages.interview_launch import LAUNCH_BUTTON_RE


# The interview link is resolved centrally (EMH_INTERVIEW_URL
# override or INTERVIEW_URL) - the same session every other
# test uses. BASE_URL remains an explicit escape hatch.
def resolve_base_url() -> str:
    explicit = os.getenv("BASE_URL")
    if explicit:
        return explicit
    try:
        return get_tests_url()
    except InterviewSessionError:
        return ""


def mask_websocket_url(url: str) -> str:
    """
    Remove sensitive token values from a WebSocket URL
    before printing it to the terminal.
    """

    parts = urlsplit(url)

    query = parse_qs(
        parts.query,
        keep_blank_values=True,
    )

    if "token" in query:
        query["token"] = ["***REDACTED***"]

    safe_query = urlencode(
        query,
        doseq=True,
    )

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            safe_query,
            parts.fragment,
        )
    )


@pytest.mark.asyncio
async def test_socket_connection():
    """
    Verify that the EMH application establishes a
    WebSocket connection during the interview flow.
    """

    print("\n========================================")
    print("SOCKET CONNECTION TEST")
    print("========================================")

    base_url = resolve_base_url()
    if not base_url:
        pytest.fail(
            "No interview URL configured - set INTERVIEW_URL "
            "(or the EMH_INTERVIEW_URL / BASE_URL override)."
        )

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=False
        )

        context = await browser.new_context(
            permissions=["microphone"]
        )

        page = await context.new_page()

        websocket_connections = []
        websocket_errors = []

        # Deterministic readiness signals:
        #   socket_opened - a WebSocket connection exists
        #   socket_active - at least one frame was exchanged,
        #                   so the connection is actually live
        socket_opened = asyncio.Event()
        socket_active = asyncio.Event()

        # -------------------------------------------------
        # Capture WebSocket connections
        # -------------------------------------------------

        def handle_websocket(websocket):
            websocket_connections.append(websocket)
            socket_opened.set()

            websocket.on(
                "framesent",
                lambda payload: socket_active.set(),
            )

            websocket.on(
                "framereceived",
                lambda payload: socket_active.set(),
            )

            print(
                "\nWebSocket detected:"
            )

            print(
                mask_websocket_url(
                    websocket.url
                )
            )

            websocket.on(
                "close",
                lambda: print(
                    "WebSocket closed"
                ),
            )

            websocket.on(
                "socketerror",
                lambda error: websocket_errors.append(error),
            )

        page.on(
            "websocket",
            handle_websocket,
        )

        try:

            # -------------------------------------------------
            # Open EMH interview
            # -------------------------------------------------

            print(
                "\nOpening EMH interview..."
            )

            await page.goto(
                base_url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            await page.wait_for_load_state(
                "networkidle"
            )

            print(
                "Application loaded"
            )

            # -------------------------------------------------
            # Start interview if required
            # -------------------------------------------------

            # Accept BOTH "Start interview" and "Continue
            # interview" - a session opened once before shows
            # Continue (session state, not a missing button).
            start_button = page.get_by_role(
                "button",
                name=LAUNCH_BUTTON_RE,
            )

            if await start_button.count() > 0:

                try:
                    await start_button.first.click(
                        timeout=10_000
                    )

                    print(
                        "Start Interview clicked"
                    )

                except Exception as error:

                    print(
                        "Could not click Start Interview:",
                        error,
                    )

            else:

                print(
                    "Start Interview button not displayed"
                )

            # -------------------------------------------------
            # Handle consent checkbox
            # -------------------------------------------------

            consent_checkbox = page.locator(
                'input[type="checkbox"]'
            )

            if await consent_checkbox.count() > 0:

                try:

                    if not await consent_checkbox.first.is_checked():

                        await consent_checkbox.first.check(
                            timeout=10_000
                        )

                        print(
                            "Consent checkbox selected"
                        )

                except Exception as error:

                    print(
                        "Consent checkbox handling:",
                        error,
                    )

            # -------------------------------------------------
            # Handle consent / privacy button
            # -------------------------------------------------

            consent_button_names = [
                "I Agree",
                "Agree",
                "Accept",
                "Continue",
                "Proceed",
            ]

            consent_clicked = False

            for button_name in consent_button_names:

                button = page.get_by_role(
                    "button",
                    name=button_name,
                )

                if await button.count() > 0:

                    try:

                        await button.first.click(
                            timeout=5_000
                        )

                        print(
                            f"'{button_name}' clicked"
                        )

                        consent_clicked = True

                        break

                    except Exception:
                        continue

            if not consent_clicked:

                print(
                    "No additional consent button required"
                )

            # -------------------------------------------------
            # Wait for Socket.IO connection
            #
            # Deterministic application-state wait: return as
            # soon as a WebSocket opens and exchanges a frame,
            # instead of sleeping for a fixed duration.
            # -------------------------------------------------

            print(
                "\nWaiting for WebSocket connection..."
            )

            try:

                await asyncio.wait_for(
                    socket_opened.wait(),
                    timeout=30,
                )

                print(
                    "WebSocket connection established."
                )

            except asyncio.TimeoutError:
                # The assertion below reports the failure
                # with full connection details.
                pass

            if websocket_connections:

                print(
                    "Waiting for WebSocket traffic..."
                )

                try:

                    await asyncio.wait_for(
                        socket_active.wait(),
                        timeout=15,
                    )

                    print(
                        "WebSocket frame exchanged - "
                        "connection is live."
                    )

                except asyncio.TimeoutError:
                    print(
                        "No WebSocket frames were observed."
                    )

            # -------------------------------------------------
            # Print WebSocket information
            # -------------------------------------------------

            print(
                "\n========================================"
            )

            print(
                f"WebSocket connections detected: "
                f"{len(websocket_connections)}"
            )

            print(
                "========================================"
            )

            for index, websocket in enumerate(
                websocket_connections,
                start=1,
            ):

                print(
                    f"\nWebSocket #{index}"
                )

                print(
                    mask_websocket_url(
                        websocket.url
                    )
                )

            # -------------------------------------------------
            # Validate connection
            # -------------------------------------------------

            assert len(websocket_connections) > 0, (
                "No WebSocket connection was detected."
            )

            assert socket_active.is_set(), (
                "A WebSocket connection was opened but no "
                "frames were exchanged - the connection "
                "never became live."
            )

            # -------------------------------------------------
            # Validate no socket errors
            # -------------------------------------------------

            if websocket_errors:

                print(
                    "\nWebSocket errors detected:"
                )

                for error in websocket_errors:
                    print(error)

            # -------------------------------------------------
            # Test passed
            # -------------------------------------------------

            print(
                "\n========================================"
            )

            print(
                "SOCKET CONNECTION TEST PASSED"
            )

            print(
                "========================================"
            )

        finally:

            # -------------------------------------------------
            # Cleanup
            # -------------------------------------------------

            await context.close()
            await browser.close()