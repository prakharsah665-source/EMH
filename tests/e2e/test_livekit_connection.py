"""
LiveKit connection E2E test.

Validates, against the run's SINGLE shared EMH_TESTS_URL room
join (tests/e2e/shared_room.py), that entering the interview
room establishes a live LiveKit connection:

  * the LiveKit WebSocket opened - observed by the shared
    join's websocket listener, which is installed BEFORE the
    room is joined so the handshake is always captured, no
    matter which of the three room-join tests performed the
    join;
  * signalling frames were actually exchanged, so the room
    connection is live (not just an opened socket);
  * the endpoint is the expected easemyhiring LiveKit cloud.

The session budget is one joinable room per run (INTERVIEW_URL
is reserved for the full-interview evaluation; a room join
consumes the session), so this test shares the join with
test_continue_to_interview and test_interview_room instead of
skipping with SESSION ALREADY CONSUMED.
"""

import asyncio

import pytest

from evaluation.redaction import redact_pii
from tests.e2e.screenshots import save_screenshot


EXPECTED_LIVEKIT_ENDPOINT = (
    "wss://easemyhiring-m2nfeq60.livekit.cloud/rtc/v1"
)


@pytest.mark.asyncio(loop_scope="session")
async def test_livekit_connection(shared_room):

    print("\n========================================")
    print("LIVEKIT CONNECTION TEST")
    print("========================================")

    # Shared room join: first consumer joins through the
    # unchanged session policy, later consumers attach to the
    # same live page - tests/e2e/shared_room.py.
    page = await shared_room.ensure_joined("test_livekit_connection")

    # ==================================================
    # 1. MEDIA PERMISSIONS
    #
    # The LiveKit room needs microphone and camera access.
    # Verify the granted browser permissions deterministically
    # instead of discovering the problem later as a missing
    # WebSocket.
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
    # 2. WAIT FOR LIVEKIT
    #
    # The WebSocket events were recorded by the shared join's
    # listener at join time; if the join happened in an earlier
    # test, these waits return immediately.
    # ==================================================

    print("\nWaiting for the LiveKit WebSocket...")

    try:
        await asyncio.wait_for(
            shared_room.livekit_opened.wait(),
            timeout=30,
        )
        print("\n>>> LIVEKIT CONNECTION DETECTED")
    except asyncio.TimeoutError:
        # The assertions below report the failure with full
        # debug information.
        print(
            "\nLiveKit WebSocket was not detected "
            "within 30 seconds."
        )

    if shared_room.livekit_connections:

        print("Waiting for LiveKit signalling traffic...")

        try:
            await asyncio.wait_for(
                shared_room.livekit_active.wait(),
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
    # 3. DEBUG INFORMATION
    # ==================================================

    print("\n========================================")
    print("LIVEKIT DEBUG INFORMATION")
    print("========================================")

    print("\nCurrent URL:")
    print(page.url)

    print("\nPage title:")
    print(await page.title())

    print("\nLiveKit connections captured:")
    print(len(shared_room.livekit_connections))

    for websocket_url in shared_room.livekit_connections:
        # Only print the safe endpoint - the query string
        # contains the LiveKit access token.
        print(websocket_url.split("?")[0])

    print("\nCurrent page text:")

    body = await page.locator("body").inner_text()

    # PII-redacted and bounded: raw page bodies must never
    # reach stdout (it lands in the JUnit/HTML reports).
    print(redact_pii(body[:6000]))

    # ==================================================
    # 4. VERIFY LIVEKIT CONNECTION
    # ==================================================

    assert len(shared_room.livekit_connections) > 0, (
        "LiveKit WebSocket was not established "
        "after entering the interview."
    )

    assert shared_room.livekit_active.is_set(), (
        "The LiveKit WebSocket opened but exchanged no "
        "frames - the room join never completed. Check "
        "the LiveKit access token permissions and the "
        "room join response."
    )

    # ==================================================
    # 5. VERIFY LIVEKIT ENDPOINT
    # ==================================================

    livekit_url = shared_room.livekit_connections[0]

    assert livekit_url.startswith(EXPECTED_LIVEKIT_ENDPOINT), (
        "Unexpected LiveKit WebSocket endpoint."
    )

    print("\nLiveKit endpoint verified:")
    print(EXPECTED_LIVEKIT_ENDPOINT)

    # ==================================================
    # 6. SCREENSHOT + RESULT
    # ==================================================

    await save_screenshot(page, "livekit_connection")

    print("\n========================================")
    print("LIVEKIT CONNECTION TEST PASSED")
    print("========================================")

    # No teardown here: the shared_room fixture leaves the room
    # gracefully and closes the browser exactly once, after the
    # LAST room-join test (tests/e2e/conftest.py).
