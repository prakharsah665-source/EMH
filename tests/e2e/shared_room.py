"""
ONE shared room join for the three room-joining E2E tests.

Session budget: a run has exactly one joinable session for these
tests - EMH_TESTS_URL. INTERVIEW_URL is reserved for the
full-interview evaluation, and a room join fires the agent
greeting, consuming the session, so it can never be lent out.
Three tests need a live interview room:

    test_continue_to_interview
    test_interview_room
    test_livekit_connection

Previously each performed its own join: the first consumed
EMH_TESTS_URL and the other two skipped with SESSION ALREADY
CONSUMED on every run, deterministically. Now the FIRST of them
to run performs the run's single join - through the UNCHANGED
session policy (resolve_room_session requires an unconsumed
session; mark_room_joined records the join in the used-session
ledger) - and the others attach to the same live page and
validate their own responsibility against it. The browser
leaves the room gracefully exactly once, after the last of the
three has finished (see the shared_room fixture in conftest.py).

The consumed-session safeguards are untouched: a pre-consumed
EMH_TESTS_URL still skips all three tests (each consumer that
reaches the join path re-runs the unchanged policy check), and
the single successful join is still ledgered so a later run
cannot silently reuse the session.
"""

import asyncio

import pytest


# The three consumers, by test-file name. Used by the shared_room
# fixture in conftest.py to close the room exactly once, after
# the LAST collected consumer has finished.
ROOM_JOIN_TEST_FILES = (
    "test_continue_to_interview.py",
    "test_interview_room.py",
    "test_livekit_connection.py",
)


def is_room_join_nodeid(nodeid: str) -> bool:
    """True if a pytest nodeid belongs to a room-join consumer."""

    file_part = nodeid.split("::", 1)[0].replace("\\", "/")
    return file_part.rsplit("/", 1)[-1] in ROOM_JOIN_TEST_FILES


class SharedRoomJoin:
    """
    Owner of the run's single EMH_TESTS_URL room join: browser
    lifecycle, LiveKit WebSocket observation (listener installed
    BEFORE the join so test_livekit_connection can assert on the
    join handshake), and exactly-once graceful teardown.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._raw_page = None  # exists from browser start (for teardown)
        self.page = None  # set only after a fully successful join
        self.interview_url = None
        self.claims = None
        self.joined_by = None
        self.join_failed_in = None
        self.closed = False
        # LiveKit observation, recorded at join time.
        self.livekit_connections: list[str] = []
        self.livekit_opened: asyncio.Event | None = None
        self.livekit_active: asyncio.Event | None = None

    # ---------------------------------------------- observation

    def _observe_websocket(self, websocket):
        url = websocket.url

        # Never print the query string - it contains the LiveKit
        # access token.
        print(f"WebSocket opened: {url.split('?')[0]}")

        if "livekit.cloud" not in url:
            return
        print(">>> LIVEKIT WEBSOCKET DETECTED")
        self.livekit_connections.append(url)
        self.livekit_opened.set()
        websocket.on(
            "framesent", lambda payload: self.livekit_active.set()
        )
        websocket.on(
            "framereceived", lambda payload: self.livekit_active.set()
        )

    # ----------------------------------------------------- join

    async def ensure_joined(self, test_name: str):
        """
        The live interview-room page. First consumer joins (the
        session policy applies unchanged and may skip/fail);
        later consumers attach to the same page.
        """

        if self.join_failed_in:
            pytest.skip(
                "SHARED ROOM JOIN FAILED UPSTREAM - this run's "
                "single EMH_TESTS_URL room join failed in "
                f"{self.join_failed_in} (the authoritative failure "
                f"is reported there). {test_name} will not re-drive "
                "a possibly half-consumed session. Harness/session "
                "layer, not a bot failure."
            )
        if self.closed:
            pytest.fail(
                f"{test_name}: the shared room session was already "
                "closed before this test ran - shared_room fixture "
                "ordering bug (evaluator/harness layer, not a bot "
                "failure)."
            )
        if self.page is not None:
            if self.page.is_closed():
                pytest.fail(
                    "SHARED ROOM PAGE CLOSED - the page joined by "
                    f"{self.joined_by} died before {test_name} "
                    "could validate it. Capture/harness layer, not "
                    "a bot failure."
                )
            print(
                f"{test_name}: attaching to the shared room joined "
                f"by {self.joined_by} (session budget: one joinable "
                "EMH_TESTS_URL room per run)."
            )
            return self.page
        return await self._join(test_name)

    async def _start_page(self):
        """
        Launch the shared headless browser with the proven
        capture-run setup: fake media devices, granted
        permissions, and the fake-mic injection harness (the
        setup screen's microphone check needs real audio energy,
        Chrome's beep tone is not speech).
        """

        from playwright.async_api import async_playwright

        # Imported lazily: the INIT_SCRIPT lives in the (heavy)
        # capture module and is only needed on the join path.
        from tests.e2e.test_bot_responsiveness import INIT_SCRIPT

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ],
        )
        self._context = await self._browser.new_context(
            permissions=["camera", "microphone"],
        )
        await self._context.add_init_script(INIT_SCRIPT)
        return await self._context.new_page()

    async def _join(self, test_name: str):
        from pages.interview_launch import launch_into_interview_room
        from tests.e2e.session_policy import (
            mark_room_joined,
            resolve_room_session,
        )

        # UNCHANGED session policy: needs an unconsumed
        # EMH_TESTS_URL (skips with SESSION ALREADY CONSUMED
        # otherwise, fails on stale/expired/same-as-primary).
        url, claims = resolve_room_session(test_name)

        print(
            f"{test_name}: performing this run's SINGLE shared "
            "room join..."
        )
        try:
            page = await self._start_page()
            self._raw_page = page
            self.livekit_opened = asyncio.Event()
            self.livekit_active = asyncio.Event()
            # BEFORE the join, so the LiveKit handshake is observed.
            page.on("websocket", self._observe_websocket)

            # Shared launch: Start OR Continue, joyride
            # prevention, one-tab lock, system config, speaker
            # test, consent, Continue - with per-step
            # post-conditions (pages/interview_launch.py).
            await launch_into_interview_room(
                page, log=print, interview_url=url
            )
        except Exception:
            # One authoritative failure (this test); later
            # consumers skip as upstream instead of re-driving a
            # possibly half-consumed session.
            self.join_failed_in = test_name
            await self.close()
            raise

        mark_room_joined(claims, f"{test_name} shared")
        self.page = page
        self.interview_url = url
        self.claims = claims
        self.joined_by = test_name
        return page

    # ------------------------------------------------- teardown

    async def close(self):
        """
        Graceful room leave + browser teardown, exactly once
        (idempotent). Called by the shared_room fixture after the
        LAST consumer, or immediately on a failed join.
        """

        if self.closed:
            return
        self.closed = True

        from pages.interview_launch import graceful_leave_interview_room

        page = self.page or self._raw_page
        try:
            if page is not None and not page.is_closed():
                await graceful_leave_interview_room(page, log=print)
        except Exception as error:
            print(f"[WARNING] Shared-room graceful leave failed: {error}")

        for label, close in (
            ("context", getattr(self._context, "close", None)),
            ("browser", getattr(self._browser, "close", None)),
            ("playwright", getattr(self._playwright, "stop", None)),
        ):
            if close is None:
                continue
            try:
                await close()
            except Exception as error:
                print(f"[WARNING] Shared-room {label} teardown: {error}")

        print(
            "Shared room closed (graceful leave + browser teardown "
            "ran exactly once, after the last room-join test)."
        )
