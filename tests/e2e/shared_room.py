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
validate their own responsibility against it. After the last of
the three has finished (pass, fail or skip) the browser EXITS
the room exactly once (see the shared_room fixture in
conftest.py): click the app's End/Leave Interview control,
explicitly close the LiveKit + socket.io WebSockets, peer
connections and local tracks, verify they are closed, navigate
away so unload handlers flush, then close the browser. A
session-level safety net in conftest.py repeats this if the
fixture path was bypassed.

The consumed-session safeguards are untouched: a pre-consumed
EMH_TESTS_URL still skips all three tests (each consumer that
reaches the join path re-runs the unchanged policy check), and
the single successful join is still ledgered so a later run
cannot silently reuse the session.
"""

import asyncio
import re

import pytest


# Teardown time-boxes (seconds). The room exit must never hang
# the suite: each step is bounded and exception-safe.
END_BUTTON_TIMEOUT_S = 6.0
APP_LEAVE_SETTLE_S = 4.0
EXIT_VERIFY_TIMEOUT_S = 10.0
NAVIGATE_AWAY_TIMEOUT_S = 15.0
BROWSER_TEARDOWN_TIMEOUT_S = 15.0

# Matches the app's own room-exit control (the launch helper
# treats "end interview" / "leave interview" as room markers).
END_BUTTON_PATTERN = re.compile(
    r"(end|leave|exit)\s+(the\s+)?(interview|call|room|session)",
    re.IGNORECASE,
)
CONFIRM_BUTTON_PATTERN = re.compile(
    r"^\s*(confirm|yes|ok|okay|end|leave|exit|end interview|"
    r"leave interview|yes, end|yes, leave)\s*$",
    re.IGNORECASE,
)


# Installed in the shared page AFTER the capture INIT_SCRIPT, so
# it wraps the already-instrumented constructors. It records the
# page's WebSockets (LiveKit signalling + socket.io), peer
# connections and local media streams so the teardown can
# EXPLICITLY close them and VERIFY they are closed - instead of
# relying on the app's unload handlers alone.
ROOM_EXIT_INIT_SCRIPT = """
(() => {
    if (window.__emhRoomExit) return;

    const sockets = [];
    const pcs = [];
    const streams = [];

    const NativeWS = window.WebSocket;
    const WrappedWS = function (...args) {
        const ws = new NativeWS(...args);
        sockets.push(ws);
        return ws;
    };
    WrappedWS.prototype = NativeWS.prototype;
    for (const k of ['CONNECTING', 'OPEN', 'CLOSING', 'CLOSED']) {
        WrappedWS[k] = NativeWS[k];
    }
    Object.setPrototypeOf(WrappedWS, NativeWS);
    window.WebSocket = WrappedWS;

    const PrevPC = window.RTCPeerConnection;
    if (PrevPC) {
        const WrappedPC = function (...args) {
            const pc = new PrevPC(...args);
            pcs.push(pc);
            return pc;
        };
        WrappedPC.prototype = PrevPC.prototype;
        Object.setPrototypeOf(WrappedPC, PrevPC);
        window.RTCPeerConnection = WrappedPC;
    }

    const md = navigator.mediaDevices;
    if (md && md.getUserMedia) {
        const prevGUM = md.getUserMedia.bind(md);
        md.getUserMedia = async (constraints) => {
            const stream = await prevGUM(constraints);
            streams.push(stream);
            return stream;
        };
    }

    const isLiveKit = (ws) => ((ws.url || '').includes('livekit'));
    const isOpen = (ws) => ws.readyState === 0 || ws.readyState === 1;

    const state = () => ({
        sockets: sockets.map((ws) => ({
            url: (ws.url || '').split('?')[0],
            readyState: ws.readyState,
            livekit: isLiveKit(ws),
        })),
        openSockets: sockets.filter(isOpen).length,
        openLiveKitSockets: sockets.filter(
            (ws) => isLiveKit(ws) && isOpen(ws)
        ).length,
        openPeerConnections: pcs.filter(
            (pc) => pc.connectionState !== 'closed'
        ).length,
        liveLocalTracks: streams
            .flatMap((s) => s.getTracks())
            .filter((t) => t.readyState === 'live').length,
    });

    const leave = () => {
        const before = state();
        for (const s of streams) {
            for (const t of s.getTracks()) {
                try { t.stop(); } catch (e) {}
            }
        }
        for (const pc of pcs) {
            try { pc.close(); } catch (e) {}
        }
        for (const ws of sockets) {
            try {
                if (isOpen(ws)) ws.close(1000, 'e2e teardown: leaving room');
            } catch (e) {}
        }
        return { before, after: state() };
    };

    window.__emhRoomExit = { state, leave };
})();
"""


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
        # Room-exit outcome, filled in by close().
        self.left_room = False
        self.exit_report: dict = {}
        # Event loop the Playwright objects live on (for the
        # session-level safety net in conftest.py).
        self.loop: asyncio.AbstractEventLoop | None = None
        # LiveKit observation, recorded at join time.
        self.livekit_connections: list[str] = []
        self.livekit_opened: asyncio.Event | None = None
        self.livekit_active: asyncio.Event | None = None
        self.livekit_closed: asyncio.Event | None = None
        self._livekit_open_count = 0

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
        self._livekit_open_count += 1
        self.livekit_opened.set()
        websocket.on(
            "framesent", lambda payload: self.livekit_active.set()
        )
        websocket.on(
            "framereceived", lambda payload: self.livekit_active.set()
        )
        websocket.on("close", lambda _ws: self._on_livekit_closed())

    def _on_livekit_closed(self):
        self._livekit_open_count = max(0, self._livekit_open_count - 1)
        print(
            "LiveKit WebSocket closed "
            f"({self._livekit_open_count} still open)."
        )
        if self._livekit_open_count == 0 and self.livekit_closed:
            self.livekit_closed.set()

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
        # After INIT_SCRIPT on purpose: wraps the instrumented
        # constructors so teardown can close/verify every socket,
        # peer connection and local track the room opened.
        await self._context.add_init_script(ROOM_EXIT_INIT_SCRIPT)
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
            self.loop = asyncio.get_running_loop()
            page = await self._start_page()
            self._raw_page = page
            self.livekit_opened = asyncio.Event()
            self.livekit_active = asyncio.Event()
            self.livekit_closed = asyncio.Event()
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

    # ---------------------------------------------- room exit

    async def _click_end_interview(self, page) -> bool:
        """
        Use the app's OWN exit control first, so the server sees
        a normal participant leave (LiveKit leave request +
        socket.io end events) rather than a dropped connection.
        Best-effort: returns False when no such control exists.
        """

        # A native confirm() dialog must be accepted, or the
        # click would hang the page.
        def _accept(dialog):
            asyncio.ensure_future(dialog.accept())

        page.once("dialog", _accept)

        button = page.get_by_role("button", name=END_BUTTON_PATTERN).first
        if await button.count() == 0:
            print("Room exit: no End/Leave Interview control found.")
            return False
        await button.click(timeout=int(END_BUTTON_TIMEOUT_S * 1000))
        print("Room exit: clicked the app's End/Leave Interview control.")

        # In-app confirmation dialog (if any).
        confirm = page.get_by_role(
            "button", name=CONFIRM_BUTTON_PATTERN
        ).first
        try:
            await confirm.wait_for(state="visible", timeout=2_000)
            await confirm.click(timeout=2_000)
            print("Room exit: confirmed the leave dialog.")
        except Exception:
            pass
        return True

    async def _wait_for_exit(self, page, timeout_s: float) -> dict:
        """
        Poll the in-page exit state until no socket / peer
        connection / local track is live, or the time-box ends.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        state = {}
        while True:
            state = await page.evaluate("window.__emhRoomExit.state()")
            if (
                state.get("openSockets", 0) == 0
                and state.get("openPeerConnections", 0) == 0
                and state.get("liveLocalTracks", 0) == 0
            ):
                return state
            if loop.time() >= deadline:
                return state
            await asyncio.sleep(0.25)

    async def _leave_room(self, page) -> None:
        """
        EXPLICIT exit from the interview room, before the browser
        goes away:

          1. click the app's End/Leave Interview control
             (normal participant leave, server-side cleanup);
          2. let the app's own disconnect settle briefly;
          3. force-close whatever is still open (LiveKit +
             socket.io WebSockets, RTCPeerConnections, local
             mic/camera tracks) from inside the page;
          4. VERIFY nothing is left open and record the result
             in self.exit_report / self.left_room.

        Every step is time-boxed and exception-safe; teardown
        must never fail or hang the suite.
        """

        report: dict = {"clicked_end": False, "verified": False}
        self.exit_report = report

        try:
            report["clicked_end"] = await asyncio.wait_for(
                self._click_end_interview(page),
                timeout=END_BUTTON_TIMEOUT_S + 5,
            )
        except Exception as error:
            print(f"[WARNING] Room exit: End Interview click failed: {error}")

        if report["clicked_end"]:
            # Give the app's own leave a chance to complete.
            try:
                report["after_app_leave"] = await asyncio.wait_for(
                    self._wait_for_exit(page, APP_LEAVE_SETTLE_S),
                    timeout=APP_LEAVE_SETTLE_S + 5,
                )
            except Exception as error:
                print(f"[WARNING] Room exit: settle wait failed: {error}")

        try:
            result = await asyncio.wait_for(
                page.evaluate("window.__emhRoomExit.leave()"),
                timeout=5,
            )
            report["forced"] = result
            before = result.get("before", {})
            print(
                "Room exit: explicit disconnect issued "
                f"(open sockets before={before.get('openSockets')}, "
                f"peer connections={before.get('openPeerConnections')}, "
                f"live local tracks={before.get('liveLocalTracks')})."
            )
        except Exception as error:
            print(f"[WARNING] Room exit: explicit disconnect failed: {error}")

        try:
            final = await asyncio.wait_for(
                self._wait_for_exit(page, EXIT_VERIFY_TIMEOUT_S),
                timeout=EXIT_VERIFY_TIMEOUT_S + 5,
            )
            report["final"] = final
            report["verified"] = (
                final.get("openSockets", 0) == 0
                and final.get("openPeerConnections", 0) == 0
                and final.get("liveLocalTracks", 0) == 0
            )
        except Exception as error:
            print(f"[WARNING] Room exit: verification failed: {error}")

        # Cross-check with Playwright's own view of the LiveKit
        # signalling socket(s) observed at join time.
        if self.livekit_closed is not None and self._livekit_open_count:
            try:
                await asyncio.wait_for(self.livekit_closed.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
        report["livekit_sockets_still_open"] = self._livekit_open_count

        self.left_room = bool(
            report["verified"] and self._livekit_open_count == 0
        )
        if self.left_room:
            print(
                "Room exit VERIFIED: LiveKit/socket.io disconnected, "
                "peer connections closed, local tracks stopped."
            )
        else:
            print(
                "[WARNING] Room exit NOT fully verified: "
                f"{report.get('final')} / LiveKit sockets still open: "
                f"{self._livekit_open_count}. Falling through to "
                "navigate-away + browser teardown."
            )

    async def close(self):
        """
        Explicit room exit + graceful leave + browser teardown,
        exactly once (idempotent). Called by the shared_room
        fixture after the LAST consumer (pass, fail or skip), by
        the session-level safety net in conftest.py, or
        immediately on a failed join.

        Order: leave the room (End Interview click, explicit
        socket/PC/track close, verified) -> navigate away so the
        app's unload handlers flush -> close context/browser/
        playwright. Each stage is time-boxed and exception-safe.
        """

        if self.closed:
            return
        self.closed = True

        from pages.interview_launch import graceful_leave_interview_room

        page = self.page or self._raw_page
        page_alive = page is not None and not page.is_closed()

        if page_alive and self.page is not None:
            # Only a fully joined page has a room to leave.
            try:
                await self._leave_room(page)
            except Exception as error:
                print(f"[WARNING] Shared-room explicit exit failed: {error}")

        if page_alive:
            try:
                await asyncio.wait_for(
                    graceful_leave_interview_room(page, log=print),
                    timeout=NAVIGATE_AWAY_TIMEOUT_S,
                )
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
                await asyncio.wait_for(
                    close(), timeout=BROWSER_TEARDOWN_TIMEOUT_S
                )
            except Exception as error:
                print(f"[WARNING] Shared-room {label} teardown: {error}")

        print(
            "Shared room closed (explicit room exit + graceful leave "
            "+ browser teardown ran exactly once, after the last "
            "room-join test)."
        )
