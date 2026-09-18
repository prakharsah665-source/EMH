"""
Offline verification of the shared room join
(tests/e2e/shared_room.SharedRoomJoin) - no browser, no network.

Proves the session-budget fix behaves as designed:

  * the EMH_TESTS_URL room is joined exactly ONCE (first
    consumer), later consumers ATTACH to the same live page;
  * the join goes through the UNCHANGED session policy - a
    pre-consumed session still skips every consumer (the
    safeguard is re-checked per consumer, not cached);
  * a failed join produces one authoritative failure and
    classified upstream SKIPs for the other consumers (no
    re-join of a possibly half-consumed session);
  * teardown (graceful leave + browser close) runs exactly
    once, no matter how often close() is called;
  * the conftest fixture's nodeid filter recognises exactly
    the three room-join test files.
"""

import pytest

from tests.e2e import shared_room as sr


# ------------------------------------------------------------
# Fakes
# ------------------------------------------------------------

class FakeLocator:
    """Minimal role-locator fake: one End Interview button."""

    def __init__(self, page, present):
        self._page = page
        self._present = present

    @property
    def first(self):
        return self

    async def count(self):
        return 1 if self._present else 0

    async def click(self, timeout=None):
        self._page.recorder.events.append("click_end")
        self._page.app_left = True

    async def wait_for(self, state=None, timeout=None):
        raise TimeoutError("no confirm dialog")


class FakePage:
    """
    Simulates the in-page room-exit hooks: the app's End button
    closes everything by itself only when `app_leave_works`;
    otherwise the explicit __emhRoomExit.leave() must do it.
    """

    def __init__(self, recorder=None, has_end_button=True,
                 app_leave_works=True):
        self._closed = False
        self.recorder = recorder or Recorder()
        self.has_end_button = has_end_button
        self.app_leave_works = app_leave_works
        self.app_left = False
        self.open = {"sockets": 2, "pcs": 1, "tracks": 1}

    def is_closed(self):
        return self._closed

    def on(self, *_args, **_kwargs):
        pass

    def once(self, *_args, **_kwargs):
        pass

    def get_by_role(self, role, name=None):
        return FakeLocator(self, self.has_end_button)

    def _state(self):
        if self.app_left and self.app_leave_works:
            self.open = {"sockets": 0, "pcs": 0, "tracks": 0}
        return {
            "openSockets": self.open["sockets"],
            "openLiveKitSockets": self.open["sockets"],
            "openPeerConnections": self.open["pcs"],
            "liveLocalTracks": self.open["tracks"],
        }

    async def evaluate(self, expression):
        if "state()" in expression:
            return self._state()
        if "leave()" in expression:
            before = self._state()
            self.recorder.events.append("explicit_leave")
            self.open = {"sockets": 0, "pcs": 0, "tracks": 0}
            return {"before": before, "after": self._state()}
        raise AssertionError(f"unexpected evaluate: {expression}")


class Recorder:
    def __init__(self):
        self.resolves = []
        self.launches = []
        self.marks = []
        self.leaves = []
        self.events = []


@pytest.fixture
def rig(monkeypatch):
    """
    A SharedRoomJoin wired to fakes: no Playwright, no policy
    side effects. Returns (manager, recorder, fake_page).
    """

    manager = sr.SharedRoomJoin()
    recorder = Recorder()
    fake_page = FakePage(recorder)
    fake_claims = object()

    async def fake_start_page():
        return fake_page

    def fake_resolve(test_name):
        recorder.resolves.append(test_name)
        return "https://fake.invalid/interview?token=x", fake_claims

    async def fake_launch(page, *, log=None, interview_url=None):
        recorder.launches.append(interview_url)

    def fake_mark(claims, reason):
        assert claims is fake_claims
        recorder.marks.append(reason)

    async def fake_leave(page, log=None):
        recorder.leaves.append(page)
        recorder.events.append("navigate_away")

    monkeypatch.setattr(manager, "_start_page", fake_start_page)

    import pages.interview_launch as launch_module
    import tests.e2e.session_policy as policy_module

    monkeypatch.setattr(
        policy_module, "resolve_room_session", fake_resolve
    )
    monkeypatch.setattr(policy_module, "mark_room_joined", fake_mark)
    monkeypatch.setattr(
        launch_module, "launch_into_interview_room", fake_launch
    )
    monkeypatch.setattr(
        launch_module, "graceful_leave_interview_room", fake_leave
    )

    return manager, recorder, fake_page


# ------------------------------------------------------------
# Join once, attach afterwards
# ------------------------------------------------------------

async def test_first_consumer_joins_then_others_attach(rig):
    manager, recorder, fake_page = rig

    page1 = await manager.ensure_joined("test_continue_to_interview")
    page2 = await manager.ensure_joined("test_interview_room")
    page3 = await manager.ensure_joined("test_livekit_connection")

    assert page1 is fake_page
    assert page2 is fake_page and page3 is fake_page

    # Exactly ONE policy resolution, ONE launch, ONE ledger
    # entry - attaching consumers never re-resolve (the session
    # is consumed by then; re-resolving would falsely skip).
    assert recorder.resolves == ["test_continue_to_interview"]
    assert len(recorder.launches) == 1
    assert recorder.marks == ["test_continue_to_interview shared"]
    assert manager.joined_by == "test_continue_to_interview"


async def test_close_runs_exactly_once(rig):
    manager, recorder, fake_page = rig

    await manager.ensure_joined("test_continue_to_interview")
    await manager.close()
    await manager.close()  # idempotent

    assert recorder.leaves == [fake_page]
    assert manager.closed


# ------------------------------------------------------------
# Explicit room exit before the browser goes away
# ------------------------------------------------------------

async def test_close_exits_room_explicitly_before_navigating_away(rig):
    manager, recorder, fake_page = rig

    await manager.ensure_joined("test_continue_to_interview")
    await manager.close()

    # Order: app End button -> explicit socket/PC/track close ->
    # navigate away (unload flush) -> browser teardown.
    assert recorder.events == ["click_end", "explicit_leave", "navigate_away"]
    assert manager.left_room is True
    assert manager.exit_report["clicked_end"] is True
    assert manager.exit_report["verified"] is True
    assert manager.exit_report["final"]["openSockets"] == 0
    assert manager.exit_report["final"]["openPeerConnections"] == 0
    assert manager.exit_report["final"]["liveLocalTracks"] == 0


async def test_explicit_disconnect_covers_app_leave_that_does_nothing(
    rig, monkeypatch
):
    manager, recorder, fake_page = rig
    fake_page.app_leave_works = False
    monkeypatch.setattr(sr, "APP_LEAVE_SETTLE_S", 0.3)

    await manager.ensure_joined("test_continue_to_interview")
    await manager.close()

    # The app's End button left everything open; the explicit
    # in-page disconnect still closed it all and was verified.
    assert recorder.events == ["click_end", "explicit_leave", "navigate_away"]
    assert manager.exit_report["forced"]["before"]["openSockets"] == 2
    assert manager.exit_report["forced"]["after"]["openSockets"] == 0
    assert manager.left_room is True


async def test_exit_without_end_button_still_disconnects(rig):
    manager, recorder, fake_page = rig
    fake_page.has_end_button = False

    await manager.ensure_joined("test_continue_to_interview")
    await manager.close()

    assert recorder.events == ["explicit_leave", "navigate_away"]
    assert manager.exit_report["clicked_end"] is False
    assert manager.left_room is True


async def test_exit_runs_even_after_a_consumer_failed(rig):
    """
    The shared_room fixture tears down after the last consumer
    whatever its outcome; the manager must exit the room then
    regardless of how the tests went.
    """

    manager, recorder, fake_page = rig

    await manager.ensure_joined("test_continue_to_interview")
    with pytest.raises(AssertionError):
        assert False, "simulated consumer failure"
    await manager.close()

    assert recorder.events == ["click_end", "explicit_leave", "navigate_away"]
    assert manager.closed and manager.left_room


async def test_exit_is_exception_safe_when_page_hooks_break(rig, monkeypatch):
    manager, recorder, fake_page = rig

    async def broken_evaluate(expression):
        raise RuntimeError("page gone")

    await manager.ensure_joined("test_continue_to_interview")
    monkeypatch.setattr(fake_page, "evaluate", broken_evaluate)
    await manager.close()  # must not raise

    assert manager.closed
    assert manager.left_room is False
    # Navigate-away + browser teardown still ran.
    assert recorder.leaves == [fake_page]


# ------------------------------------------------------------
# Policy safeguards pass through unchanged
# ------------------------------------------------------------

async def test_preconsumed_session_still_skips_every_consumer(
    rig, monkeypatch
):
    manager, recorder, _fake_page = rig

    import tests.e2e.session_policy as policy_module

    def consumed_resolve(test_name):
        recorder.resolves.append(test_name)
        pytest.skip(
            f"SESSION ALREADY CONSUMED - {test_name} cannot join."
        )

    monkeypatch.setattr(
        policy_module, "resolve_room_session", consumed_resolve
    )

    for name in (
        "test_continue_to_interview",
        "test_interview_room",
        "test_livekit_connection",
    ):
        with pytest.raises(pytest.skip.Exception, match="ALREADY CONSUMED"):
            await manager.ensure_joined(name)

    # The safeguard fired PER consumer - never cached away.
    assert len(recorder.resolves) == 3
    assert recorder.launches == [] and recorder.marks == []


async def test_failed_join_gives_one_failure_then_upstream_skips(
    rig, monkeypatch
):
    manager, recorder, fake_page = rig

    import pages.interview_launch as launch_module

    async def broken_launch(page, *, log=None, interview_url=None):
        raise RuntimeError("LAUNCH STEP FAILED: enter interview room")

    monkeypatch.setattr(
        launch_module, "launch_into_interview_room", broken_launch
    )

    # One authoritative failure in the joining test...
    with pytest.raises(RuntimeError, match="LAUNCH STEP FAILED"):
        await manager.ensure_joined("test_continue_to_interview")

    assert manager.join_failed_in == "test_continue_to_interview"
    # ...the half-driven page was already left/closed...
    assert manager.closed and recorder.leaves == [fake_page]
    # ...no ledger entry for a join that never completed...
    assert recorder.marks == []

    # ...and the other consumers SKIP as upstream instead of
    # re-driving the session.
    for name in ("test_interview_room", "test_livekit_connection"):
        with pytest.raises(
            pytest.skip.Exception, match="FAILED UPSTREAM"
        ):
            await manager.ensure_joined(name)
    assert len(recorder.resolves) == 1  # no second resolution/join


async def test_dead_page_fails_instead_of_silently_rejoining(rig):
    manager, _recorder, fake_page = rig

    await manager.ensure_joined("test_continue_to_interview")
    fake_page._closed = True

    with pytest.raises(
        pytest.fail.Exception, match="SHARED ROOM PAGE CLOSED"
    ):
        await manager.ensure_joined("test_interview_room")


# ------------------------------------------------------------
# Fixture bookkeeping helper
# ------------------------------------------------------------

def test_room_join_nodeid_filter():
    assert sr.is_room_join_nodeid(
        "tests/e2e/test_continue_to_interview.py::test_continue_to_interview"
    )
    assert sr.is_room_join_nodeid(
        "tests/e2e/test_interview_room.py::test_interview_room"
    )
    assert sr.is_room_join_nodeid(
        "tests/e2e/test_livekit_connection.py::test_livekit_connection"
    )
    assert not sr.is_room_join_nodeid(
        "tests/e2e/test_bot_responsiveness.py::test_bot_responsiveness"
    )
    assert not sr.is_room_join_nodeid(
        "tests/e2e/test_socket_connection.py::test_socket_connection"
    )
