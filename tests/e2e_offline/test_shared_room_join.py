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

class FakePage:
    def __init__(self):
        self._closed = False

    def is_closed(self):
        return self._closed

    def on(self, *_args, **_kwargs):
        pass


class Recorder:
    def __init__(self):
        self.resolves = []
        self.launches = []
        self.marks = []
        self.leaves = []


@pytest.fixture
def rig(monkeypatch):
    """
    A SharedRoomJoin wired to fakes: no Playwright, no policy
    side effects. Returns (manager, recorder, fake_page).
    """

    manager = sr.SharedRoomJoin()
    recorder = Recorder()
    fake_page = FakePage()
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
