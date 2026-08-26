"""
Session isolation for the E2E suite.

Two separate FRESH interview sessions per run:

  INTERVIEW_URL  -> reserved EXCLUSIVELY for the full-interview
                    evaluation (test_bot_responsiveness).
  EMH_TESTS_URL  -> every other E2E test (setup screens, interview
                    room, LiveKit, socket, permissions, launch).

The autouse fixture below resolves which session a test module
owns and holds the cross-process session lock for the test's
duration, so two harness processes can never drive the same
session concurrently (the app's one-tab lock would eject the
first). Tests that cannot get a URL still fail/skip through their
own session-layer messages - the fixture only enforces the lock.
"""

import pytest
import pytest_asyncio

from config.interview_session import (
    InterviewSessionError,
    decode_interview_claims,
    get_interview_url,
    get_tests_url,
    interview_url_configured,
    tests_url_configured,
)
from config.session_lock import acquire_session_lock
from tests.e2e.shared_room import SharedRoomJoin, is_room_join_nodeid


FULL_INTERVIEW_MODULES = {"test_bot_responsiveness"}


def session_role_for(module_name: str) -> str:
    short = module_name.rsplit(".", 1)[-1]
    return "full_interview" if short in FULL_INTERVIEW_MODULES else "tests"


@pytest.fixture(autouse=True)
def _own_session_lock(request):
    role = session_role_for(request.module.__name__)
    try:
        if role == "full_interview":
            url = get_interview_url() if interview_url_configured() else None
        else:
            url = get_tests_url() if tests_url_configured() else None
        claims = decode_interview_claims(url) if url else None
    except InterviewSessionError:
        claims = None

    if claims is None:
        yield  # no URL: the test reports the missing config itself
        return

    try:
        with acquire_session_lock(claims, request.node.nodeid):
            yield
    except InterviewSessionError as error:
        pytest.fail(str(error))


# ============================================================
# Shared room join (tests/e2e/shared_room.py)
#
# The three room-joining tests (continue_to_interview /
# interview_room / livekit_connection) run on ONE shared
# EMH_TESTS_URL room join: the session budget is one joinable
# room per run, so per-test joins made two of them skip with
# SESSION ALREADY CONSUMED deterministically. The consumers are
# marked asyncio(loop_scope="session") so they share the event
# loop the Playwright objects live on.
# ============================================================


def pytest_collection_modifyitems(session, config, items):
    """
    Keep the three room-join consumers CONTIGUOUS, at the end of
    the collected E2E block (the order scripts/run_all_tests.py
    already imposes). Under pytest's default alphabetical order
    the shared live room would otherwise stay open across the
    setup-screen tests collected between them - and the app's
    one-tab lock would eject/deadlock those tests on the same
    EMH_TESTS_URL session.
    """

    room = [item for item in items if is_room_join_nodeid(item.nodeid)]
    if len(room) <= 1:
        return
    room_ids = {item.nodeid for item in room}
    rest = [item for item in items if item.nodeid not in room_ids]
    e2e_positions = [
        index
        for index, item in enumerate(rest)
        if "tests/e2e/" in item.nodeid.replace("\\", "/")
    ]
    insert_at = (
        e2e_positions[-1] + 1 if e2e_positions else len(rest)
    )
    items[:] = rest[:insert_at] + room + rest[insert_at:]


@pytest.fixture(scope="session")
def _shared_room_state(request):
    """
    One SharedRoomJoin manager per pytest session, plus the set
    of COLLECTED room-join consumers still to run - so the room
    is closed exactly once, right after the last of them, and is
    never left open through the later (judging) suites.
    """

    pending = {
        item.nodeid
        for item in request.session.items
        if is_room_join_nodeid(item.nodeid)
    }
    return SharedRoomJoin(), pending


@pytest_asyncio.fixture(loop_scope="session")
async def shared_room(request, _shared_room_state):
    """
    The shared room-join manager. Function-scoped on the SESSION
    event loop: every consumer test gets the same manager (and
    live page); teardown after the last consumer - pass, fail or
    skip - gracefully leaves the room and closes the browser.
    """

    manager, pending = _shared_room_state
    yield manager
    pending.discard(request.node.nodeid)
    if not pending:
        await manager.close()
