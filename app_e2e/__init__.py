"""
Application E2E validation (independent of the interview flow).

This package validates the EMH web application itself - the
recruiter/admin dashboard and, later, other application screens -
against its backend APIs. It shares nothing with the interview
evaluation flow (tests/e2e, evaluation/, simulator/): no interview
session, no one-tab lock, no transcript capture.

Sub-packages:

  app_e2e.dashboard  Dashboard data validation - logs in, reads the
                     dashboard-data API, derives the values the UI
                     must display and compares them with the live
                     dashboard page (PASS/FAIL report).
"""
