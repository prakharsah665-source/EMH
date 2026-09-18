"""
Dashboard data validation.

Pipeline (see tests/app_e2e/conftest.py for the wiring):

  config    -> every URL / credential / token from .env
  auth      -> ACCESS_TOKEN if still valid, else POST /auth/sign-in
  api       -> POST <Dashboard_API_URL>/getDashboardData (+ GetLookup)
  schema    -> required fields, data types, value ranges, cross-field
               consistency of the API payload
  expected  -> the exact numbers the dashboard must render, derived
               from the payload with the same formulas the web app uses
  ui        -> Playwright: log in, intercept the payload the page
               received, read every rendered KPI / breakdown / chart
  validator -> UI vs expected comparisons
  report    -> PASS/FAIL HTML + JSON report
"""
