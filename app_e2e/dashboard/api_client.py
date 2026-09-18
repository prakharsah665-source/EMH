"""
Thin client for the dashboard-data API.

The web app issues two POSTs when the dashboard opens (verified by
intercepting the page's own traffic):

  POST <Dashboard_API_URL>/getDashboardData
       {"variables": {"candidatesWhere": {"is_individual_user": {"_eq": false}},
                      "jobsWhere": {}, "campaignWhere": {}, "creditsWhere": {}}}
  POST <Dashboard_API_URL>/GetLookup
       {"variables": {"hasCompanyId": false, "includeGlobalJobRoles": true}}

Both carry "Authorization: Bearer <idToken>". The harness sends the
same bodies so its independent read is directly comparable with the
payload the page rendered.
"""

import copy
import time
from dataclasses import dataclass
from typing import Any

import requests

from app_e2e.dashboard.config import DashboardConfig


DASHBOARD_DATA_OPERATION = "getDashboardData"
LOOKUP_OPERATION = "GetLookup"

DEFAULT_DASHBOARD_VARIABLES: dict = {
    "candidatesWhere": {"is_individual_user": {"_eq": False}},
    "jobsWhere": {},
    "campaignWhere": {},
    "creditsWhere": {},
}

DEFAULT_LOOKUP_VARIABLES: dict = {
    "hasCompanyId": False,
    "includeGlobalJobRoles": True,
}


@dataclass
class ApiResponse:
    url: str
    status_code: int
    elapsed_s: float
    content_type: str
    text: str
    json: Any

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def snippet(self, limit: int = 200) -> str:
        return " ".join(self.text.split())[:limit]


class DashboardApiClient:
    def __init__(
        self,
        config: DashboardConfig,
        token: str,
        session: requests.Session | None = None,
        timeout_s: float = 30,
    ) -> None:
        self.config = config
        self.token = token
        self.http = session or requests.Session()
        self.timeout_s = timeout_s

    def operation_url(self, operation: str) -> str:
        return f"{self.config.dashboard_api_url}/{operation}"

    def post(self, operation: str, variables: dict) -> ApiResponse:
        url = self.operation_url(operation)
        started = time.perf_counter()
        response = self.http.post(
            url,
            json={"variables": copy.deepcopy(variables)},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
            },
            timeout=self.timeout_s,
        )
        elapsed = time.perf_counter() - started
        try:
            body = response.json()
        except ValueError:
            body = None
        return ApiResponse(
            url=url,
            status_code=response.status_code,
            elapsed_s=elapsed,
            content_type=response.headers.get("content-type", ""),
            text=response.text,
            json=body,
        )

    def dashboard_data(self, variables: dict | None = None) -> ApiResponse:
        return self.post(
            DASHBOARD_DATA_OPERATION,
            variables or DEFAULT_DASHBOARD_VARIABLES,
        )

    def lookup(self, variables: dict | None = None) -> ApiResponse:
        return self.post(
            LOOKUP_OPERATION, variables or DEFAULT_LOOKUP_VARIABLES
        )


def token_accepted(
    config: DashboardConfig, token: str
) -> tuple[bool, str]:
    """Probe the API with a token: (accepted, 'HTTP <code> <snippet>')."""

    try:
        response = DashboardApiClient(config, token).dashboard_data()
    except requests.RequestException as error:
        return False, f"request failed: {error}"
    accepted = response.ok and isinstance(response.json, dict)
    return accepted, f"HTTP {response.status_code} {response.snippet(120)}"
