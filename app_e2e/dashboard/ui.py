"""
Playwright capture of the live dashboard page.

The browser logs in through the real login form (LOGIN_URL with
LOGIN_EMAIL / LOGIN_PASSWORD), lands on /dashboard, and the harness:

  * intercepts the getDashboardData / GetLookup responses the PAGE
    itself received (so UI-vs-API comparisons use the exact payload
    the page rendered, immune to real-time drift),
  * waits for the KPI count-up animation to settle,
  * reads every rendered number: KPI cards, stage breakdown rows,
    pipeline bar labels, donut totals + legends, and the
    "View by company" credits dialog,
  * saves a full-page screenshot as evidence.

Nothing here knows the expected values - it only observes.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from playwright.async_api import (
    Page,
    TimeoutError as PlaywrightTimeout,
    async_playwright,
)

from app_e2e.dashboard.config import DashboardConfig, mask_email


NUMBER_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+$|^-?\d+$")
LOGIN_BUTTON_RE = re.compile(r"^\s*(log ?in|sign ?in)\s*$", re.IGNORECASE)
DASHBOARD_URL_RE = re.compile(r"/dashboard")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# The widget headings sit at different depths inside their cards;
# climb to the nearest card container instead of the direct parent.
CARD_ANCESTOR_XPATH = "xpath=ancestor::div[contains(@class, 'rounded-xl')][1]"

KPI_SETTLE_READS = 3
KPI_SETTLE_INTERVAL_S = 0.3
KPI_SETTLE_MAX_S = 20.0


class DashboardUiError(RuntimeError):
    """The page could not be driven to a readable dashboard."""


def parse_number(text: str | None) -> int | None:
    if text is None:
        return None
    cleaned = text.strip().replace(" ", "").replace("\xa0", "")
    if not NUMBER_RE.match(cleaned):
        return None
    return int(cleaned.replace(",", ""))


def redact(text: str, limit: int = 600) -> str:
    return EMAIL_RE.sub("<email>", " ".join(text.split()))[:limit]


@dataclass
class KpiCard:
    title: str
    value_text: str
    subtitle: str | None
    badge: str | None

    @property
    def value(self) -> int | None:
        return parse_number(self.value_text)


@dataclass
class DonutCard:
    title: str
    total_text: str | None
    legend: list[tuple[str, str]]


@dataclass
class UiSnapshot:
    page_url: str
    page_title: str
    kpi_cards: dict[str, KpiCard]
    stage_rows: list[tuple[str, str, str | None]]
    donuts: dict[str, DonutCard]
    pipeline_labels: list[str]
    credits_button_present: bool
    credits_by_company: list[tuple[str, str]] | None
    dashboard_requests: list[dict]
    intercepted_payload: dict | None
    lookup_payload: dict | None
    user_data: dict
    roles: list[str]
    screenshot_path: str | None
    login_elapsed_s: float
    render_elapsed_s: float
    main_text: str
    log: list[str] = field(default_factory=list)

    @property
    def intercepted_request(self) -> dict | None:
        return self.dashboard_requests[-1] if self.dashboard_requests else None


async def _page_hint(page: Page) -> str:
    try:
        body = await page.locator("body").inner_text()
    except Exception as error:  # noqa: BLE001
        return f"(page unreadable: {error})"
    return f"URL {page.url} | text: {redact(body)}"


async def _login(page: Page, config: DashboardConfig, log: Callable) -> None:
    log(f"Opening login page {config.login_url}")
    await page.goto(
        config.login_url,
        wait_until="domcontentloaded",
        timeout=config.timeout_ms,
    )

    email = page.locator(
        'input[type="email"], input[name="email"], '
        'input[autocomplete="username"]'
    ).first
    password = page.locator('input[type="password"]').first
    try:
        await email.wait_for(state="visible", timeout=config.timeout_ms)
        await password.wait_for(state="visible", timeout=config.timeout_ms)
    except PlaywrightTimeout:
        raise DashboardUiError(
            "Login form (email + password inputs) not found on "
            f"LOGIN_URL. {await _page_hint(page)}"
        )

    await email.fill(config.login_email)
    await password.fill(config.login_password)

    button = page.get_by_role("button", name=LOGIN_BUTTON_RE)
    if await button.count():
        await button.first.click()
    else:
        await password.press("Enter")
    log(f"Submitted credentials for {mask_email(config.login_email)}")

    try:
        await page.wait_for_url(DASHBOARD_URL_RE, timeout=config.timeout_ms)
    except PlaywrightTimeout:
        raise DashboardUiError(
            "Login did not reach /dashboard within "
            f"{config.timeout_ms / 1000:.0f}s - wrong credentials, a "
            "new-password challenge, or a changed login flow. "
            f"{await _page_hint(page)}"
        )
    log(f"Logged in, landed on {page.url}")


async def _read_kpi_cards(page: Page) -> dict[str, KpiCard]:
    grid = page.locator("div.emh-dashboard > div.grid").first
    cards = grid.locator(":scope > div")
    count = await cards.count()
    result: dict[str, KpiCard] = {}
    for index in range(count):
        card = cards.nth(index)
        title = (await card.locator("p").first.inner_text()).strip()
        value_loc = card.locator("p.font-bold")
        value_text = (
            (await value_loc.first.inner_text()).strip()
            if await value_loc.count()
            else ""
        )
        subtitle_loc = card.locator("p.text-xs")
        subtitle = (
            (await subtitle_loc.first.inner_text()).strip()
            if await subtitle_loc.count()
            else None
        )
        badge_loc = card.locator("span.rounded-full")
        badge = (
            (await badge_loc.first.inner_text()).strip()
            if await badge_loc.count()
            else None
        )
        result[title.lower()] = KpiCard(
            title=title,
            value_text=value_text,
            subtitle=subtitle or None,
            badge=badge or None,
        )
    return result


async def _wait_for_settled_kpis(
    page: Page, log: Callable
) -> dict[str, KpiCard]:
    """
    The KPI values count up with a spring animation; wait until
    KPI_SETTLE_READS consecutive reads are identical.
    """

    deadline = time.monotonic() + KPI_SETTLE_MAX_S
    previous: list[str] | None = None
    stable = 0
    cards: dict[str, KpiCard] = {}
    while time.monotonic() < deadline:
        cards = await _read_kpi_cards(page)
        signature = [c.value_text for c in cards.values()]
        if cards and signature == previous:
            stable += 1
            if stable >= KPI_SETTLE_READS - 1:
                log(
                    "KPI values settled: "
                    + ", ".join(
                        f"{c.title}={c.value_text}" for c in cards.values()
                    )
                )
                return cards
        else:
            stable = 0
        previous = signature
        await asyncio.sleep(KPI_SETTLE_INTERVAL_S)
    log(
        "WARNING: KPI values did not settle within "
        f"{KPI_SETTLE_MAX_S:.0f}s; using the last read."
    )
    return cards


async def _read_stage_rows(page: Page) -> list[tuple[str, str, str | None]]:
    heading = page.locator("h2", has_text="Stage breakdown")
    if not await heading.count():
        return []
    card = heading.first.locator(CARD_ANCESTOR_XPATH)
    items = card.locator("ul > li")
    rows: list[tuple[str, str, str | None]] = []
    for index in range(await items.count()):
        item = items.nth(index)
        label_loc = item.locator("span.truncate")
        if await label_loc.count():
            label = (await label_loc.first.inner_text()).strip()
        else:
            label = (await item.locator("span").first.inner_text()).strip()
        value_loc = item.locator("span.font-semibold")
        value = (
            (await value_loc.first.inner_text()).strip()
            if await value_loc.count()
            else ""
        )
        pct_loc = item.locator("span.w-8")
        pct = (
            (await pct_loc.first.inner_text()).strip()
            if await pct_loc.count()
            else None
        )
        rows.append((label, value, pct))
    return rows


def parse_donut_text(title: str, text: str) -> DonutCard:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    total: str | None = None
    legend: list[tuple[str, str]] = []
    if "TOTAL" in lines:
        at = lines.index("TOTAL")
        if at > 0:
            total = lines[at - 1]
        rest = lines[at + 1:]
        index = 0
        while index + 1 < len(rest):
            name, value = rest[index], rest[index + 1]
            if parse_number(value) is not None and parse_number(name) is None:
                legend.append((name, value))
                index += 2
            else:
                index += 1
    return DonutCard(title=title, total_text=total, legend=legend)


async def _read_donuts(page: Page) -> dict[str, DonutCard]:
    donuts: dict[str, DonutCard] = {}
    for title in ("Communications", "Open rate", "Response mix"):
        heading = page.locator("h3", has_text=title)
        if not await heading.count():
            continue
        card = heading.first.locator(CARD_ANCESTOR_XPATH)
        donuts[title] = parse_donut_text(title, await card.inner_text())
    return donuts


async def _read_pipeline_labels(page: Page) -> list[str]:
    heading = page.locator("h2", has_text="Interview pipeline")
    if not await heading.count():
        return []
    card = heading.first.locator(CARD_ANCESTOR_XPATH)
    # SVG elements have no innerText - read textContent instead.
    texts = await card.locator("svg text").all_text_contents()
    return [t.strip() for t in texts if parse_number(t.strip()) is not None]


async def _read_credits_dialog(
    page: Page, config: DashboardConfig, log: Callable
) -> tuple[bool, list[tuple[str, str]] | None]:
    button = page.get_by_role(
        "button", name=re.compile(r"view by company", re.IGNORECASE)
    )
    if not await button.count():
        return False, None
    await button.first.click()
    dialog = page.get_by_role("dialog").first
    try:
        await dialog.wait_for(state="visible", timeout=config.timeout_ms)
        rows = dialog.locator("tbody tr")
        await rows.first.wait_for(state="visible", timeout=10_000)
    except PlaywrightTimeout:
        await page.keyboard.press("Escape")
        return True, []
    entries: list[tuple[str, str]] = []
    for index in range(await rows.count()):
        cells = await rows.nth(index).locator("td").all_inner_texts()
        if len(cells) >= 2:
            entries.append((cells[0].strip(), cells[1].strip()))
    await page.keyboard.press("Escape")
    try:
        await dialog.wait_for(state="hidden", timeout=5_000)
    except PlaywrightTimeout:
        pass
    log(f"Credits by company dialog: {len(entries)} rows")
    return True, entries


async def capture_dashboard(
    config: DashboardConfig,
    artifacts_dir: Path,
    log: Callable[[str], None] = print,
) -> UiSnapshot:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def _log(message: str) -> None:
        lines.append(message)
        log(message)

    dashboard_requests: list[dict] = []
    lookup_holder: dict = {"payload": None}
    payload_ready = asyncio.Event()

    async def on_response(response) -> None:
        url = response.url
        path = url.split("?", 1)[0].rstrip("/")
        if path.endswith("/getDashboardData"):
            try:
                body = await response.json()
            except Exception:  # noqa: BLE001
                body = None
            request = response.request
            try:
                post = json.loads(request.post_data or "null")
            except ValueError:
                post = request.post_data
            dashboard_requests.append(
                {
                    "url": url,
                    "status": response.status,
                    "authorized": bool(request.headers.get("authorization")),
                    "variables": post,
                    "payload": body,
                }
            )
            _log(f"Intercepted getDashboardData -> HTTP {response.status}")
            payload_ready.set()
        elif path.endswith("/GetLookup"):
            try:
                lookup_holder["payload"] = await response.json()
            except Exception:  # noqa: BLE001
                lookup_holder["payload"] = None

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=config.headless)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 1000}
        )
        page = await context.new_page()
        page.on("response", on_response)

        try:
            started = time.perf_counter()
            await _login(page, config, _log)
            login_elapsed = time.perf_counter() - started

            render_started = time.perf_counter()
            try:
                await asyncio.wait_for(
                    payload_ready.wait(), timeout=config.timeout_ms / 1000
                )
            except asyncio.TimeoutError:
                raise DashboardUiError(
                    "The dashboard page never called getDashboardData "
                    f"within {config.timeout_ms / 1000:.0f}s. "
                    f"{await _page_hint(page)}"
                )

            try:
                await page.locator("div.emh-dashboard").first.wait_for(
                    state="visible", timeout=config.timeout_ms
                )
            except PlaywrightTimeout:
                raise DashboardUiError(
                    "Dashboard content (div.emh-dashboard) did not render. "
                    f"{await _page_hint(page)}"
                )

            kpi_cards = await _wait_for_settled_kpis(page, _log)
            # Charts and dialogs are rendered after the KPI grid; a
            # short networkidle wait keeps the reads deterministic.
            try:
                await page.wait_for_load_state("networkidle", timeout=10_000)
            except PlaywrightTimeout:
                pass

            stage_rows = await _read_stage_rows(page)
            donuts = await _read_donuts(page)
            pipeline_labels = await _read_pipeline_labels(page)
            render_elapsed = time.perf_counter() - render_started

            main_loc = page.locator("main")
            main_text = (
                await main_loc.first.inner_text()
                if await main_loc.count()
                else await page.locator("body").inner_text()
            )

            screenshot = artifacts_dir / "dashboard_validation.png"
            await page.screenshot(path=str(screenshot), full_page=True)

            button_present, credits_rows = await _read_credits_dialog(
                page, config, _log
            )

            raw_user = await page.evaluate(
                "() => localStorage.getItem('userData')"
            )
            user_data: dict = {}
            if raw_user:
                try:
                    user_data = json.loads(raw_user)
                except ValueError:
                    user_data = {}
            user_data.pop("token", None)
            roles = [
                str(role) for role in (user_data.get("roles") or [])
            ]

            snapshot = UiSnapshot(
                page_url=page.url,
                page_title=await page.title(),
                kpi_cards=kpi_cards,
                stage_rows=stage_rows,
                donuts=donuts,
                pipeline_labels=pipeline_labels,
                credits_button_present=button_present,
                credits_by_company=credits_rows,
                dashboard_requests=dashboard_requests,
                intercepted_payload=(
                    dashboard_requests[-1]["payload"]
                    if dashboard_requests
                    else None
                ),
                lookup_payload=lookup_holder["payload"],
                user_data=user_data,
                roles=roles,
                screenshot_path=str(screenshot),
                login_elapsed_s=login_elapsed,
                render_elapsed_s=render_elapsed,
                main_text=main_text,
                log=lines,
            )
            return snapshot
        finally:
            await context.close()
            await browser.close()
