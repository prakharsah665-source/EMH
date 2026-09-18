import os
from playwright.sync_api import Page, expect


def test_login(page: Page):
    page.goto("https://hiring.easemyhiring.ai/")

    # Enter email
    page.get_by_role("textbox", name="Email").fill(
        os.environ["EMH_EMAIL"]
    )

    # Enter password
    page.locator("input[name='password']").fill(
        os.environ["EMH_PASSWORD"]
    )

    # Click Login
    page.get_by_role("button", name="Login").click()

    # Verify successful login
    expect(page.get_by_role("link", name="Jobs")).to_be_visible()