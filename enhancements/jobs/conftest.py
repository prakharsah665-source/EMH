import pytest
from playwright.sync_api import Page


@pytest.fixture
def logged_in_page(page: Page):

    page.goto("https://hiring.easemyhiring.ai/")

    page.get_by_role("textbox", name="Email").fill(
        "shivansh@volumetree.com"
    )

    page.locator("input[name='password']").fill(
        "Volumetree@123"
    )

    page.get_by_role("button", name="Login").click()

    return page