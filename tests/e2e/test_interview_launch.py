import pytest
from playwright.async_api import async_playwright

from config.settings import INTERVIEW_URL


@pytest.mark.asyncio
async def test_interview_launch():

    assert INTERVIEW_URL, (
        "INTERVIEW_URL is missing from .env"
    )

    async with async_playwright() as playwright:

        browser = await playwright.chromium.launch(
            headless=False
        )

        context = await browser.new_context(
            permissions=[
                "camera",
                "microphone",
            ]
        )

        page = await context.new_page()

        response = await page.goto(
            INTERVIEW_URL,
            wait_until="domcontentloaded",
        )

        assert response is not None

        await page.wait_for_timeout(5000)

        body = await page.locator("body").inner_text()

        assert body.strip(), (
            "Interview page is blank."
        )

        await browser.close()