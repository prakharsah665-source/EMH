import pytest
from playwright.async_api import async_playwright

from config.interview_session import (
    InterviewSessionError,
    require_fresh_tests_url,
)


@pytest.mark.asyncio
async def test_interview_launch():

    # Resolve the ONE primary session (EMH_INTERVIEW_URL override
    # or INTERVIEW_URL) - never read the raw .env value, so this
    # smoke test opens the same session as the full interview.
    try:
        interview_url, _claims = require_fresh_tests_url()
    except InterviewSessionError as error:
        pytest.fail(str(error))

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
            interview_url,
            wait_until="domcontentloaded",
        )

        assert response is not None

        await page.wait_for_timeout(5000)

        body = await page.locator("body").inner_text()

        assert body.strip(), (
            "Interview page is blank."
        )

        await browser.close()