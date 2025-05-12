# screenshot_capture.py

import os
import asyncio
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
from config import WEBAPP_URL, SCREENSHOT_DIR


def is_internal_link(href: str, base_domain: str) -> bool:
    parsed = urlparse(href)
    return parsed.netloc == "" or base_domain in parsed.netloc


async def capture_screenshots_for_all_pages():
    if not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(WEBAPP_URL, wait_until="networkidle")

        base_domain = urlparse(WEBAPP_URL).netloc

        # Extract internal links
        links = await page.eval_on_selector_all(
            "a[href]",
            "elements => elements.map(e => e.href)"
        )
        internal_links = list({urljoin(WEBAPP_URL, href) for href in links if is_internal_link(href, base_domain)})

        visited = set()
        for idx, link in enumerate(internal_links):
            if link in visited:
                continue

            try:
                await page.goto(link, wait_until="networkidle")
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"page_{idx + 1}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                print(f"📸 Screenshot saved: {screenshot_path}")
                visited.add(link)
            except Exception as e:
                print(f"⚠️ Failed to capture {link}: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(capture_screenshots_for_all_pages())
