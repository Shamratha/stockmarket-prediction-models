"""Capture dashboard screenshots for the README (headless Chrome via Playwright).

Requires the dashboard running locally:  python -m uvicorn app:app --port 8600
Usage: python scripts/capture_screenshots.py
"""

import os

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, 'docs', 'screenshots')
URL = 'http://localhost:8600'

os.makedirs(DEST, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    page = browser.new_page(viewport={'width': 1440, 'height': 1000},
                            device_scale_factor=1.5)
    page.goto(URL, wait_until='networkidle')
    page.wait_for_selector('#f-table tr')

    page.screenshot(path=os.path.join(DEST, 'dashboard-forecasting.png'))
    print('forecasting ok')

    page.click('nav button[data-tab="agents"]')
    page.wait_for_selector('#a-table tr')
    page.wait_for_timeout(800)  # let the agent chart image load
    page.screenshot(path=os.path.join(DEST, 'dashboard-agents.png'))
    print('agents ok')

    page.click('nav button[data-tab="live"]')
    page.click('#l-go')
    page.wait_for_selector('#fc-chart path', timeout=60_000)
    page.wait_for_timeout(300)
    page.screenshot(path=os.path.join(DEST, 'dashboard-live-forecast.png'))
    print('live ok')

    browser.close()

print(f'saved to {DEST}')
