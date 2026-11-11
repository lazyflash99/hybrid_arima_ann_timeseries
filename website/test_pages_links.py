
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

WEB = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(WEB), "_pages")
PORT = 8767
BASE = f"http://127.0.0.1:{PORT}/"
failures, errors = [], []


def expect_research(page, how):
    page.wait_for_url(f"{BASE}research_site/", timeout=10000)
    try:
        page.wait_for_function("() => window.__ready && window.__ready.size > 0", timeout=30000)
    except Exception:
        failures.append(f"{how}: research site did not load")
    print(f"  ok  {how} -> {page.url}")


def expect_playground(page, how):
    page.wait_for_url(f"{BASE}draw_your_own_series/", timeout=10000)
    try:
        page.wait_for_function("() => window.__playground && window.__playground.done", timeout=30000)
    except Exception:
        failures.append(f"{how}: playground did not load")
    print(f"  ok  {how} -> {page.url}")


def main():
    subprocess.run([sys.executable, os.path.join(WEB, "build_pages.py")], check=True)
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"], cwd=OUT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("console", lambda m: m.type == "error" and errors.append(m.text))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status} for {r.url}"))

            page.goto(BASE)
            page.click("#to-research")
            expect_research(page, "landing page -> research site")
            page.click("#playground-link")
            expect_playground(page, "research site menu -> playground")
            page.click("#research-link")
            expect_research(page, "playground menu -> research site")
            page.goto(f"{BASE}draw_your_own_series/")
            page.click("#research-footer-link")
            expect_research(page, "playground footer -> research site")
            page.goto(BASE)
            page.click("#to-playground")
            expect_playground(page, "landing page -> playground")
            pdf = page.request.get(f"{BASE}research_site/report.pdf")
            if pdf.status != 200:
                failures.append(f"report.pdf: HTTP {pdf.status}")
            browser.close()
    finally:
        server.terminate()
    failures.extend(f"console: {e}" for e in errors)
    if failures:
        print(f"{len(failures)} failures:\n" + "\n".join("  " + f for f in failures))
        sys.exit(1)
    print("all links work in both directions, no failed requests or console errors")


if __name__ == "__main__":
    main()
