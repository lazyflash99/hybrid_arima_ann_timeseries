
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

WEB = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(WEB, "draw_your_own_series", "screenshots")
PORT = 8766
URL = f"http://127.0.0.1:{PORT}/draw_your_own_series/index.html"
CHARTS = ["main-chart", "loss-chart", "acf-chart"]
VERDICTS = {"Network helped", "No clear gain", "Network overfit"}
failures, console_errors = [], []
QUICK = "--quick" in sys.argv  # a few values per slider instead of every value

CHECK_PAGE = """() => {
  const text = document.body.innerText;
  const bad = [];
  if (/\\bNaN\\b/.test(text)) bad.push('NaN on screen');
  if (/\\bundefined\\b/.test(text)) bad.push('"undefined" on screen');
  if (/Infinity/.test(text)) bad.push('Infinity on screen');
  const small = [];
  for (const node of document.querySelectorAll('body *')) {
    const own = [...node.childNodes].some(c => c.nodeType === 3 && c.textContent.trim());
    if (!own) continue;
    const r = node.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (node.closest('.modebar')) continue;
    const size = parseFloat(getComputedStyle(node).fontSize);
    if (size < 13.95) small.push(`${node.tagName.toLowerCase()} ${size}px "${node.textContent.trim().slice(0, 30)}"`);
  }
  if (small.length) bad.push(`text under 14 px: ${small.slice(0, 5).join('; ')}`);
  return bad;
}"""


def wait_done(page, where, timeout=30000):
    try:
        page.wait_for_function("() => window.__playground && window.__playground.done", timeout=timeout)
    except Exception:
        failures.append(f"{where}: training did not finish")
    err = page.evaluate("() => window.__playground.error")
    if err:
        failures.append(f"{where}: model error {err}")


def check(page, where, charts=True):
    for b in page.evaluate(CHECK_PAGE):
        failures.append(f"{where}: {b}")
    if charts:
        empty = page.evaluate("""(ids) => ids.filter(id => { const p = document.getElementById(id);
            return !(p && p.data && p.data.some(t => t.y && t.y.some(v => v !== null && Number.isFinite(v)))); })""", CHARTS)
        if empty:
            failures.append(f"{where}: charts without data {empty}")
        v = page.inner_text("#verdict").strip()
        if v not in VERDICTS:
            failures.append(f"{where}: verdict badge reads {v!r}")
        cards = page.eval_on_selector_all("#cards .card", "cs => cs.length")
        if cards != 6:
            failures.append(f"{where}: {cards} metric cards, expected 6")


def stroke(page, points):
    """Draws one stroke through points given as fractions (x, y) of the canvas."""
    box = page.locator("#canvas").bounding_box()
    xy = [(box["x"] + fx * box["width"], box["y"] + fy * box["height"]) for fx, fy in points]
    page.mouse.move(*xy[0])
    page.mouse.down()
    for p in xy[1:]:
        page.mouse.move(*p, steps=3)
    page.mouse.up()


def set_slider(page, sel, v):
    page.fill(sel, str(v))
    page.dispatch_event(sel, "input")
    page.wait_for_timeout(200)  # debounce


def interact(page):
    import math
    # a drawn stroke
    page.click("#clear")
    page.wait_for_timeout(100)
    if "Draw a series" not in page.inner_text("#status"):
        failures.append("clear: no prompt to draw")
    stroke(page, [(0.03 + 0.94 * i / 80, 0.5 - 0.3 * math.sin(i / 6) - 0.002 * i) for i in range(81)])
    wait_done(page, "drawn stroke")
    check(page, "drawn stroke")
    page.screenshot(path=os.path.join(SHOTS, "drawn_stroke.png"), full_page=True)
    if page.inner_text("#warnings").strip():
        failures.append(f"forward stroke gave a warning: {page.inner_text('#warnings')}")

    # a backwards stroke on top of it, then undo
    stroke(page, [(0.6 - 0.2 * i / 20, 0.3) for i in range(21)])
    wait_done(page, "backward stroke")
    if "backwards" not in page.inner_text("#warnings"):
        failures.append("backward stroke: no warning")
    check(page, "backward stroke")
    page.click("#undo")
    wait_done(page, "undo")
    if page.inner_text("#warnings").strip():
        failures.append("undo did not remove the backward stroke")

    # two strokes with a gap
    page.click("#clear")
    stroke(page, [(0.05 + 0.35 * i / 30, 0.5 + 0.1 * math.sin(i / 3)) for i in range(31)])
    stroke(page, [(0.6 + 0.35 * i / 30, 0.4 + 0.1 * math.sin(i / 3)) for i in range(31)])
    wait_done(page, "gap")
    if "gap" not in page.inner_text("#warnings"):
        failures.append("gap: no warning")
    check(page, "gap")

    # a stroke too short to use
    page.click("#clear")
    stroke(page, [(0.4 + 0.05 * i / 10, 0.5) for i in range(11)])
    page.wait_for_timeout(200)
    if "fifth" not in page.inner_text("#warnings"):
        failures.append("short stroke: no warning")
    check(page, "short stroke", charts=False)

    # every preset
    for name in ("Sine + trend", "Sawtooth", "Random walk", "Quiet then spike"):
        page.click(f".preset[data-preset='{name}']")
        wait_done(page, f"preset {name}")
        check(page, f"preset {name}")
        slug = name.lower().replace(" + ", "_").replace(" ", "_")
        page.locator("#results").screenshot(path=os.path.join(SHOTS, f"preset_{slug}.png"))
        print(f"  preset {name}: {page.inner_text('#verdict')} | held-out change "
              f"{page.inner_text('#cards .card:nth-child(6) .v')}")

    # every slider through every value (short training while the other sliders move)
    set_slider(page, "#epochs", 50)
    wait_done(page, "epochs 50")
    sliders = ["#n-points", "#noise", "#ar-p", "#hidden", "#lags", "#epochs"]
    for sel in sliders:
        lo, hi, st = (int(page.get_attribute(sel, a)) for a in ("min", "max", "step"))
        start = page.input_value(sel)
        values = list(range(lo, hi + 1, st))
        if QUICK:
            values = values[:: max(1, len(values) // 4)] + [hi]
        for v in values:
            set_slider(page, sel, v)
            wait_done(page, f"{sel}={v}")
            check(page, f"{sel}={v}")
        set_slider(page, sel, start if sel != "#epochs" else 50)
        wait_done(page, f"{sel} reset")

    # the hardest setting: fewest points, highest orders, differencing
    set_slider(page, "#n-points", 60); set_slider(page, "#ar-p", 6); set_slider(page, "#lags", 6)
    page.check("#diff"); page.wait_for_timeout(200)
    wait_done(page, "N=60 p=6 lags=6 differenced")
    check(page, "N=60 p=6 lags=6 differenced")
    page.uncheck("#diff"); set_slider(page, "#n-points", 100); set_slider(page, "#ar-p", 2); set_slider(page, "#lags", 2)
    wait_done(page, "back to defaults")

    # noise re-roll changes the series; buttons
    set_slider(page, "#noise", 20)
    wait_done(page, "noise 20")
    before = page.evaluate("() => document.getElementById('main-chart').data[0].y.slice(0, 5)")
    page.click("#reroll")
    wait_done(page, "re-roll")
    after = page.evaluate("() => document.getElementById('main-chart').data[0].y.slice(0, 5)")
    if before == after:
        failures.append("re-roll noise did not change the series")
    set_slider(page, "#epochs", 500)
    page.wait_for_timeout(300)
    page.click("#stop")
    page.wait_for_timeout(100)
    ep = page.evaluate("() => window.__playground.epoch")
    if not (0 < ep < 500) or "Stopped" not in page.inner_text("#status"):
        failures.append(f"stop: epoch {ep}, status {page.inner_text('#status')!r}")
    check(page, "stopped")
    page.click("#retrain"); wait_done(page, "retrain"); check(page, "retrain")
    seed = page.inner_text("#seed-text")
    page.click("#reinit"); wait_done(page, "re-initialise"); check(page, "re-initialise")
    if page.inner_text("#seed-text") == seed:
        failures.append("re-initialise did not change the seed")
    for t in ("series", "linear", "hybrid"):
        page.uncheck(f"#show-{t}"); page.wait_for_timeout(50); page.check(f"#show-{t}")
    check(page, "toggles")
    page.click("#theme-toggle"); page.wait_for_timeout(300); check(page, "theme toggled")
    page.click("#theme-toggle"); page.wait_for_timeout(300)

    # links and keyboard
    for sel in ("#research-link", "#research-footer-link"):
        if page.get_attribute(sel, "href") != "../research_site/":
            failures.append(f"{sel} points to {page.get_attribute(sel, 'href')}")
    if "not the full pipeline used in our experiments" not in page.inner_text("#demo-note"):
        failures.append("the demo note is missing")
    count = page.evaluate("() => document.querySelectorAll('main button, main input, nav a, #theme-toggle').length")
    seen = set()
    page.focus("body")
    for _ in range(count + 10):
        page.keyboard.press("Tab")
        seen.add(page.evaluate("() => document.activeElement && (document.activeElement.id || document.activeElement.textContent.trim().slice(0, 30))"))
    if len(seen) < count * 0.9:
        failures.append(f"keyboard: only {len(seen)} of {count} controls reached with Tab")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(SHOTS, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"], cwd=WEB,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            for width in (1280, 1920):
                for theme in ("light", "dark"):
                    where = f"{width}px {theme}"
                    ctx = browser.new_context(viewport={"width": width, "height": 1000}, color_scheme=theme)
                    page = ctx.new_page()
                    page.on("console", lambda m, w=where: m.type == "error" and console_errors.append(f"{w}: {m.text}"))
                    page.on("pageerror", lambda e, w=where: console_errors.append(f"{w}: {e}"))
                    page.on("response", lambda r, w=where: r.status >= 400 and console_errors.append(f"{w}: HTTP {r.status} for {r.url}"))
                    page.goto(URL)
                    wait_done(page, where)
                    page.wait_for_timeout(500)
                    check(page, where)
                    page.screenshot(path=os.path.join(SHOTS, f"full_{width}_{theme}.png"), full_page=True)
                    if width == 1280 and theme == "light":
                        interact(page)
                    ctx.close()
            browser.close()
    finally:
        server.terminate()
    failures.extend(f"console: {e}" for e in console_errors)
    print(f"screenshots in {SHOTS}")
    if failures:
        print(f"{len(failures)} failures:")
        print("\n".join("  " + f for f in failures[:60]))
        sys.exit(1)
    print("all checks passed: 4 configurations, strokes, presets, every slider value and button, no console errors")


if __name__ == "__main__":
    main()
