
import os
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

SITE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_site")
SHOTS = os.path.join(SITE, "screenshots")
PORT = 8765
URL = f"http://127.0.0.1:{PORT}/index.html?all=1"
SECTIONS = ["idea", "explorer", "residuals", "stationarity", "breakeven", "formlab", "rolling", "seedfan", "margin", "grid"]
failures, console_errors = [], []

CHECK_PAGE = """() => {
  const text = document.body.innerText;
  const bad = [];
  if (/\\bNaN\\b/.test(text)) bad.push('NaN on screen');
  if (/\\bundefined\\b/.test(text)) bad.push('"undefined" on screen');
  // smallest font among visible text (HTML and chart SVG text)
  let small = [];
  for (const node of document.querySelectorAll('body *')) {
    if (!node.childNodes.length) continue;
    const own = [...node.childNodes].some(c => c.nodeType === 3 && c.textContent.trim());
    if (!own) continue;
    const r = node.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (node.closest('.katex, .modebar, .sr-only')) continue;
    const size = parseFloat(getComputedStyle(node).fontSize);
    if (size < 13.95) small.push(`${node.tagName.toLowerCase()} ${size}px "${node.textContent.trim().slice(0, 30)}"`);
  }
  if (small.length) bad.push(`text under 14 px: ${small.slice(0, 5).join('; ')}`);
  const empty = [...document.querySelectorAll('.js-plotly-plot')].filter(p => !p.data || !p.data.length).map(p => p.id);
  if (empty.length) bad.push(`charts without data: ${empty.join(', ')}`);
  return bad;
}"""


def charts_ok(page, ids, where):
    res = page.evaluate("""(ids) => ids.map(id => { const p = document.getElementById(id);
        const ok = p && p.data && p.data.some(t => (t.y && t.y.length) || (t.z && t.z.length) || (t.type === 'histogram' && t.x && t.x.length));
        return ok ? null : id; }).filter(Boolean)""", ids)
    if res:
        failures.append(f"{where}: empty charts {res}")
    bad = page.evaluate(CHECK_PAGE)
    for b in bad:
        if "text under" not in b:
            failures.append(f"{where}: {b}")


def settle(page):
    page.wait_for_timeout(120)
    page.wait_for_function("() => !document.querySelector('.js-plotly-plot') || true")


def wait_ready(page, where):
    try:
        page.wait_for_function(f"() => window.__ready && window.__ready.size === {len(SECTIONS)} || (window.__errors && window.__errors.length)", timeout=60000)
    except Exception:
        failures.append(f"{where}: sections did not finish loading")
    errs = page.evaluate("() => window.__errors")
    ready = page.evaluate("() => [...window.__ready]")
    if errs:
        failures.append(f"{where}: section errors {errs}")
    missing = [s for s in SECTIONS if s not in ready]
    if missing:
        failures.append(f"{where}: sections not ready {missing}")


def select_all(page, sel_id, then):
    values = page.eval_on_selector(f"#{sel_id}", "s => [...s.options].map(o => o.value)")
    for v in values:
        page.select_option(f"#{sel_id}", v)
        settle(page)
        then(v)
    return values


def interact(page):
    # 1. the idea: every step on every dataset, and play
    def idea_steps(ds):
        page.click("#idea-dots button >> nth=0")
        for _ in range(4):
            page.click("#idea-next")
            page.wait_for_timeout(80)
        charts_ok(page, ["idea-chart"], f"idea {ds} last step")
    select_all(page, "idea-dataset", idea_steps)
    page.click("#idea-back"); page.click("#idea-play"); page.wait_for_timeout(300); page.click("#idea-play")

    # 2. explorer: every dataset x protocol, all models on, every focus model, layers off/on
    for proto in ("fixed", "rolling"):
        page.click(f"#ex-protocol button[data-value='{proto}']")
        for box in page.query_selector_all("#ex-models input:not([disabled])"):
            if not box.is_checked():
                box.check()
        def ex_dataset(ds, proto=proto):
            charts_ok(page, ["ex-chart"], f"explorer {proto} {ds}")
            for f in page.eval_on_selector("#ex-focus", "s => [...s.options].map(o => o.value)"):
                page.select_option("#ex-focus", f)
                page.wait_for_timeout(30)
                bad = page.evaluate(CHECK_PAGE)
                if any("NaN" in b or "undefined" in b for b in bad):
                    failures.append(f"explorer {proto} {ds} focus {f}: {bad}")
        select_all(page, "ex-dataset", ex_dataset)
    for layer in ("arima", "correction", "final"):
        page.uncheck(f"#ex-l-{layer}"); settle(page); page.check(f"#ex-l-{layer}"); settle(page)
    charts_ok(page, ["ex-chart"], "explorer layers")

    # 3. residual lab: every dataset x every model
    def res_dataset(ds):
        for m in page.eval_on_selector("#res-model", "s => [...s.options].map(o => o.value)"):
            page.select_option("#res-model", m)
            settle(page)
            charts_ok(page, ["res-time", "res-acf", "res-hist", "res-qq"], f"residuals {ds} {m}")
    select_all(page, "res-dataset", res_dataset)

    # 4. stationarity: every dataset
    select_all(page, "st-dataset", lambda ds: charts_ok(page, ["st-chart"], f"stationarity {ds}"))

    # 5. break-even (static) and 6. form lab: every form x strength
    charts_ok(page, ["be-chart"], "break-even")
    n = int(page.get_attribute("#fl-strength", "max"))
    for form in ("additive", "multiplicative"):
        page.click(f"#fl-form button[data-value='{form}']")
        for i in range(n + 1):
            page.fill("#fl-strength", str(i)); page.dispatch_event("#fl-strength", "input"); settle(page)
            charts_ok(page, ["fl-chart"], f"form lab {form} {i}")

    # 7. rolling: both datasets, every slider position, play/pause
    for ds in ("Lynx", "Nile"):
        page.click(f"#ro-dataset button[data-value='{ds}']")
        n = int(page.get_attribute("#ro-step", "max"))
        for i in range(n + 1):
            page.fill("#ro-step", str(i)); page.dispatch_event("#ro-step", "input"); page.wait_for_timeout(40)
        charts_ok(page, ["ro-chart"], f"rolling {ds}")
        page.click("#ro-play"); page.wait_for_timeout(1000); page.click("#ro-play")

    # 8. seed fan, 9. scale margin
    select_all(page, "sf-dataset", lambda ds: charts_ok(page, ["sf-chart"], f"seed fan {ds}"))
    n = int(page.get_attribute("#mg-margin", "max"))
    for i in range(n + 1):
        page.fill("#mg-margin", str(i)); page.dispatch_event("#mg-margin", "input"); settle(page)
        charts_ok(page, ["mg-chart"], f"scale margin {i}")

    # 10. grid: every dataset x combiner
    def gr_dataset(ds):
        for c in page.eval_on_selector("#gr-combiner", "s => [...s.options].map(o => o.value)"):
            page.select_option("#gr-combiner", c)
            settle(page)
            charts_ok(page, ["gr-chart"], f"grid {ds} {c}")
    select_all(page, "gr-dataset", gr_dataset)

    # theme toggle redraws every chart; links
    page.click("#theme-toggle"); page.wait_for_timeout(800)
    charts_ok(page, [f"{s}-chart" for s in ("idea", "ex", "st", "be", "fl", "ro", "sf", "mg", "gr")], "after theme toggle")
    page.click("#theme-toggle"); page.wait_for_timeout(300)
    href = page.get_attribute("#playground-link", "href")
    if href != "../draw_your_own_series/":
        failures.append(f"playground link is {href}")

    # keyboard: every control can receive focus with Tab
    count = page.evaluate("() => document.querySelectorAll('main select, main button, main input, nav a, #theme-toggle').length")
    seen = set()
    page.focus("body")
    for _ in range(count + 40):
        page.keyboard.press("Tab")
        seen.add(page.evaluate("() => document.activeElement && (document.activeElement.id || document.activeElement.textContent.trim().slice(0, 30))"))
    if len(seen) < count * 0.8:
        failures.append(f"keyboard: only {len(seen)} of about {count} controls reached with Tab")


def main():
    os.makedirs(SHOTS, exist_ok=True)
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"], cwd=SITE,
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
                    wait_ready(page, where)
                    page.wait_for_timeout(1500)
                    for b in page.evaluate(CHECK_PAGE):
                        failures.append(f"{where}: {b}")
                    page.screenshot(path=os.path.join(SHOTS, f"full_{width}_{theme}.png"), full_page=True)
                    if width == 1280:
                        page.add_style_tag(content=".topbar{position:static !important}")
                        for s in SECTIONS:
                            page.locator(f"#{s}").screenshot(path=os.path.join(SHOTS, f"{s}_{theme}.png"))
                        page.evaluate("() => document.querySelectorAll('style').forEach(e => e.textContent.includes('.topbar{position:static') && e.remove())")
                    if width == 1280 and theme == "light":
                        interact(page)
                    ctx.close()
            browser.close()
    finally:
        server.terminate()
    for e in console_errors:
        failures.append(f"console: {e}")
    print(f"screenshots in {SHOTS}")
    if failures:
        print(f"{len(failures)} failures:")
        print("\n".join("  " + f for f in failures[:60]))
        sys.exit(1)
    print("all checks passed: 4 configurations, every control exercised, no console errors")


if __name__ == "__main__":
    main()
