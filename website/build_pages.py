"""
Builds _pages/ at the repository root: a small landing page plus both websites, ready for static hosting.

  _pages/index.html               landing page linking to both sites
  _pages/research_site/           copy of website/research_site/ (without test screenshots)
  _pages/draw_your_own_series/    copy of website/draw_your_own_series/ (without tests and screenshots)

The research site reads the JSON already in website/research_site/data/ (written by website/build_site_data.py);
this script only copies files, so it needs no Python packages.   Usage:  python website/build_pages.py
"""
import os
import shutil
import sys

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
OUT = os.path.join(ROOT, "_pages")
SITES = {
    "research_site": ("screenshots",),
    "draw_your_own_series": ("screenshots", "tests", "package.json"),
}

LANDING = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1280">
<title>Hybrid ARIMA and Neural Network Forecasting</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='M1 12 L5 6 L9 9 L15 2' stroke='%230072B2' stroke-width='2' fill='none'/%3E%3C/svg%3E">
<style>
  :root { --bg: #fbfbfa; --surface: #ffffff; --ink: #17181a; --ink-2: #4a4c50; --line: #d9d8d3; --accent: #0b5cad; color-scheme: light; }
  :root[data-theme="dark"] { --bg: #121315; --surface: #1b1d20; --ink: #ececea; --ink-2: #b7b8b5; --line: #3a3d42; --accent: #79b5f2; color-scheme: dark; }
  body { margin: 0; background: var(--bg); color: var(--ink); min-width: 1280px;
         font: 17px/1.6 "Segoe UI", system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif; }
  main { max-width: 1000px; margin: 0 auto; padding: 80px 20px; }
  h1 { font-size: 36px; line-height: 1.25; margin: 0 0 12px; }
  p.lead { font-size: 19px; color: var(--ink-2); margin: 0 0 36px; }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  a.card { display: block; padding: 26px 28px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
           color: var(--ink); text-decoration: none; }
  a.card:hover { border-color: var(--accent); }
  a.card:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
  a.card h2 { font-size: 23px; margin: 0 0 8px; color: var(--accent); }
  a.card p { margin: 0; color: var(--ink-2); font-size: 16px; }
  footer { text-align: center; color: var(--ink-2); font-size: 15px; padding: 24px; }
</style>
<script>
  try {
    const t = localStorage.getItem("theme")
      || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = t;
  } catch (e) { /* storage blocked */ }
</script>
</head>
<body>
<main>
  <h1>Hybrid ARIMA and neural network forecasting</h1>
  <p class="lead">A reproduction of Zhang (2003) and our Extension, tested on 11 datasets. DA 3041 course project, IIT Guwahati.</p>
  <div class="cards">
    <a class="card" id="to-research" href="research_site/">
      <h2>Research site</h2>
      <p>The results of our experiments: forecasts, residual diagnostics, significance tests and the alternative hybrids,
        all loaded from the files our Python code wrote.</p>
    </a>
    <a class="card" id="to-playground" href="draw_your_own_series/">
      <h2>Draw your own series</h2>
      <p>Draw a time series and watch a simplified linear model plus neural network fit it, live in your browser.</p>
    </a>
  </div>
</main>
<footer>Both sites are static pages; nothing is sent to a server.</footer>
</body>
</html>
"""


def main():
    for name in SITES:
        src = os.path.join(WEB, name)
        if not os.path.isfile(os.path.join(src, "index.html")):
            sys.exit(f"{src} has no index.html")
    if not os.path.isfile(os.path.join(WEB, "research_site", "data", "datasets_index.json")):
        sys.exit("website/research_site/data/ is empty: run python website/build_site_data.py first")
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    for name, skip in SITES.items():
        shutil.copytree(os.path.join(WEB, name), os.path.join(OUT, name), ignore=shutil.ignore_patterns(*skip))
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(LANDING)
    open(os.path.join(OUT, ".nojekyll"), "w").close()  # serve files as they are on GitHub Pages
    n = sum(len(files) for _, _, files in os.walk(OUT))
    size = sum(os.path.getsize(os.path.join(d, x)) for d, _, files in os.walk(OUT) for x in files)
    print(f"built {OUT}: {n} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
