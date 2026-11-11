// Entry point: theme toggle, menu, outbound links, and lazy start of each section when it comes near the viewport.
import { GITHUB_URL, PLAYGROUND_URL } from "./links.js";
import { redrawAll } from "./charts.js";

const SECTIONS = {
  idea: () => import("./sections/idea.js"),
  explorer: () => import("./sections/explorer.js"),
  residuals: () => import("./sections/residuals.js"),
  stationarity: () => import("./sections/stationarity.js"),
  breakeven: () => import("./sections/breakeven.js"),
  formlab: () => import("./sections/formlab.js"),
  rolling: () => import("./sections/rolling.js"),
  seedfan: () => import("./sections/seedfan.js"),
  margin: () => import("./sections/margin.js"),
  grid: () => import("./sections/grid.js"),
};

// sections that have finished their first render, for the automated tests
window.__ready = new Set();
window.__errors = [];

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("theme", theme); } catch (e) { /* storage blocked */ }
  document.getElementById("theme-toggle").textContent = theme === "dark" ? "Light mode" : "Dark mode";
}

async function start(id) {
  try {
    const mod = await SECTIONS[id]();
    await mod.init();
    window.__ready.add(id);
  } catch (err) {
    window.__errors.push(`${id}: ${err.message}`);
    const panel = document.getElementById(id);
    panel.insertAdjacentHTML("beforeend", `<p class="notice">This section could not load its data (${err.message}).</p>`);
    console.error(err);
  }
}

function main() {
  document.getElementById("playground-link").href = PLAYGROUND_URL;
  const repo = document.getElementById("repo-link");
  repo.href = GITHUB_URL;
  repo.textContent = `Source code on GitHub (${GITHUB_URL})`;

  const initial = document.documentElement.dataset.theme
    || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  setTheme(initial);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    redrawAll();
  });

  // start each section once it is within 600 px of the viewport; ?all=1 starts every section at once
  const startAll = new URLSearchParams(location.search).has("all");
  const started = new Set();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !started.has(e.target.id)) { started.add(e.target.id); start(e.target.id); }
    }
  }, { rootMargin: "600px 0px" });
  for (const id of Object.keys(SECTIONS)) {
    if (startAll) { started.add(id); start(id); } else io.observe(document.getElementById(id));
  }

  // highlight the menu entry of the section in view
  const links = [...document.querySelectorAll(".menu a[href^='#']")];
  const spy = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("section.panel").forEach((s) => spy.observe(s));
}

main();
