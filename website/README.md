# Website data

Content for a future project website. It is kept apart from the code: no script reads these files.

- `export_json.py` writes `data/` from the files in `results/`. It runs as the last step of `python run_all.py`,
  or on its own with `python website/export_json.py`.
- `data/<dataset>.json`: one file per dataset with
  - `series` (the full series), `actual_test` and `n_train`;
  - `arima_forecast` and `arima_error` on the test part;
  - for each protocol (`fixed`, `rolling`) and each model: `forecast` (mean over seeds), `forecast_seed_std`,
    `nn_correction` (forecast minus ARIMA forecast) and `final_error`.
- `data/<table>.json`: the result tables of `results/` as lists of records (see `results/README.md` for their columns).

Missing values are written as `null`.

## Websites

- `research_site/`: the results website (data built by `build_site_data.py`, tested by `test_research_site.py`).
- `draw_your_own_series/`: a self-contained browser demo of the hybrid idea on a series the visitor draws. Tested by
  `test_draw_your_own_series.py`; its model code is unit-tested with
  `node --test website/draw_your_own_series/tests/draw_models.test.mjs`.
- `build_pages.py` puts both sites and a landing page in `_pages/`; `test_pages_links.py` checks the links between
  them. Hosting steps are in `DEPLOY.md`.
