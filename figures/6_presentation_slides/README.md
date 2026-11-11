# 6. Presentation slide figures

Larger fonts for the slides, drawn at their size on a 4:3 slide. Written by `make_slide_figures.py` and
`make_error_figure.py`. The slide deck in `presentation/` uses its own copies in `presentation/images/`.

These figures use the slide names: **"Extension 1" is the Extension** and **"Extension 1 Improved" is the
Extension + margin**.

| File | Content |
|---|---|
| `big_data_row.png` | The three original series: training part in black, test part in blue. |
| `big_hybrid_three.png` | Actual against Zhang's hybrid for the three datasets. |
| `big_all_three.png` | Actual against all three models for the three datasets; each model has its own line type. |
| `big_overlay_ext1.png`, `big_sunspot_overlay.png` | Sunspot test years with the forecasts overlaid, showing the record peak of 1957–59. |
| `big_mse_mad.png` | MSE and MAD bars, mean ± standard deviation over 10 seeds. |
| `big_error_accumulation.png` | Running total of squared errors over the test part. A steeper line means larger errors at that time. The end value divided by the number of test points is the MSE, labelled at the right end. |
