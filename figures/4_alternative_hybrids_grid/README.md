# 4. Alternative hybrids grid figures

**Files:** `<dataset>.png`, one heatmap per dataset. Written by `hybrid_framework.py`; redraw them from the saved
results with `python hybrid_framework.py --plots-only`.
**Numbers:** `results/4_alternative_hybrids_grid/`, whose README explains the 105 designs.

Each figure has four panels, one per combiner (additive, multiplicative, joint-input, validation-weighted).
- Rows are the linear models, labelled once on the left for all panels.
- Columns are the nonlinear models.
- Each cell is the design's mean test MSE divided by the test MSE of the best single model, which the title names.

**How to read it:**
- Values below 1, in blue, beat the best single model. Values above 1, in red, are worse.
- The colour scale is capped, so very poor designs share the darkest red.
- Blank cells are designs that are not defined (GRU with joint input).
- Compare along a row to see which nonlinear model suits a linear model, and across panels to see which combiner
  works.

**Why it is here:** it shows whether the hybrid idea works better with components other than ARIMA and the
logistic network, and which of the three parts matters most.
