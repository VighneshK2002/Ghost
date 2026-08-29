# Result provenance

`results.csv` is transcribed from saved Marimo output in Git revision
`deccc33`. Rates were printed to three decimal places. `timeout_rate_inferred`
is `1 - success - wrong` using those rounded values and therefore is not an
exact count-derived metric.

The evaluator stops after crossing a requested episode minimum, and the saved
output does not retain its final exact denominator. `192` is consequently
recorded as `evaluation_episodes_minimum`, not an exact episode count.

The saved output explicitly says checkpoint saving was disabled. Blank
checkpoint cells are intentional. The CSV must not be represented as a newly
reproduced result or a learning curve. `plot_results.py` therefore draws one
final bar per seed and does not invent intermediate evaluations.
