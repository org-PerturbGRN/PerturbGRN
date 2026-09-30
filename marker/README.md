# Marker Gene Presets

`marker_genes.csv` records the marker genes used by
`scripts/prepare_pseudotime_cell_order.py`.

Available presets:

| Preset | Purpose |
| --- | --- |
| `T` | T-cell state scoring. |
| `ESC` | Pluripotency/ESC state scoring. |
| `fibroblast` | Fibroblast state scoring. |

These markers are used only to choose the pseudo-time root among control cells.
Custom marker genes can be supplied with `--marker-file`.
