# Paper figures and source data

This directory contains the eight TIFF figures used in the manuscript. File sizes, dimensions, resolution and SHA-256 hashes are listed in `INDEX.json`.

Fig. 3 is provided as `fig3.tif`. Its numerical source is `../data/training_curves/fig3_data.csv`, with six JSON histories and the seed/panel mapping in the same data directory. Run `python export_training_data.py` from the repository root to export the 287 recorded epochs.

| Panels | Branch | Display seeds | Random seeds | Recorded epochs |
| --- | --- | --- | --- | --- |
| a, b, c | R | 1, 2, 3 | 20260720, 20260719, 20260718 | 56, 56, 30 |
| d, e, f | Q | 1, 2, 3 | 20260720, 20260719, 20260718 | 33, 56, 56 |

The remaining figures illustrate the feature taxonomy, recognition architecture, CAD/CAE workflow and engineering applications. The corresponding data and execution records are indexed in `../docs/PAPER_RESULTS.md`.
