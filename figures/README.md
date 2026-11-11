# Author figures

Eight author-supplied TIFF exports accompany the current manuscript. Fig. 3 has one image, `fig3.tif`. The 11 November update replaces Figs. 3 and 7 and includes the updated author Origin project.

Fig. 3 raw data are in `../data/training_curves/fig3_data.csv`. Display seeds 1, 2 and 3 correspond to 20260720, 20260719 and 20260718. R history lengths in that order are 56, 56 and 30 epochs; Q lengths are 33, 56 and 56. `export_training_data.py` exports all 287 observed rows without interpolation or added epochs.

The new Fig. 3 labels its lower panels d/e/f correctly. Panel d still displays the 56-epoch R/20260720 curve rather than the 33-epoch Q/20260720 history. The supplied TIFF and Origin project are retained as received, with the six raw JSON histories and CSV as the numerical source. Remaining manuscript/figure corrections and the case-to-panel mapping are documented in `../docs/MANUSCRIPT_ALIGNMENT.md`.
