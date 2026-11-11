# Author figures

Eight author-supplied TIFF exports accompany the manuscript. Fig. 3 has one image, `fig3.tif`. The final 11 November update includes the corrected Fig. 3(d) and the R4/R3 labels in Fig. 6(a); these two files are byte-identical to the figures embedded in the current Word manuscript.

Fig. 3 raw data are in `../data/training_curves/fig3_data.csv`. Display seeds 1, 2 and 3 correspond to 20260720, 20260719 and 20260718. R history lengths in that order are 56, 56 and 30 epochs; Q lengths are 33, 56 and 56. Panel d corresponds to the 33-epoch Q/20260720 history. `export_training_data.py` exports all 287 observed rows without interpolation or added epochs.

The six raw JSON histories, CSV and seed/panel mapping are the numerical source for the figure. An older Origin project predating the corrected image is omitted from this snapshot. Further manuscript alignment notes are in `../docs/MANUSCRIPT_ALIGNMENT.md`.
