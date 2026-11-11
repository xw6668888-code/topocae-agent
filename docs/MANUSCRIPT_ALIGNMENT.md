# Paper-to-package correspondence

The package accompanies *BiRel-FO: Engineering geometric feature object recognition for CAD editing and CAE modeling*. The manuscript identity is recorded in `MANUSCRIPT_SOURCE.json`.

## Recognition experiments

Tables 1 and 2 are reconstructed from 21,600 predictions covering six model configurations, three seeds and 1,200 test parts. Their 64 mean and standard-deviation entries match the manuscript tables. The overall BiRel-FO scores are 92.79% exact-instance macro F1 and 77.08% whole-part accuracy.

## Engineering experiments

The application records support 182 recognized instances, 13/13 target selections, six CAD edits and seven CAE analyses. The language-model comparison evaluates 13 tasks per configuration and reports engineering-task counts of 4/13, 13/13 and 11/13. The two experiments have separate task records and are reproduced through the commands in [PAPER_RESULTS.md](PAPER_RESULTS.md).

## Figures

`figures/` contains the eight paper figures. The six recorded training histories and their 287-row CSV provide the data for Fig. 3. Display seeds 1, 2 and 3 correspond to random seeds 20260720, 20260719 and 20260718. Fig. 3(d) shows the 33-epoch Q/20260720 history. Fig. 6(a) shows the Bearing Housing radius edit from R4 to R3.

Figure file identities are listed in `figures/INDEX.json`; dataset, checkpoint and result-file identities are recorded in the package manifests.
