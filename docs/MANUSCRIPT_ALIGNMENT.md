# Manuscript alignment: author version dated 10 November 2026

The repository retains the measured results and the author's supplied figures. The audit found the following corrections needed in the manuscript/figure source. Experimental records are not changed to force agreement.

## Results

- **Tables 1 and 2:** the reported means and sample standard deviations are reproduced from all retained per-part predictions.
- **Table 3:** 13/13 target-region selections, 6/6 CAD edits and 7/7 CAE analyses are supported by the historical object-binding demonstrations and their archived outputs. This is a separate demonstration from Table 4.
- **Table 4:** under common functional acceptance, Plus alone is **4/13 (30.77%)**, assisted Plus **13/13 (100%)**, and Max **11/13 (84.62%)**. The manuscript currently gives Plus 3/13 (23.08%). Plus also correctly edits the four Bearing Housing holes when it returns only their cylindrical walls. Accepting functional operating faces consistently across the three groups requires this correction. The assisted gain over Plus is therefore **69.23 percentage points**, replacing 76.92.
- Response times and token-derived CNY costs agree with the archived attempts. USD 0.098, 0.109 and 1.093 are obtained with a fixed 0.15 USD/CNY reporting conversion; add this convention to the table note. It is not a documented historical FX quotation.
- Appendix A's U-Shaped Bracket demonstration uses carbon steel, while the frozen Table 4 requests use Steel45. State this setting difference if Appendix A is also used to describe the comparison tasks.

## Fig. 3

Only the author's TIFF is retained. Display seeds 1/2/3 correspond to 20260720/20260719/20260718. The lower-left Q panel should use the Q/20260720 history, which has **33 epochs** and starts with training/validation loss **2.3691177932/0.6866311047**. The supplied image instead shows a 56-epoch curve matching R/20260720. Its correct data are in `data/training_curves/fig3_data.csv`; the Origin project contains the original Q values. Change the plotted dataset in Origin and relabel the lower row **d/e/f**, matching the caption. The author's image has not been silently redrawn.

## CAD figure annotations

- In Figs. 6a and 7a, the Bearing Housing dimensions are **R4 → R3**, or equivalently **Ø8 → Ø6**. They should not be labeled Ø4 → Ø3.
- In Fig. 7b, Flanged Sleeve is **R20 → R15**, or **Ø40 → Ø30**. Both sides of the supplied image currently read Ø20.
- Fig. 7c shows rib **thickness 2 → 4 mm**, without a diameter symbol.
- Fig. 7e shows a through-hole **radius 22.5 → 24 mm**; label R22.5/R24, or use diameters 45/48 mm consistently.
- Fig. 7f shows web **thickness 4.6985 → 3 mm**. Rounding the initial displayed value to 4.7 is reasonable, but it is not a diameter.

The new Fig. 7 panel order is (a) Bearing Housing, (b) Flanged Sleeve, (c) Support Bracket, (d) Guide Rail, (e) Mounting Plate, (f) Support Frame. Update the Appendix A cross-references: Guide Rail→7d, Flanged Sleeve→7b, Mounting Plate→7e, Support Bracket→7c.

In Fig. 8, (f) is **Clevis** and (g) is **U-Shaped Bracket**. Their current Appendix A references are reversed.

Fig. 1 contains five example base-body drawings, while the text describes six generator families. A composite drawing may illustrate multiple families; clarify that it is an example grouping if that is the intended presentation.

## Availability and reproduction scope

The current tree excludes off-paper exploratory prediction channels, geometric-perturbation data, old plotting variants, development queues and discarded run folders. Necessary imported source modules, all observations for the retained experiments, and failures within those experiments remain available. The previous local workspace and Git history are preserved.

The package verifies frozen inference, actual CAD replay, CAE compilation, archived solver results, and independent table reconstruction. A full portable from-scratch training/regeneration workflow is not certified by this release. The repository remains private and does not yet have an author-selected public code/data license.
