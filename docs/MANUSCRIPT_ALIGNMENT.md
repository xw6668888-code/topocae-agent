# Manuscript alignment: 11 November 2026

The current authority is the author's updated `Manuscript.docx`; its hash is in `MANUSCRIPT_SOURCE.json`. Only publication-relevant data, code, results and author figures are distributed. The manuscript file is not included.

## Verified results and updates

- Tables 1 and 2: all 64 mean/standard-deviation entries agree with the independent reconstruction from 21,600 retained predictions.
- Table 3: the separate object-binding demonstrations support 13/13 target selections, 6/6 CAD edits and 7/7 CAE analyses.
- Table 4: the updated manuscript agrees with the common functional engineering assessment: Plus 4/13 (30.77%), Plus + BiRel-FO 13/13 (100%), Max 11/13 (84.62%). The assisted gain over Plus is correctly stated as 69.23 percentage points. API times and costs match the archived calls. The original strict scores and solver outcomes remain available separately.
- Fig. 7 now correctly labels radius changes R4 to R3, R20 to R15 and R22.5 to R24, and thickness changes 2 to 4 and 4.7 to 3 mm.
- Fig. 3(d) now uses the 33-epoch Q/20260720 curve. Fig. 6(a) uses R4/R3. Both corrected TIFFs match the current Word manuscript byte for byte. Only one author Fig. 3 image is included, with all six raw histories and the unchanged 287-row CSV.
- Appendix A, Table 5 now matches the Fig. 7 panel order: Bearing Housing 7a; Flanged Sleeve 7b; Support Bracket 7c; Guide Rail 7d; Mounting Plate 7e; Support Frame 7f.

## Remaining figure and text corrections

1. **Appendix A, Table 6:** Clevis corresponds to Fig. 8f and U-Shaped Bracket to Fig. 8g. These two references remain reversed in the manuscript.
2. **Paragraph after Table 4:** calculate relative increases from unrounded data. Assisted Plus increases mean API time by **7.86%** and API cost by **11.48%** over Plus. The manuscript uses 7.85% and 11.22%, obtained from rounded display values.
3. **Table 4 currency note:** USD values use a fixed reporting conversion of **USD 0.15 per CNY**. State this convention; it is not a verified historical exchange quotation.
4. **Task settings:** Appendix A's U-Shaped Bracket demonstration uses carbon steel; the separately frozen Table 4 comparison uses Steel45. Identify this setting difference where Appendix A is used to describe comparison tasks.
5. **Data availability:** the manuscript still says that the data are confidential. Replace that statement when the paper package becomes publicly accessible; the present repository remains private. See `AVAILABILITY.md`.

Fig. 1 presents five illustrated base-body groups while the dataset describes six generator families. If the composite example represents multiple families, clarify the grouping in the caption.

## Reproduction scope

This update changes author figure/source files and publication documentation. The inference code, weights, training/test inputs, original predictions, language responses and CAD/CAE outputs are unchanged. No API call, training run or solver execution was added. The reported tables retain the previously verified reconstruction; current source and asset hashes were checked again. Public anonymous and authenticated private downloads were both exercised successfully. Frozen inference and engineering execution retain the previously recorded validation scope. A fully portable from-scratch training/regeneration workflow is not certified by this package.
