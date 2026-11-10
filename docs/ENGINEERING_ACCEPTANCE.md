# Engineering-task acceptance and API accounting

The three Qwen groups are evaluated on their original 13 task slots each. No response, parameter or face selection is edited during evaluation. All 42 attempts, including three quota failures, remain in the evidence. The current functional assessment was performed after the original experiment; it is a retrospective, common acceptance rule, not a new blind experiment.

For **CAD**, the selected operating faces must belong to every intended target object and to no other object. The archived edited STEP must match its recorded hash, pass solid/readback checks, achieve the requested dimensions and satisfy geometric constraints. Selecting the two side faces is sufficient for a verified symmetric thickness edit; selecting the cylinder wall is sufficient for a verified hole-radius edit. Full feature membership remains the recognition metric, rather than an additional CAD-output requirement.

For **CAE**, face regions and physical action parameters must match the requested model. Actual saved-model/INP readbacks verify material, mesh, supports and loads. Solver completion is a separate operational outcome. The assisted Clevis run reached its time limit with a different solver configuration; its actual INP has the same 33 physical sections as the successfully solved reference model, with only the matrix-solver setting changed. Recheck the two complete INPs after downloading assets:

```bash
python src/inp_equivalence.py data/engineering/clevis/solver_input.zip data/qwen_unified/physical/QP-S__clevis/solver_input.zip
```

`src/engineering_acceptance.py` applies these criteria to every group. The original scores remain in `qwen_unified_per_task.csv` (`face_correct`, `task_success`); the manuscript-facing assessment uses `operation_targets_correct` and `engineering_task_success`. Differences affect Plus/Bearing Housing, assisted/Clevis, assisted/Support Frame, and Max/Support Frame. The resulting functional task counts are 4, 13 and 11. No model plan has been silently repaired.

## Timing, tokens and currency

Elapsed API time is the sum of all attempt times for a task, from request dispatch to complete response; the table reports the arithmetic mean across 13 tasks. This excludes recognition, CAD execution and CAE solution. Token totals sum provider-reported input and output tokens across the recorded attempts. Pricing comes from the archived 23 October 2026 configuration, with separate cached-input rates where reported.

The original total costs are CNY 0.652168, 0.727010 and 7.289448. The manuscript's USD values are reproduced with a **fixed reporting conversion of USD 0.15 per CNY**, not a verified historical exchange quote. This gives USD 0.0978252, 0.1090515 and 1.0934172 (rounded: 0.098, 0.109, 1.093). These are list-price estimates, not account invoices or proof of an actual payment. Three failed quota attempts have no reported usage; their time remains included and their unknown billing is explicitly recorded.

## Experimental settings

The three-condition requests all use Steel45 for U-Shaped Bracket, as confirmed by the author for that experiment. The separate reference demonstration in Appendix A uses PlainCarbonSteel. These two material settings and their artifacts are preserved separately. They should be stated explicitly rather than described as identical instructions across the two experiments.
