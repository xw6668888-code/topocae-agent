# Engineering-task evaluation and API accounting

## Evaluation design

The language-model comparison evaluates three configurations on the same 13 engineering tasks. The recorded responses are assessed using one common engineering criterion applied to all 39 tasks. This assessment was applied retrospectively to the recorded experiments. The original face-set scores, solver states and all 42 API attempts are retained in the per-task records.

## CAD editing

A task succeeds when the operating faces identify all intended target objects and the resulting STEP satisfies the requested dimensions and geometric constraints. The edited solid is checked by geometric readback. Hole-radius changes can act on the cylindrical wall, and symmetric thickness changes can act on the two side faces. Complete feature membership is evaluated separately in the recognition experiment.

## CAE modeling

A task succeeds when the operating regions, materials, mesh settings, boundary conditions and loads match the requested physical model. Checks use the saved model and solver input. Solver completion is recorded as an additional outcome.

For the assisted Clevis case, the time-limited run and the completed reference analysis have identical 33 physical input sections, with different matrix-solver settings. Their complete input files are included for comparison:

```bash
python src/inp_equivalence.py data/engineering/clevis/solver_input.zip data/qwen_unified/physical/QP-S__clevis/solver_input.zip
```

`src/engineering_acceptance.py` implements the shared criteria. The paper table uses `operation_targets_correct` and `engineering_task_success`; the original `face_correct` and `task_success` fields are also provided. The engineering-task counts are 4/13 for Qwen3.6-Plus, 13/13 for Qwen3.6-Plus + BiRel-FO and 11/13 for Qwen3.8-Max-0902.

## Time, tokens and API cost

API time is measured from request dispatch to complete response. All attempts belonging to a task are summed, and the reported mean is the arithmetic mean over 13 tasks. Recognition, CAD execution and CAE solution time are measured outside this API-time metric.

Token consumption sums provider-reported input and output tokens. API cost is estimated from those counts and the archived 23 October 2026 service prices, including cached-input rates when available. The three group totals are CNY 0.652168, 0.727010 and 7.289448. The paper reports USD using the fixed conversion USD 0.15 per CNY, giving USD 0.0978252, 0.1090515 and 1.0934172. These values represent usage-based estimates at the listed prices. Three quota-failure attempts have no reported usage; their durations are included, and their usage and cost remain marked as unknown.

## Application and comparison settings

The object-based application examples and the three-condition comparison have separate task records. The U-Shaped Bracket case uses PlainCarbonSteel in the application example and Steel45 in all three comparison conditions. The associated material parameters and plans are retained with each experiment.
