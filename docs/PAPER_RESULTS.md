# Paper-to-artifact mapping

This mapping follows the paper's content rather than development goal numbers. `E3` means the frozen BiRel-FO system, `D0` the training/validation collection and `T10` the final 1,200-part test collection. Legacy identifiers inside immutable records preserve traceability.

| Paper item | Individual observations | Reproduction |
| --- | --- | --- |
| STEP-FO18 generation/split | `data/d0/split_and_labels.json`, training STEP archive; `data/t10/targets/`, `step_index.json`, test STEP archive | Verify counts, split membership and final STEP identities |
| Main component ablation | R, Q-final, rule score and E3 prediction channels in `data/t10/predictions/` | `python reproduce.py --only recognition` |
| AAGNet18-SI / MFTReNet-SI-D0 comparison | All three seeds, 1,200 predictions per seed/method, same target definitions | Same command; all four metrics and I/P/S scores are independently checked |
| Calibration limitation | Additional E1 learned-score predictions on the same test set | Kept as supplementary evidence: E1 macro F1 97.9880%, E3 92.7866%; the calibration is not claimed to uniformly improve the generated test distribution |
| Training/validation curves | Six hash-identified histories in `data/training_curves/` | `python plot_training.py` |
| 13 engineering examples | STEP, recognition, role/parameter records, edited STEP, solver decks, ODB and readback | `reproduce.py --only engineering`; `run_case.py`; `read_odb.py` |
| Historical Qwen-assisted 13/13 | Actual provider content, original requests, binding catalog and physical-equivalence records | `reproduce.py --only historical` |
| Unified Qwen comparison | 39 original requests; 42 actual API attempts, including three quota-error attempts with no usage | `reproduce.py --only unified` |
| Geometric perturbations | 3,600 individual records in `data/robustness/measurements.json` | Supplementary robustness recount |

## Recognition values

Values below are percentages, averaged over seeds 20260718, 20260719 and 20260720.

| Configuration | Whole-part accuracy | Face accuracy | Exact-instance macro F1 | Exact-instance micro F1 |
| --- | ---: | ---: | ---: | ---: |
| R only | 68.7500 | 99.6832 | 90.6778 | 91.7454 |
| Q only | 61.3611 | 99.6375 | 85.8923 | 84.7536 |
| Joint candidates, rule score | 35.1944 | 99.6832 | 74.2020 | 60.4707 |
| BiRel-FO (E3) | 77.0833 | 99.6832 | 92.7866 | 94.2530 |
| AAGNet18-SI | 8.7500 | 68.3924 | 27.7829 | 36.4236 |
| MFTReNet-SI-D0 | 14.7500 | 76.9442 | 34.4796 | 47.4136 |

The source histories contain R: 30/56/56 epochs and Q: 56/56/33 epochs in seed order. Curves must use those observed lengths. A blanket statement that all six archived training runs contain 56 epochs is inconsistent with these records.

## Language results and timing/cost definitions

The unified experiment fixes 13 tasks for each model condition. Qwen3.6-Plus uses temperature 0, thinking disabled and an 8,192-token output cap. Qwen3.8-Max-0902 uses `xhigh` reasoning and a 65,536-token cap. Requests to the two Plus conditions are checked to differ only by the semantic service. All output plans use the same direct face-handle contract in this experiment.

| Condition | Localization correct | Task success | Mean API time / task (s) | Median API time / task (s) | Total tokens | Estimated total CNY | Estimated mean CNY / task |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-Plus | 3/13 | 3/13 | 10.0638 | 10.0940 | 294,624 | 0.652168 | 0.050167 |
| Qwen3.6-Plus + BiRel-FO | 12/13 | 11/13 | 10.8545 | 10.4370 | 328,940 | 0.727010 | 0.055924 |
| Qwen3.8-Max-0902 | 10/13 | 10/13 | 206.4327 | 238.8900 | 398,602 | 7.289448 | 0.560727 |

For each task, API time is the sum of recorded attempt wall times; the mean and median are then computed over all 13 tasks. This excludes feature extraction, recognition, CAD editing and CAE solving. Token totals include all attempts with reported usage. Three quota-failure attempts have unknown usage and remain identified as such. API cost uses the archived 2026-10-23 list prices, with input, cached input and output charged separately. These are list-price estimates, not invoices or measured account charges. No exchange rate is invented for USD conversion.

The historical object-binding protocol instead obtains 13/13 correct localization and verified task evidence for the assisted Qwen condition (2/13 for the baseline). It uses a different output contract and a shared object-backed execution backend; its successful artifacts include equivalent-command reuse. These results are available separately for the 13-case demonstration.

## Author-draft checks before submission

The inspected author draft mixes the historical assisted 13/13 with the unified baseline and flagship numbers. A uniform comparison should use the three unified rows above, while the historical result belongs in the separate engineering demonstration. The displayed values 10.09/10.44/238.89 correspond to medians, not arithmetic means. Any cost column must specify total versus per-task mean and its currency conversion date. This artifact preserves both experiments and their original failures; it does not modify the author draft.
