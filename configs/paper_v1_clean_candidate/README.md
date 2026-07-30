# TopoCAE-Recognizer paper_v1 clean candidate

This directory is the Phase 1 clean-candidate configuration surface. It does
not replace or modify `configs/paper_v1/`.

- `datasets.json` freezes the original GNN Synthetic split and the allowed
  Real-Dev / Rotation-Dev graph sources.
- `models.json` records the frozen GNN, the clean auxiliary artifacts, and the
  two comparison variants.
- `evaluation.json` freezes metrics, matching rules, bootstrap settings, and
  the Clean-Rib selection rule.

All paths containing an excluded S3 token are rejected before enumeration.
Real-Dev is report-only. Model fitting and threshold selection use only the
formal Synthetic-Train and Synthetic-Val partitions respectively.

