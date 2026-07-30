# TopoCAE-Recognizer paper_v1 freeze

This directory is the only Phase 0 paper-recognition configuration surface.

- `recognition.json` pins models and inference parameters.
- `label_schema.json` pins the exact checkpoint label spaces by source-file and canonical-content SHA256.
- `datasets.json` defines the paper data boundary and the mandatory excluded-artifact guard.
- `evaluation.json` freezes metric names and repeatability requirements; Phase 1 owns the full metric rerun.

Run model/hash verification:

```powershell
W:\python\envs\feagpt\python.exe run_paper_recognition_pipeline.py --verify-only
```

Run recognition on an inference-graph file or directory:

```powershell
W:\python\envs\feagpt\python.exe run_paper_recognition_pipeline.py `
  --input-path <graph-json-or-directory> `
  --output-dir <new-output-directory>
```

The entrypoint rejects the excluded s3 research branch, its derived output paths, and recursive ancestors that could include it.

