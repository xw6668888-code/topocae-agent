# Data and code availability

The repository provides the BiRel-FO implementation, STEP-FO18 data and annotations, pretrained models, training histories, evaluation records, and CAD/CAE examples supporting the paper.

- Repository: https://github.com/xw6668888-code/topocae-agent
- Fixed paper package: https://github.com/xw6668888-code/topocae-agent/releases/tag/paper-v1.4-20261111
- Dataset and checkpoint layout: [DATA.md](DATA.md)
- Installation and reproduction commands: [REPRODUCTION.md](REPRODUCTION.md)

Source code, annotations, prediction records and paper tables are included in the repository. Large STEP archives, model checkpoints and Abaqus outputs are distributed as release assets. `assets.json` records each asset's exact release and checksum, and `fetch_assets.py` downloads it to the expected location.

```bash
python fetch_assets.py --anonymous
python verify.py
```

The paper's data-availability statement can cite this repository and its fixed release:

> The source code, datasets, pretrained models, and experimental records supporting this study are available at https://github.com/xw6668888-code/topocae-agent. The repository includes scripts for reproducing the reported results and running the CAD/CAE examples.

Third-party software and license notices are documented in [THIRD_PARTY.md](../THIRD_PARTY.md). Abaqus is installed separately for CAE solving and result-database access.
