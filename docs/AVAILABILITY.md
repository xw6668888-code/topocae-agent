# Data and code availability

The paper package contains STEP inputs and annotations, fixed splits, model checkpoints, prediction records, six training histories and their CSV export, author figures, archived language-model requests/responses and usage records, CAD outputs, CAE scripts/model evidence, solver outputs, and table-reconstruction code. Large files are resolved through `assets.json`, including explicit earlier-release references where the data are unchanged. Run `fetch_assets.py` followed by `verify.py`.

The manuscript-aligned snapshot is `paper-v1.3-20261111`. It is currently hosted in the author's private repository. Public visibility and an author-selected code/data license have not been enabled by this update. Third-party license terms remain documented in `THIRD_PARTY.md` and `licenses/`.

After the repository and referenced release assets become publicly accessible, the following statement can be used in the manuscript:

> The code, datasets, trained model checkpoints, and experimental records supporting this study are available at https://github.com/xw6668888-code/topocae-agent. The manuscript-specific package is archived as release paper-v1.3-20261111, which includes scripts for reproducing the reported tables and instructions for replaying the CAD/CAE examples.

Link the exact release in addition to the repository. An archived copy with a persistent DOI is recommended for long-term citation; it has not been created by this update. Abaqus installation and licensing are required to rerun the CAE analyses. The package supports frozen inference, replay and result reconstruction; it does not claim that all training and data generation have been certified from scratch.

The AEI Guide for Authors, checked on 11 November 2026, specifies research-data Option C: deposit research data in a suitable repository and provide the dataset link during submission, or explain why sharing is not possible. See https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors#writing-and-formatting-research-data . Public access and an open-source reuse license are separate matters; see https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository .

## Minimal release and reuse information

This is a paper reproduction package: it provides the frozen model, inputs, evaluation records and commands needed to check the reported results and replay the engineering cases. Publishing it does not require rerunning training, changing results, or adding exploratory experiments.

Public access determines who can obtain the files. A license determines how recipients may reuse or redistribute them. The author can state the original code license and the data/weight terms in one clearly scoped licensing document; separate license files for every material type are not inherently required. The existing third-party notices must be retained. A license for original materials has not yet been selected.

The downloader now supports anonymous public release assets and authenticated private assets. Use `python fetch_assets.py --anonymous` after public access is enabled. The same byte-size and SHA-256 checks apply in both modes. Anonymous download support does not change the repository's visibility.
