# Data and code availability

The paper package contains STEP inputs and annotations, fixed splits, model checkpoints, prediction records, six training histories and their CSV export, author figures, archived language-model requests/responses and usage records, CAD outputs, CAE scripts/model evidence, solver outputs, and table-reconstruction code. Large files are resolved through `assets.json`, including explicit earlier-release references where the data are unchanged. Run `fetch_assets.py` followed by `verify.py`.

The manuscript-aligned snapshot is `paper-v1.2-20261111`. It is currently hosted in the author's private repository. Public visibility and an author-selected code/data license have not been enabled by this update. Third-party license terms remain documented in `THIRD_PARTY.md` and `licenses/`.

After the repository and referenced release assets become publicly accessible, the following statement can be used in the manuscript:

> The code, datasets, trained model checkpoints, and experimental records supporting this study are available at https://github.com/xw6668888-code/topocae-agent. The manuscript-specific package is archived as release paper-v1.2-20261111, which includes scripts for reproducing the reported tables and instructions for replaying the CAD/CAE examples.

Link the exact release in addition to the repository. An archived copy with a persistent DOI is recommended for long-term citation; it has not been created by this update. Abaqus installation and licensing are required to rerun the CAE analyses. The package supports frozen inference, replay and result reconstruction; it does not claim that all training and data generation have been certified from scratch.

The AEI Guide for Authors, checked on 11 November 2026, specifies research-data Option C: deposit research data in a suitable repository and provide the dataset link during submission, or explain why sharing is not possible. See https://www.sciencedirect.com/journal/advanced-engineering-informatics/publish/guide-for-authors#writing-and-formatting-research-data . Public access and an open-source reuse license are separate matters; see https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository .
