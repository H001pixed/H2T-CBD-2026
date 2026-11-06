# Code and data statements

## Code availability

The implementation and English figure scripts for the current H2T-CBD manuscript are available in this repository. The code covers data preparation, full-text embedding extraction, model comparisons, anchoring ablations, direct ARS selection, independent validation, result aggregation, and illustrative case selection.

## Data availability

ASAP-1, ASAP-2.0, PERSUADE 2.0, and CELA are obtained from their original providers, listed in [data/README.md](data/README.md). Access and reuse follow each provider's terms. This release contains no datasets, student essays, per-essay predictions, experimental result tables, model checkpoints, pretrained weights, or generated figures. Plotting inputs must be supplied locally using the schemas in `figures/data/README.md`.

The PERSUADE 2.0 independent-validation subset excludes prompts shared with ASAP-2.0 and duplicate texts. Preparation scripts preserve the original holistic scores; CELA uses its original `Holistic` column.

## License

The project code is distributed under the [MIT License](LICENSE). Third-party dependencies, datasets, and pretrained models retain their original licenses and terms, as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Installed environments and dependency distributions are not bundled; `requirements.txt` lists the Python dependencies.

## Reproduction scope

The default experiment plan uses the four corpora in the current manuscript. ARS selection and feature standardization use fitting samples. The plotting scripts read locally supplied inputs. Coordinate contributions are signed terms in the model's internal score computation.
