# Documentation and Figures

Use this directory for public documentation assets that are safe to redistribute.

## Figure policy

- Store diagrams, screenshots, and verified experiment figures under `docs/assets/`.
- Use descriptive, stable filenames such as `pipeline-overview.png` or `validation-roc-example.png`.
- Reference assets from Markdown with relative paths, for example `![Pipeline overview](assets/pipeline-overview.png)`.
- Add a caption and record the source or generation command near each figure.
- For thesis-derived summaries, identify the thesis table/figure and the
  evaluation level (image, exam or patient) in the surrounding Markdown.
- For generated results, record the configuration, checkpoint provenance
  and validation run that produced the asset.
- Label illustrative graphics clearly. Do not present placeholder data or unverified results as scientific findings.
- Do not add patient images, identifiable metadata, private dataset samples, credentials, or checkpoints.

The following assets are extracted from the associated thesis and are
reference-only summaries of private-data experiments:

- `thesis_supervised_roc.png` (thesis Figure 9);
- `thesis_convnext_confusion_matrix.png` (thesis Figure 10);
- `thesis_learning_strategies_roc.png` (thesis Figure 15).

They must remain captioned as thesis-derived and must not be presented as
results reproduced from the public repository. The Grad-CAM figure from
the thesis is intentionally not included because it contains clinical
mammography images.
