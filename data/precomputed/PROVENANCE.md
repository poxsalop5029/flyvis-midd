# Figure provenance metadata

Each main and supplementary figure has `metadata.json`. Shared fields identify the figure/artifact version, source repositories and commits, source paths and files, cohort/model or seed IDs, statistical unit, stimulus/rendering/timing contracts, metric/sign/estimands, reuse policy and limitations. Existing figure-specific fields remain available to notebook consumers. Fig.6 retains panel-specific cohorts and contracts; a shared schema does not imply a shared dataset.

`source_files` lists bundled filenames and their source repository-relative paths. A per-file `source_commit_sha` is asserted only for an already documented commit or when the original bytes match that path in the named Git commit. An observed checkout HEAD is recorded separately when it cannot verify the source artifact. A null source commit means unresolved provenance, not the latest upstream version. Locally generated products identify their committed `flyvis-midd` artifact and retain null upstream provenance where no exact originating copy is identified. Top-level commit maps summarize the known per-file commits; null per-file entries remain authoritative.

`source_sha256` identifies the original source bytes. `copied_sha256` identifies the bundled public bytes. These can differ when local path references in copied JSON/Markdown are sanitized. Numerical tables and binary arrays/figures remain unchanged. Notebook integrity checks use `copied_sha256`; original checksums remain available for source verification. SHA256 is an integrity identifier, not evidence of scientific validity.

Public text references use `fvsp/<repository-relative path>`, `fvpg/<repository-relative path>` or `flyvis-midd/<repository-relative path>`. Markdown source links use pinned GitHub references. `historical-local/Experiment_MIDD/<path>` denotes an unpublished historical source root whose repository/commit has not been established; it is an archival identifier, not an executable path or a claim that the source is publicly hosted. Such sources have null repository and commit fields. Other `historical-local` references retain incomplete archival locations without exposing machine-specific directories. Bundled products and their checksums provide the available reproducibility record.

Known limits remain: some main Fig.3 and Fig.6 historical source commits are unresolved; a matching source working-tree file may be untracked at the recorded checkout commit; upstream checkpoints and official FlyVis data have separate distribution policies. This provenance update performs no model inference or scientific reanalysis.

Lightweight verification from the repository root:

```bash
python -m pip install -e ".[analysis]"
python -m unittest discover -s tests -v
```

These checks require NumPy/PyTorch but no FlyVis installation, pretrained-model download, or figure regeneration.

## Canonical scientific inputs

All final figure inputs are bundled under `data/precomputed/fig01/` through `fig06/`, `data/precomputed/supplementary/`, `data/stimuli/` or the explicitly documented raster snapshots in `figures/`. Shared tables are stored once. `src/flyvis_midd/figure_inputs.json` is the authoritative input list; `canonical_sources.json` records scientific-table/config origins and checksum continuity at relocation.

The supported route is README → canonical notebooks → precomputed data → figures → manuscript. No acquisition notebook is required for processed-input replay. Shared calculations live in `figure_analysis.py`, `figure_rendering.py` and `round2_analysis.py`. Statistical definitions, numerical values, cohorts and signs are unchanged by relocation. Existing generated figures are unchanged; integration manifests record current path bindings and distinguish relocation from actual execution.

Official50 fixed diagnostic Sintel evaluation uses15samples and40resampled frames and is not held out. Model identities, fixed configs, stimulus hashes and source replay checks are bundled with the relevant tables. Four-point learning controls use56checkpoint members across14seeds and one fixed shuffle. Individual physical directions for crossed network/decoder pairs retain their original source Q checks. Native CW-positive and centered-eight-field CCW-positive signs remain separate. Fig1B is frozen-raster replay because complete representative vectors were not saved.

Tables S1–S4 are embedded in the manuscript PDF, with exact CSV companions in `figures/supplementary/tables/`. A1, B1 and D1 saved results are integrated as Supplementary Tables S5A–S5B from `data/precomputed/supplementary/biological_constraints/`; detailed data, acquisition manifests, model/phase records, and hashes are preserved there. A2/B/D0 inventories and pilot/raw supporting records are retained alongside them. `canonical_sources.json` records original/copy hashes and source commits for relocated data files, while `biological_constraints/metadata.json` records source identities and checksums. The public replay notebook reads saved tables only. Upstream FlyVis data/models and raw acquisition dependencies remain separate from repository-specific products; executable acquisition routes are listed in `docs/reproduction_map.md`.


## D1 target-node pilot acquisition

`data/precomputed/supplementary/biological_constraints/D1/pilot/pilot_target_node_distribution.csv` is retained as a canonical, reproducible acquisition output (1,661,184 data rows; SHA-256 `fbb384f5e7cbfc11fdad7feae0df57c95e00f4d1dd8b9342a40946e9a5d970ca`). Its public generator is `scripts/generate_d1_source_currents.py`, with the top-to-bottom route documented in `notebooks/supplementary/biological_constraints_D1_acquisition.ipynb`. The exact FlyVis v1.2.0 / commit `92b3845cc426dd309a1a0e1b3890156c42e14021`, official best checkpoints `flow/0000/000`, `/005`, `/010`, 3MIDD C=48 Original/Mirror stimuli, 12 phase settings, 50 frames, `dt=0.02`, 1 s fade-in, frames 10–49, BoxEye 15/13, and source/target reducer are frozen in the generator.

FlyVis and its pretrained model data are external dependencies; use the pinned upstream installation and `flyvis download-pretrained` as described in the notebook. The default generator action verifies the retained file without inference. `--regenerate-to-temp` writes a separate temporary CSV and compares it with the retained canonical content; it never replaces the canonical output. `--generate-canonical-if-missing` refuses an existing file and creates a missing canonical file only after the output matches the recorded SHA-256. Do not hand-edit the CSV. The earlier recorded pilot took about 91 seconds for 72 integrations on its CPU environment; this is an approximate historical runtime, not a guarantee.

The three-model pilot node table is not the source of official-50 D1 statistics in Table S5B. The committed S5B table is read by `notebooks/supplementary/biological_constraints_replay.ipynb`; this acquisition route does not recompute its statistics.
