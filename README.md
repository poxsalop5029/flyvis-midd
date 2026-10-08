# MIDD × FlyVis

[Manuscript (PDF) ](manuscript/flyvis_midd_manuscript.pdf)

[Graphical Abstract ](manuscript/graphical.png)

This repository accompanies a study of decoded motion responses to physically static motion illusion-like discrete disk (MIDD) patterns in FlyVis networks trained for optical-flow estimation. It contains the manuscript, stimuli, analysis code, processed results, and figure notebooks.

本研究では、静止MIDD刺激に対するFlyVisネットワークのdecoded flowを調べます。日本語での概要は以下のとおりです。

本研究では、静止MIDD刺激から生じるdecoded motionを、Original/Mirror反対称成分 \(A\) と共有成分 \(S\) に分解し、実際の方向反転が \(|A|>|S|\) で決まることを示します。さらに、局所3MIDD応答から4MIDDの大域成分を予測できること、非線形motion-detector arrayが静止方向残差を生成しうること、そしてこの応答がMIDDを学習目標に含まないoptic-flow学習中に形成され、trained network–decoder pairingに依存することを示します。decoded flowはモデル出力であり、人間やハエの知覚を直接測定したものではありません。

### Research agents

- OpenAI GPT
- Google Gemini
- Anthropic Claude

## Main findings

- Static MIDD patterns evoke directional decoded flow in optic-flow-trained FlyVis networks (a model output, not perception).
- Under a centered eight-field rendering, the four-level antisymmetric scalar follows an affine prediction from three-level components: \(A_4 \approx \alpha\,[A_3(C_1)-A_3(C_2)]+\beta\) (scalar only, not vector-field additivity).
- Matched shuffles and other static controls give comparable or larger outputs, so the response is not MIDD-specific. A classical nonlinear Reichardt array partially reproduces its directional structure (a candidate account, not an identified mechanism).
- The antisymmetric component emerges during ordinary optic-flow learning, although MIDD is not a training target.
- Output varies across models, phase, and rendering, and depends on network–decoder pairing, consistent with distributed, task-general computation and co-adaptation rather than a dedicated MIDD circuit.

## Reproduce the figures

The canonical notebooks replay bundled processed inputs; they do not download pretrained networks or run FlyVis inference. Use Python 3.9 or newer:

```bash
git clone --branch main https://github.com/poxsalop5029/flyvis-midd.git
cd flyvis-midd
python -m pip install -e ".[all]" scipy pillow jupyterlab
python -m jupyter lab
```

Run the desired notebook from the repository root. Supplementary notebooks for S1–S10 are in [`notebooks/supplementary/`](notebooks/supplementary/). See the [reproduction guide](docs/reproduction_map.md) for inputs, outputs, and verification details.

## Main figures

| Figure | Topic | Notebook |
|---|---|---|
| Fig. 1 | Static MIDD and decoded flow | [Open](notebooks/fig01/fig_1.ipynb) |
| Fig. 2 | Luminance dependence and controls | [Open](notebooks/fig02/fig_2.ipynb) |
| Fig. 3 | Local-to-global scalar prediction | [Open](notebooks/fig03/fig_3.ipynb) |
| Fig. 4 | Phase, geometry, and shared components | [Open](notebooks/fig04/fig_4.ipynb) |
| Fig. 5 | Learning dynamics and task performance | [Open](notebooks/fig05/fig_5.ipynb) |
| Fig. 6 | Population interventions and readout | [Open](notebooks/fig06/fig_6.ipynb) |

## Repository structure

- `data/` — stimuli and processed analysis inputs
- `notebooks/` — main and supplementary figure notebooks
- `src/flyvis_midd/` — shared analysis and plotting code
- `figures/` — generated figures and numerical summaries
- `manuscript/` — manuscript, references, and PDF

## Documentation

- [Reproduction map](docs/reproduction_map.md)
- [Processed-data provenance](data/precomputed/PROVENANCE.md)

## Repository maintainance

 `poxsalop5029`.

## License

Code is licensed under [MIT](LICENSE); original manuscript, figures, documentation, and research data are licensed under [CC BY 4.0](LICENSE-CONTENT.md). Third-party materials are excluded and retain their original licenses.
