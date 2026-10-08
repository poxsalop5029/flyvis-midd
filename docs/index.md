---
layout: default
title: MIDD × FlyVis
---

# MIDD × FlyVis

![Graphical Abstract: static MIDD stimuli and decoded motion in FlyVis networks](assets/graphical.png)

We study decoded motion responses to physically static motion illusion-like discrete disk (MIDD) patterns in FlyVis networks trained for optical-flow estimation. The decoded flow is a model output; it does not establish human or fly perception.

## Main findings

- Static MIDD patterns evoke decoded flow, with direction and magnitude varying across models.
- Responses depend on luminance, stimulus phase, and rendering geometry.
- Local three-level responses predict global four-level scalar responses under a defined rendering condition; this does not establish full vector-field additivity.
- MIDD responses emerge during ordinary optical-flow training, and interventions affect both static-pattern and physical-motion outputs, consistent with a role for general motion-processing and readout machinery.
- Matched shuffle controls do not establish stronger responses to canonical MIDD patterns, and the analyses do not establish an MIDD-specific mechanism.

## Manuscript

[Read the manuscript (PDF)](https://github.com/poxsalop5029/flyvis-midd/blob/main/manuscript/flyvis_midd_manuscript.pdf) · [Repository](https://github.com/poxsalop5029/flyvis-midd)

## Reproduce

The canonical notebooks reproduce figures from bundled processed inputs without running FlyVis inference. See the [reproduction guide](reproduction_map.md) for the complete input and output map.

| Figure | Topic | Notebook |
|---|---|---|
| Fig. 1 | Static MIDD and decoded flow | [Open](https://github.com/poxsalop5029/flyvis-midd/blob/main/notebooks/fig01/fig_1.ipynb) |
| Fig. 2 | Luminance dependence and controls | [Open](https://github.com/poxsalop5029/flyvis-midd/blob/main/notebooks/fig02/fig_2.ipynb) |
| Fig. 3 | Local-to-global scalar prediction | [Open](https://github.com/poxsalop5029/flyvis-midd/blob/main/notebooks/fig03/fig_3.ipynb) |
| Fig. 4 | Phase, geometry, and shared components | [Open](https://github.com/poxsalop5029/flyvis-midd/blob/main/notebooks/fig04/fig_4.ipynb) |
| Fig. 5 | Learning dynamics and task performance | [Open](https://github.com/poxsalop5029/flyvis-midd/blob/main/notebooks/fig05/fig_5.ipynb) |
| Fig. 6 | Population interventions and readout | [Open](https://github.com/poxsalop5029/flyvis-midd/blob/main/notebooks/fig06/fig_6.ipynb) |

## Code and data

The [GitHub repository](https://github.com/poxsalop5029/flyvis-midd) contains the analysis code, stimuli, processed results, and figure notebooks. For processed-data sources and limits, see [`data/precomputed/PROVENANCE.md`](https://github.com/poxsalop5029/flyvis-midd/blob/main/data/precomputed/PROVENANCE.md).
