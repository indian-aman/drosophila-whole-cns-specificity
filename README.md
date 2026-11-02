# Hierarchy of wiring specificity, parallel descending-neuron screens and dimorphic circuit organization in whole-CNS *Drosophila* connectome simulations

**Author:** Amandeep · Independent Researcher, Vadodara, Gujarat, India · [ORCID: 0009-0009-0689-2920](https://orcid.org/0009-0009-0689-2920)

[Manuscript (PDF)](paper/manuscript.pdf) · [Markdown source](paper/manuscript.md) · [LaTeX source](paper/manuscript.tex) · [Colab Notebook (`G4` GPU)](colab/drosophila_whole_cns_specificity_colab.ipynb) · [Results JSONs](results/)

This repository provides a GPU-accelerated leaky integrate-and-fire (LIF) simulator and structural counterfactual suite for the complete adult male *Drosophila* central nervous system (**MaleCNS v1.0**: 167,565 proofread neurons, 25,623,478 directed connections, 124.2M synapses) and the female brain-and-nerve-cord connectome (**BANC v888**: 157,789 neurons, 3,036,600 connections at `≥ 3` synapses, 23.6M synapses). Using nine nested Maslov–Sneppen null models that preserve every neuron's exact in-degree, out-degree, outgoing synapse weights and Dale's-law neurotransmitter sign, alongside closed-loop 3D **NeuroMechFly v2** (`MuJoCo 3.16`) physics and a single-pass `K = 486` Common-Random-Numbers (`CRN`) virtual Split-GAL4 screen, we dissect wiring specificity, causal descending bottlenecks, cross-modality EM recall, and sexually dimorphic routing.

---

## Summary of results

### 1. Three canonical circuits occupy three distinct rungs of wiring specificity (`Figure 1`, `Table 1`)
Probing mushroom-body olfactory expansion, descending motor steering and the central-complex head-direction ring under identical noisy LIF dynamics (`τ = 20 ms`, `τ_syn = 5 ms`, `20 Hz` background Poisson input, 5 rewire seeds × 5 noisy trials) separates them onto three distinct specificity levels within MaleCNS v1.0:
- **Coarsest rung — Mushroom-body olfaction (`8 ALPN glomeruli → 4,064 KCs → 37 MBON types`):** Complete degree-preserving randomization of all `22,586` `ALPN → KC` connections preserves sparse Kenyon-cell activation (`13.2%` vs `11.0%` active KCs) and noisy eight-odor decoding (`67.0%` vs `59.0%` in KCs; `36.1%` vs `35.1%` in MBONs), collapsing only when `region` (`98%`) or `full` (`99.4%`) shuffles scatter projection-neuron axons outside the calyx (`0.3%` active KCs).
- **Middle rung — Descending motor steering (`PFL3 → DN → 815 VNC leg MNs`):** Survives `instance` (`4.13 ± 0.19 Hz`) and `type + side` (`4.11 ± 0.21` vs `3.98 ± 0.27 Hz` native) shuffles, drops sixfold under fraction-matched `48%` global random rewiring (`0.64 ± 0.67 Hz`, Mann–Whitney $p = 0.0079$), and collapses under midline-mixing `type-only` shuffling (`0.02 ± 0.16 Hz`).
- **Finest rung — Central-complex compass (`8 EPG wedges → 44 Delta7/PEN glomerular instances`):** Breaks sharply between `instance` (`47.34%` rewired) and `type + side` (`48.01%` rewired — differing across only `0.67%` of CNS connections): `Delta7`/`PEN` firing drops `3.3×` (`1.06 → 0.33 Hz`, $p = 0.0079$), circular antipodal cosine modulation halves (`0.46 → 0.20`, $p = 0.0079$) and eight-wedge decoding drops from `88.1%` to `70.9%` (`12.5%` chance). Across `50 ms` post-stimulus windows, `EPG` firing decays to `2.5–2.7 Hz` (`3.8%` persistence), delineating where static ring topology ends and slow biophysical conductances begin.

![Three-circuit wiring specificity hierarchy](figures/fig1_wiring_specificity_hierarchy.png)

---

### 2. Hemisphere-specific cell-type connectivity supports closed-loop 3D walking and steering in both sexes (`Figure 2`, `Table 2`)
- **Closed-loop 3D NeuroMechFly v2 (`MuJoCo 3.16`, `0.48 s` goal-directed episodes):** `type + side` connectivity (`48%` MaleCNS, `37%` BANC rewired) preserves 3D walking speed (`17.46 ± 1.48` vs `16.77 ± 0.42 mm/s` male; `11.47 ± 0.97` vs `10.91 ± 1.13 mm/s` female), goal progress (`75.1%` vs `68.0%` male; `44.2%` vs `42.0%` female) and goal steering (`67.1 ± 2.9` vs `60.7 ± 6.4 Hz` male; `42.4 ± 1.5` vs `43.1 ± 1.2 Hz` female).
- **Midline cancellation vs locomotor gating:** `type-only` shuffling abolishes steering in both sexes (`4.4 ± 5.1` and `3.5 ± 7.2 Hz`), whereas `region` and `full` shuffles drop brain-wide firing below the `0.02 Hz` tripod-oscillator activity gate and halt locomotion (`0.15–0.37 mm/s` male; `0.01 mm/s` female).

![Closed-loop 3D walking and steering](figures/fig2_embodied_steering_3d.png)

---

### 3. Parallel `K = 486` Common-Random-Numbers virtual Split-GAL4 screen across all 485 descending-neuron cell types (`Figure 3`, `Table 3`)
Sharing one CSR matrix across `K = 486` independent state vectors under **Common Random Numbers (`CRN`)** (`0.000000 Hz` max spread across unperturbed GPU slots; `17.2 s` for 5 CRN seeds on a single NVIDIA `G4` GPU) acutely silences every named descending-neuron cell type (`slots 1..485` vs wild-type `slot 0`, modeling `Kir2.1 + TNT`) against strictly disjoint VNC motor pools:
- **Leg-motor steering (`815` VNC leg MNs, WT `3.06 Hz`):** Separates descending readout members (`DNb01` `18.8 Hz`, `DNb02` `17.9 Hz`, `DNa02` `8.2 Hz` on `turn_hz`) from downstream VNC leg-motor bottlenecks led by bilateral cholinergic **`DNa06` (`Δ = 2.77 ± 0.13 Hz`, `90%` loss)** and **`DNg75` (`Δ = 1.49 ± 0.37 Hz`)**.
- **Courtship wing drive (`495` VNC wing MNs, WT `3.21 Hz`):** Blindly recovers male-specific *fru*+ **`pIP10` #1 out of all 485 DN types (`Δ = 1.51 ± 0.11 Hz`, `47%` loss)**, followed by `DNp31` (`0.56 ± 0.10 Hz`), `DNp68*` (`0.52 ± 0.21 Hz`), `DNpe025*` (`0.46 ± 0.14 Hz`) and `DNpe002*` (`0.24 ± 0.07 Hz`).
- **Synapse-count independence:** Causal impact across all `485` DN types is uncorrelated with total VNC output synapse count (Pearson $r = 0.048$ leg, $r = 0.088$ wing; Spearman $\rho = 0.015, p = 0.74$ wing), survives `type + side` shuffling (`DNa06` `2.74 Hz`, `pIP10` `1.54 Hz`), and vanishes (`0.00 Hz`) under `full` shuffling.

![Virtual Split-GAL4 screen across 485 DN cell types](figures/fig3_virtual_split_gal4_dn_screen.png)

---

### 4. Threshold-matched EM recall audit and symmetric within-connectome *fru*/*dsx* double dissociation (`Figure 4`, `Tables 4–5`)
- **FIB-SEM vs GridTape TEM recall audit:** At a matched effective weight cutoff `|w| ≥ 3`, MaleCNS (8 nm isotropic FIB-SEM) retains **`3.57×` more directed connections** than BANC (serial-section GridTape TEM) (`10,326,400` vs `2,892,647`; `4.98×` at `≥ 10`, `5.81×` at `≥ 20`). Pruning MaleCNS to `|w| ≥ 3` leaves its brain-wide firing rate at `0.312 Hz` (`15.6×` above the `0.02 Hz` 3D gate; `64.1 Hz` steering), whereas native BANC fires at `0.019 Hz`.
- **Within-connectome *fru*/*dsx* double dissociation:**
  - **MaleCNS (`5,933` *fru*+/*dsx*+/dimorphic neurons = `3.54%`):** Rewiring internal *fru*/*dsx* edges (`1.9%`) breaks `mAL`-like unaroused sensory gating onto `pIP10` (`1.0 → 22.8 Hz`, $p = 0.011$); rewiring *fru*/*dsx*-touching edges (`10.7%`) collapses aroused `pIP10` (`176.2 → 23.2 Hz`) and wing MNs (`3.14 → 0.45 Hz`, `−86%`, $p = 0.0079$) while sparing `PFL3` navigation (`58.3 ± 2.8` vs `59.8 ± 2.2 Hz`); rewiring all purely isomorphic edges (`89%`) abolishes navigation (`0.0 Hz`) while sparing `pIP10` (`181.0 Hz`) and wing MNs (`4.46 Hz`).
  - **Female BANC (`2,718` *fru*+/*dsx*+/dimorphic neurons = `1.72%`):** Rewiring *fru*/*dsx* internal (`0.19%`) or touching (`2.5%`) edges abolishes `vpoDN` receptivity (`35.2 → 0.0` and `0.2 ± 0.6 Hz`, $p = 0.0097$ vs `36.5 Hz` matched isomorphic) while sparing navigation (`48.7` and `46.2` vs `49.2 Hz`); rewiring all purely isomorphic edges (`97%`) abolishes navigation (`0.0 Hz`) while sparing `vpoDN` (`36.5 Hz`).

![Threshold audit and within-connectome fru/dsx double dissociation](figures/fig4_fru_dsx_sex_switch_and_threshold_audit.png)

---

## Repository structure

```text
paper/        manuscript.md (source of truth), manuscript.tex, manuscript.html, manuscript.pdf, references.bib
figures/      fig1..fig4 publication figures (300 DPI PNG)
results/      rewiring_3d_results.json, structural_specificity_screen_results.json
src/          connectome_engine.py          shared-CSR LIF simulator, CUDA Triton kernel & CRN perturbation masks
              structural_specificity.py     Maslov-Sneppen CSR shuffles, cosine decoders, K=486 DN screen & fru/dsx probes
              vnc_neuromechfly_3d.py        closed-loop 3D NeuroMechFly v2 (MuJoCo 3.16) biomechanical coupling
              kc_mbon_matrix.py             15-compartment mushroom-body KC x MBON synaptic matrix
experiments/  run_structural_specificity_and_screen.py   3-circuit specificity ladder, K=486 DN screen & fru/dsx suite
              run_3d_null_ladder.py                      closed-loop 3D NeuroMechFly v2 null-model ladder
scripts/      build_paper.py, build_colab_notebook.py, compile_female_banc.py
colab/        drosophila_whole_cns_specificity_colab.ipynb   turnkey G4 / H100 GPU Google Colab notebook
tests/        test_whole_cns_specificity.py                  automated numerical & empirical invariant tests
```

---

## Reproducing

Requirements: Python 3.10+, `numpy`, `scipy`, `pandas`, `pyarrow`, `torch` (with `triton` on Linux/Colab CUDA), `matplotlib`, `flygym==2.1.1` and `mujoco==3.1.6` for closed-loop 3D physics, and `playwright` (`playwright install chromium`) to render `paper/manuscript.pdf`.

Point `MALECNS_COMPILED_DIR` at MaleCNS `compiled/` (`connectome_compiled.npz` and `neurons_meta.parquet`) and `FEMALE_COMPILED_DIR` at BANC `female_compiled/` (built via `scripts/compile_female_banc.py`):

```bash
# 1. Three-circuit specificity hierarchy, K=486 virtual Split-GAL4 DN screen & fru/dsx suite (~14 min on Colab G4 GPU)
python experiments/run_structural_specificity_and_screen.py --part all

# 2. Closed-loop 3D NeuroMechFly v2 null-model ladder across MaleCNS v1.0 and Female BANC v888
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python experiments/run_3d_null_ladder.py --workers 16

# 3. Compile Figures 1-4, Tables 1-5, LaTeX, HTML and PDF manuscript + run invariant test suite
python scripts/build_paper.py
python -m unittest discover -s tests -v
```

---

## Citation & license

Released under the [MIT License](LICENSE). Metadata for Zenodo and GitHub citation indexing are provided in [`.zenodo.json`](.zenodo.json) and [`CITATION.cff`](CITATION.cff):

```bibtex
@misc{amandeep2026hierarchy,
  title  = {Hierarchy of wiring specificity, parallel descending-neuron screens and dimorphic circuit organization in whole-CNS Drosophila connectome simulations},
  author = {Amandeep},
  year   = {2026},
  url    = {https://github.com/indian-aman/drosophila-whole-cns-specificity},
  note   = {Independent Researcher, Vadodara, Gujarat, India. ORCID: 0009-0009-0689-2920}
}
```

### Upstream connectome & physics sources
- **MaleCNS v1.0:** Berg et al., *Cell* 189, 5504–5526.e15 (2026), [doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015)
- **Female BANC:** Bates et al., *bioRxiv* (2025), [doi:10.1101/2025.07.31.667571](https://doi.org/10.1101/2025.07.31.667571)
- **LIF connectome parameters:** Shiu et al., *Nature* 634, 210–219 (2024), [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)
- **3D biomechanics:** NeuroMechFly v2, Wang-Chen et al., *Nature Methods* 21, 2353–2362 (2024), [doi:10.1038/s41592-024-02497-y](https://doi.org/10.1038/s41592-024-02497-y)
