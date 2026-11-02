"""
Generates the standalone GPU Google Colaboratory notebook for:
  colab/drosophila_whole_cns_specificity_colab.ipynb

Covers only the four experimental parts of the manuscript:
  1. Three-circuit Maslov-Sneppen wiring specificity hierarchy (MaleCNS v1.0)
  2. Closed-loop 3D NeuroMechFly v2 null-model ladder (MaleCNS v1.0 & Female BANC v888)
  3. Parallel K = 486 Common-Random-Numbers (CRN) virtual Split-GAL4 descending-neuron screen
  4. Cross-modality EM threshold recall audit & within-connectome fru/dsx double dissociation
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLAB_OUT = PROJECT_ROOT / "colab" / "drosophila_whole_cns_specificity_colab.ipynb"
COLAB_OUT.parent.mkdir(parents=True, exist_ok=True)


def md_cell(text: str) -> dict:
    lines = text.strip().splitlines()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + ("\n" if idx < len(lines) - 1 else "") for idx, ln in enumerate(lines)],
    }


def code_cell(code: str, form_view: bool = False) -> dict:
    lines = code.strip().splitlines()
    meta = {"cellView": "form"} if form_view else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": meta,
        "outputs": [],
        "source": [ln + ("\n" if idx < len(lines) - 1 else "") for idx, ln in enumerate(lines)],
    }


cells = [
    md_cell(
        r"""# Hierarchy of wiring specificity, parallel descending-neuron screens and dimorphic circuit organization in whole-CNS *Drosophila* connectome simulations
**Author:** Amandeep · Independent Researcher, Vadodara, Gujarat, India · [ORCID: 0009-0009-0689-2920](https://orcid.org/0009-0009-0689-2920)  
**Repository:** <https://github.com/indian-aman/drosophila-whole-cns-specificity>

This standalone Google Colab notebook reproduces the four experimental parts of the manuscript on an NVIDIA **`G4`** (RTX PRO 6000 Blackwell, 95 GB VRAM), **`H100`** or **`A100`** GPU:
1. **Three-circuit Maslov–Sneppen wiring specificity hierarchy** (`MaleCNS v1.0`: `167,565` neurons, `25.6M` directed connections) across mushroom-body olfaction, descending leg-motor steering and the central-complex compass ring.
2. **Closed-loop 3D `NeuroMechFly v2` (`MuJoCo 3.16`) null-model ladder** across `MaleCNS v1.0` and `Female BANC v888` (`157,789` neurons, `3.0M` connections at `|w| ≥ 3`).
3. **Single-pass `K = 486` Common-Random-Numbers (`CRN`) virtual Split-GAL4 silencing screen** across all `485` named descending-neuron cell types (`0.000000 Hz` baseline spread across unperturbed GPU slots).
4. **Cross-modality EM threshold recall audit (`|w| ≥ 1..20`) and symmetric within-connectome `fru`/`dsx` double dissociation** across both sexes."""
    ),
    code_cell(
        """#@title 1. Verify GPU Accelerator, Host RAM & CUDA Triton { display-mode: "form" }
import os, sys, shutil, subprocess
from pathlib import Path
import psutil
import torch

print("=" * 80)
print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"System RAM   : {psutil.virtual_memory().total / (1024**3):.1f} GB ({psutil.cpu_count()} threads)")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU          : {torch.cuda.get_device_name(0)} ({p.total_memory / (1024**3):.1f} GB VRAM, SM {p.major}.{p.minor})")
    try:
        import triton
        print(f"CUDA Triton  : {triton.__version__}")
    except Exception as exc:
        print(f"CUDA Triton  : fallback ({exc})")
print("=" * 80)""",
        form_view=True,
    ),
    code_cell(
        """#@title 2. Clone Clean Repository & Mount Compiled Connectomes (`MaleCNS v1.0` & `Female BANC v888`) { display-mode: "form" }
DRIVE_DATA_FOLDER = "Drosophila_Whole_CNS_Specificity" #@param {type:"string"}
INSTALL_FLYGMY_3D = False #@param {type:"boolean"}

REPO_DIR = Path("/content/drosophila-whole-cns-specificity")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "https://github.com/indian-aman/drosophila-whole-cns-specificity.git", str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

if INSTALL_FLYGMY_3D:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flygym==2.1.1", "mujoco==3.1.6"], check=True)

DRIVE_MOUNT = Path("/content/drive")
if not (DRIVE_MOUNT / "MyDrive").exists():
    try:
        from google.colab import drive
        drive.mount(str(DRIVE_MOUNT))
    except Exception as exc:
        print("Drive mount skipped:", exc)

CACHE_ROOT = Path("/content/drosophila_cns_data")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
for sub in ("compiled", "female_compiled"):
    cand = DRIVE_MOUNT / "MyDrive" / DRIVE_DATA_FOLDER / sub
    dst = CACHE_ROOT / sub
    if cand.exists() and not (dst / "connectome_compiled.npz").exists():
        dst.mkdir(parents=True, exist_ok=True)
        for fn in ("connectome_compiled.npz", "neurons_meta.parquet"):
            if (cand / fn).exists():
                shutil.copy2(cand / fn, dst / fn)
        print(f"Cached {sub} -> {dst}")

os.environ["MALECNS_COMPILED_DIR"] = str(CACHE_ROOT / "compiled")
os.environ["FEMALE_COMPILED_DIR"] = str(CACHE_ROOT / "female_compiled")""",
        form_view=True,
    ),
    code_cell(
        """#@title 3. Verify Shared-CSR Connectome Counts & Common-Random-Numbers (`0.000000 Hz` Spread) { display-mode: "form" }
import numpy as np
from src.connectome_engine import ConnectomeDataset, WholeCNSEngine, resolve_female_connectome_dir

ds_m = ConnectomeDataset()
ds_f = ConnectomeDataset(str(resolve_female_connectome_dir()))
print(f"MaleCNS v1.0    : N = {ds_m.N:,} neurons | {ds_m.nnz:,} directed edges | {ds_m.proofread_synapses:,} synapses")
print(f"Female BANC v888: N = {ds_f.N:,} neurons | {ds_f.nnz:,} directed edges | {ds_f.proofread_synapses:,} synapses")

crn_eng = WholeCNSEngine(ds_m, num_brains=16, enable_callosum=False, homogeneous_tau_ms=20.0, common_random_numbers=True, seed=700)
out = crn_eng.step_population(n_steps_ms=40, shared_stimuli=[(ds_m.idx_pfl3_turnR, 220.0)], bg_poisson_hz=20.0)
spread = float(np.ptp(out["turn_hz_per_brain"]))
print(f"CRN trajectory spread across K=16 unperturbed slots: {spread:.6f} Hz")
assert spread == 0.0""",
        form_view=True,
    ),
    code_cell(
        """#@title 4. Run Part 1, Part 3 & Part 4 GPU Experiments (`run_structural_specificity_and_screen.py`) { display-mode: "form" }
PART = "all" #@param ["all", "hierarchy", "dn_screen", "sex_switch"]
subprocess.run([sys.executable, "experiments/run_structural_specificity_and_screen.py", "--part", PART], check=True)""",
        form_view=True,
    ),
    code_cell(
        """#@title 5. Run Part 2 Closed-Loop 3D NeuroMechFly v2 Null Ladder (`run_3d_null_ladder.py`, optional) { display-mode: "form" }
CPU_WORKERS = 8 #@param {type:"integer"}
if INSTALL_FLYGMY_3D:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    subprocess.run([sys.executable, "experiments/run_3d_null_ladder.py", "--workers", str(CPU_WORKERS)], env=env, check=True)
else:
    print("Using precomputed results/rewiring_3d_results.json (enable INSTALL_FLYGMY_3D in Cell 2 to re-run MuJoCo episodes).")""",
        form_view=True,
    ),
    code_cell(
        """#@title 6. Rebuild Figures 1–4, Tables 1–5 & Run Invariant Test Suite { display-mode: "form" }
from IPython.display import Image, display

subprocess.run([sys.executable, "scripts/build_paper.py", "--no-pdf"], check=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)

for fn in sorted((REPO_DIR / "figures").glob("fig*.png")):
    print(fn.name)
    display(Image(filename=str(fn), width=900))""",
        form_view=True,
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "name": "drosophila_whole_cns_specificity_colab.ipynb",
            "provenance": [],
            "gpuType": "G4",
            "machine_shape": "hm",
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

COLAB_OUT.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {COLAB_OUT}")
