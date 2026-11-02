"""
Build the standalone manuscript from paper/manuscript.md.

Every figure and table cell is computed directly from results/*.json, so no
number in them is copied by hand. Outputs:

  figures/fig{1..4}_*.png   figures (matplotlib)
  paper/manuscript.tex      arXiv / bioRxiv / Overleaf source (pdfLaTeX + BibTeX)
  paper/manuscript.html     print-ready HTML
  paper/manuscript.pdf      the HTML printed by headless Chromium (Playwright)

manuscript.md is plain Markdown with a few additions: `[@key; @key]` cites
references.bib, `@fig:x` / `@tbl:x` refer to figures and tables,
`![caption](path){#fig:x width=60%}` places a figure and `{{table:name}}`
a generated table.

    python scripts/build_paper.py            # everything
    python scripts/build_paper.py --no-pdf   # skip the browser step
"""
from __future__ import annotations

import argparse
import html
import json
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PAPER, FIGS, RESULTS = REPO / "paper", REPO / "figures", REPO / "results"

# chart ink and the validated categorical palette
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1b8a6b"

# each structured null sits next to its fraction-matched random control
LADDER_3D = [
    ("real", "Real connectome"),
    ("rewired_type_side", "Type + side shuffle"),
    ("partial_matched_type_side", "Random rewiring, matched"),
    ("rewired_type", "Type-only shuffle"),
    ("partial_matched_type", "Random rewiring, matched"),
    ("rewired_region", "Region shuffle"),
    ("rewired", "Full shuffle"),
    ("zero_weights", "No synapses"),
]
GROUP_STARTS = (1, 3, 5)  # a gap opens before each matched pair and before the floors
HIERARCHY_CONDS = [
    ("real", "Real connectome", "all"),
    ("rewired_pn_kc_random_claws", "PN → KC random claws", "all except ALPN → KC"),
    ("rewired_kc_mbon_compartments", "KC → MBON shuffle", "all except KC → MBON"),
    ("rewired_instance", "Instance shuffle", "(type, side, wedge)²"),
    ("rewired_type_side", "Type + side shuffle", "(type, side)²"),
    ("rewired_partial_48", "Random, matched", "degrees only"),
    ("rewired_type", "Type-only shuffle", "type × type"),
    ("rewired_region", "Region shuffle", "region × region"),
    ("rewired_full", "Full shuffle", "degrees only"),
]
SEX_SWITCH_CONDS = [
    ("real", "Real connectome"),
    ("rewired_fru_both", "fru/dsx internal"),
    ("rewired_iso_matched_both", "Iso. matched"),
    ("rewired_fru_touch", "fru/dsx touching"),
    ("rewired_iso_matched_touch", "Iso. matched"),
    ("rewired_all_isomorphic", "All isomorphic"),
    ("rewired_full", "Full shuffle"),
]
UNNUMBERED = {
    "Ethics statement",
    "Competing interests",
    "Data and code availability",
    "Acknowledgements",
}

Table = Tuple[str, List[str], str, List[Optional[List[str]]]]  # caption, header, align, rows (None = rule)


# ----------------------------------------------------------------------------- data
def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def fmt_pct_frac(f: float) -> str:
    pct = 100.0 * float(f)
    if pct == 0.0:
        return "0%"
    if pct < 1.0:
        return f"{pct:.2f}%"
    if 99.0 < pct < 100.0:
        return f"{pct:.1f}%"
    return f"{pct:.1f}%" if pct < 15.0 else f"{pct:.0f}%"


class Data:
    def __init__(self):
        self.r3d = load("rewiring_3d_results.json")
        self.spec = load("structural_specificity_screen_results.json")
        self.frac3d: Dict[Tuple[str, str], List[float]] = {}
        for v in self.r3d["ladder"]["validation"]:
            self.frac3d.setdefault((v["sex"], v["condition"]), []).append(v["fraction_edges_rewired"])

    def rows3d(self, sex: str, cond: str) -> List[dict]:
        return [r for r in self.r3d["ladder"]["rows"] if r["sex"] == sex and r["condition"] == cond]

    def frac3d_str(self, cond: str, sex: str) -> str:
        if cond == "real":
            return "0%"
        f = self.frac3d.get((sex, cond))
        if not f:
            return "—"
        pct = 100 * np.mean(f)
        return f"{pct:.1f}%" if 99 < pct < 100 else f"{pct:.0f}%"  # 99.8% must not round to 100%


def ladder3d_labels(d: Data, sex: str) -> List[str]:
    return [lab if c in ("real", "zero_weights") else f"{lab} ({d.frac3d_str(c, sex)})" for c, lab in LADDER_3D]


# ----------------------------------------------------------------------------- figures
def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.6,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlesize": 8.2, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 0,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08, "savefig.facecolor": "white",
    })


def ladder_positions(n: int, gaps: Sequence[int] = GROUP_STARTS) -> np.ndarray:
    y, pos = [], 0.0
    for i in range(n):
        if i in gaps:
            pos += 0.55
        y.append(pos)
        pos += 1.0
    return np.array(y)


def ladder_panel(ax, labels, stats, title, xlabel, first: bool, gaps: Sequence[int] = GROUP_STARTS,
                 chance_x: Optional[float] = None, color: str = BLUE) -> np.ndarray:
    """Horizontal bars, real connectome first; the dashed line carries its value down the ladder."""
    y = ladder_positions(len(stats), gaps=gaps)
    means = [m for m, _ in stats]
    ax.barh(y, means, xerr=[s for _, s in stats], height=0.62, color=color, zorder=2,
            error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=0))
    ax.axvline(means[0], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    if chance_x is not None:
        ax.axvline(chance_x, color=ORANGE, lw=0.85, ls=":", zorder=3)
    span = max(abs(m) for m in means) or 1.0
    for yi, (m, s) in zip(y, stats):
        if abs(m) < 0.015 * span:  # a zero-length bar would read as missing data
            ax.text(max(0.0, m + s) + 0.02 * span, yi, "≈0", color=INK2, fontsize=6.4, va="center")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if first:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.tick_params(labelleft=False)
    return y


def fig_wiring_hierarchy(d: Data, path: Path) -> None:
    p1 = d.spec["pillar_1_wiring_hierarchy"]["summary"]
    gaps = (1, 3, 6)
    labels = []
    for c, lab, _ in HIERARCHY_CONDS:
        fr = p1[c]["fraction_edges_rewired"]["mean"]
        labels.append(lab if c == "real" else f"{lab} ({fmt_pct_frac(fr)})")

    leg_stats = [(p1[c]["steer_leg_mn_sep_hz"]["mean"], p1[c]["steer_leg_mn_sep_hz"]["sd"]) for c, _, _ in HIERARCHY_CONDS]
    cx_stats = [(p1[c]["cx_antipodal_mod"]["mean"], p1[c]["cx_antipodal_mod"]["sd"]) for c, _, _ in HIERARCHY_CONDS]
    mb_stats = [(100.0 * p1[c]["mb_kc_decode_acc"]["mean"], 100.0 * p1[c]["mb_kc_decode_acc"]["sd"]) for c, _, _ in HIERARCHY_CONDS]

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.55), sharey=True, gridspec_kw={"wspace": 0.18})
    ladder_panel(axes[0], labels, leg_stats, "a  Steering (PFL3 → leg MNs)", "Leg MN asymmetry (Hz)", True, gaps=gaps)
    y = ladder_panel(axes[1], labels, cx_stats, "b  Compass (EPG → Delta7/PEN)", "Antipodal cos modulation", False, gaps=gaps)
    ladder_panel(axes[2], labels, mb_stats, "c  Olfaction (ALPN → KCs)", "8-odor KC decoding (%)", False,
                 gaps=gaps, chance_x=12.5)
    axes[0].set_ylim(y[-1] + 1.08, -0.65)

    # Annotate Delta7/PEN mean rate drop at type + side vs instance
    tag_bg = dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.9)
    for idx_c in (0, 3, 4):
        c_key = HIERARCHY_CONDS[idx_c][0]
        hz = p1[c_key]["cx_mean_hz"]["mean"]
        m_val, s_val = cx_stats[idx_c]
        axes[1].text(m_val + s_val + 0.022, y[idx_c], f"{hz:.2f} Hz", color=INK2, fontsize=6.1,
                     va="center", zorder=4, bbox=tag_bg)
    axes[1].set_xlim(-0.26, 0.74)

    # Annotate KC sparsity collapse in region/full shuffles and 12.5% chance line
    for idx_c in (0, 1, 7, 8):
        c_key = HIERARCHY_CONDS[idx_c][0]
        sp = 100.0 * p1[c_key]["mb_kc_sparsity"]["mean"]
        m_val, s_val = mb_stats[idx_c]
        axes[2].text(m_val + s_val + 2.2, y[idx_c], f"{sp:.1f}% KCs", color=INK2, fontsize=6.0,
                     va="center", zorder=4, bbox=tag_bg)
    axes[2].text(15.0, y[-1] + 0.68, "12.5% chance", color=ORANGE, fontsize=6.1, va="center")
    axes[2].set_xlim(-4, 124)

    fig.savefig(path)
    plt.close(fig)


def fig_ladder_3d(d: Data, path: Path) -> None:
    conds = [c for c, _ in LADDER_3D]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey="row", gridspec_kw={"wspace": 0.08, "hspace": 0.5})
    for row, sex in enumerate(("male", "female")):
        speed = [mean_sd([r["mean_speed_mm_s"] for r in d.rows3d(sex, c)]) for c in conds]
        goal = [mean_sd([r["probe_goal_separation_hz"] for r in d.rows3d(sex, c)]) for c in conds]
        labels = ladder3d_labels(d, sex)
        letters = "ab" if row == 0 else "cd"
        ladder_panel(axes[row, 0], labels, speed, f"{letters[0]}  {sex.capitalize()}: walking speed", "Speed (mm/s)", True)
        ladder_panel(axes[row, 1], labels, goal, f"{letters[1]}  {sex.capitalize()}: goal steering",
                     "Turn readout separation (Hz)", False)
    axes[1, 0].set_xlim(-0.65, 13.2)
    fig.savefig(path)
    plt.close(fig)


def fig_dn_screen(d: Data, path: Path) -> None:
    p2 = d.spec["pillar_2_virtual_split_gal4_dn_screen"]
    top_leg = p2["top_15_steering_vnc_leg_bottlenecks"][:10]
    top_wing = p2["top_15_courtship_vnc_wing_bottlenecks"][:10]
    all_dn = p2["all_dn_types"]

    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(7.8, 3.05), gridspec_kw={"wspace": 0.42, "width_ratios": [1.0, 1.0, 1.26]}
    )

    # Panel a: Top 10 VNC leg-steering DN bottlenecks
    y = np.arange(len(top_leg))
    a.barh(y, [r["delta_vnc_leg_sep_hz"] for r in top_leg], xerr=[r["delta_vnc_leg_sep_sd"] for r in top_leg],
           height=0.62, color=BLUE, zorder=2, label="Real connectome",
           error_kw=dict(ecolor=INK2, elinewidth=0.75, capsize=0))
    a.scatter([r["full_null_delta_vnc_leg_sep_hz"] for r in top_leg], y, s=16, color=ORANGE, zorder=3, label="Full shuffle")
    a.axvline(0, color=AXIS, lw=0.8, zorder=1)
    a.set_yticks(y)
    a.set_yticklabels([r["dn_type"] for r in top_leg], fontsize=6.9)
    a.invert_yaxis()
    a.spines["left"].set_visible(False)
    a.set_xlim(-0.48, 3.38)
    a.grid(axis="x", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Δ Leg MN steering (Hz)")
    a.set_title("a  Leg-steering DNs")
    a.legend(loc="lower right", fontsize=6.1, handletextpad=0.3)

    # Panel b: Top 10 VNC wing-motor courtship DN bottlenecks (* = fru+/dsx+)
    b.barh(y, [r["delta_vnc_wing_mn_hz"] for r in top_wing], xerr=[r["delta_vnc_wing_mn_sd"] for r in top_wing],
           height=0.62, color=BLUE, zorder=2, error_kw=dict(ecolor=INK2, elinewidth=0.75, capsize=0))
    b.scatter([r["full_null_delta_vnc_wing_mn_hz"] for r in top_wing], y, s=16, color=ORANGE, zorder=3)
    b.axvline(0, color=AXIS, lw=0.8, zorder=1)
    b.set_yticks(y)
    b.set_yticklabels([f"{r['dn_type']}*" if r["is_fru_dsx_dimorphic"] else r["dn_type"] for r in top_wing], fontsize=6.9)
    b.invert_yaxis()
    b.spines["left"].set_visible(False)
    b.set_xlim(-0.14, 1.74)
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.set_xlabel("Δ Wing MN courtship (Hz)")
    b.set_title("b  Wing-song DNs (*fru/dsx)")

    # Panel c: VNC out-synapses vs causal impact across all 485 DN cell types
    xs = np.log10(1.0 + np.array([r["vnc_out_syn"] for r in all_dn], dtype=float))
    yl = np.array([r["delta_vnc_leg_sep_hz"] for r in all_dn], dtype=float)
    yw = np.array([r["delta_vnc_wing_mn_hz"] for r in all_dn], dtype=float)
    c.scatter(xs, yl, s=11, alpha=0.65, color=BLUE, edgecolors="none", zorder=2,
              label=f"Leg steering (r = {p2['hub_confound_audit']['pearson_r_vnc_syn_vs_leg_impact']:.2f})")
    c.scatter(xs, yw, s=11, alpha=0.65, color=ORANGE, marker="s", edgecolors="none", zorder=3,
              label=f"Wing song (r = {p2['hub_confound_audit']['pearson_r_vnc_syn_vs_wing_impact']:.2f})")
    c.axhline(0, color=AXIS, lw=0.8, zorder=1)
    by_name = {r["dn_type"]: r for r in all_dn}
    callouts = [
        ("DNa06", "DNa06", "delta_vnc_leg_sep_hz", (2.78, 2.82), "right", "center"),
        ("DNg75", "DNg75", "delta_vnc_leg_sep_hz", (3.12, 1.76), "right", "center"),
        ("pIP10", "pIP10*", "delta_vnc_wing_mn_hz", (4.18, 2.12), "center", "bottom"),
        ("DNp31", "DNp31", "delta_vnc_wing_mn_hz", (3.28, 0.96), "right", "center"),
        ("DNg100", "DNg100", "delta_vnc_leg_sep_hz", (4.42, 1.00), "center", "bottom"),
    ]
    for nm, disp, key, xy_txt, ha, va in callouts:
        r = by_name[nm]
        px = np.log10(1.0 + r["vnc_out_syn"])
        py = r[key]
        c.annotate(disp, xy=(px, py), xytext=xy_txt, fontsize=6.2, color=INK, ha=ha, va=va,
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=0.55))
    tick_syn = np.array([0, 10, 100, 1000, 10000], dtype=float)
    c.set_xticks(np.log10(1.0 + tick_syn))
    c.set_xticklabels(["0", "10", "100", "1k", "10k"])
    c.set_xlim(-0.22, 4.94)
    c.set_ylim(-0.82, 3.45)
    c.grid(color=GRID, lw=0.6)
    c.set_axisbelow(True)
    c.set_xlabel("VNC output synapses (log scale)")
    c.set_ylabel("Causal silencing effect (Hz)")
    c.set_title("c  Impact vs VNC synapses (485 DNs)")
    c.legend(loc="upper left", fontsize=6.1, handletextpad=0.25)

    fig.savefig(path)
    plt.close(fig)


def fig_sex_switch(d: Data, path: Path) -> None:
    p3 = d.spec["pillar_3_sex_switch_and_threshold_audit"]
    curve = p3["threshold_and_recall_audit"]["threshold_curve"]
    gate = p3["threshold_and_recall_audit"]["functional_gate_comparison"]
    male = p3["male_within_connectome_dissociation"]["summary"]
    fem = p3["female_within_connectome_dissociation"]["summary"]

    fig = plt.figure(figsize=(7.5, 5.25))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.25], hspace=0.48)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.42)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.56)

    a = fig.add_subplot(top[0, 0])
    b = fig.add_subplot(top[0, 1])
    c = fig.add_subplot(bot[0, 0])
    d_ax = fig.add_subplot(bot[0, 1])

    # Panel a: Threshold recall audit
    ths = [r["min_synapses"] for r in curve]
    xm = np.arange(len(ths))
    a.plot(xm, [r["male_edges"] / 1e6 for r in curve], color=BLUE, lw=1.6, marker="o", ms=4.5, label="MaleCNS (FIB-SEM)")
    a.plot(xm, [r["female_edges"] / 1e6 for r in curve], color=ORANGE, lw=1.6, marker="s", ms=4.5, label="BANC (TEM)")
    a.set_yscale("log")
    a.set_xlim(-0.25, 5.25)
    a.set_ylim(0.10, 48.0)
    a.set_xticks(xm)
    a.set_xticklabels([f"≥{t}" for t in ths])
    r3 = next(r for r in curve if r["min_synapses"] == 3)
    a.annotate(f"≥3: {r3['male_to_female_edge_ratio']:.2f}×\n(10.3M vs 2.9M)",
               xy=(2, r3["male_edges"] / 1e6), xytext=(2.4, 15.5),
               fontsize=6.2, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_xlabel("Effective weight cutoff |w|")
    a.set_ylabel("Directed connections (M, log)")
    a.set_title("a  Threshold-matched edge recall")
    a.legend(loc="lower left", fontsize=6.2, handletextpad=0.3)

    # Panel b: Functional brain-wide firing rate vs 0.02 Hz 3D walking gate
    gate_rows = [
        ("MaleCNS ≥1 (native)", gate["male_ge1_native"]["mean_brain_hz"], BLUE),
        ("MaleCNS ≥3 (pruned)", gate["male_ge3_threshold_matched"]["mean_brain_hz"], BLUE),
        ("BANC ≥3 (native)", gate["female_banc_ge3_native"]["mean_brain_hz"], ORANGE),
    ]
    yg = np.arange(len(gate_rows))
    b.barh(yg, [r[1] for r in gate_rows], height=0.52, color=[r[2] for r in gate_rows], zorder=2)
    b.set_xscale("log")
    b.set_xlim(0.005, 1.05)
    b.set_ylim(len(gate_rows) - 0.35, -0.65)
    b.axvline(0.02, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    b.text(0.0215, -0.46, "0.02 Hz 3D gate", color=INK2, fontsize=6.1, va="center")
    for yi, (_, val, _) in enumerate(gate_rows):
        b.text(val * 1.16, yi, f"{val:.3f} Hz", color=INK2, fontsize=6.2, va="center")
    b.set_yticks(yg)
    b.set_yticklabels([r[0] for r in gate_rows], fontsize=6.7)
    b.spines["left"].set_visible(False)
    b.grid(axis="x", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.set_xlabel("Brain-wide firing rate (Hz, log)")
    b.set_title("b  Threshold pruning vs 3D gate")

    # Panel c & d: Within-connectome fru/dsx double dissociation (% of real connectome)
    for ax, summ, court_key, court_lbl, title in (
        (c, male, "courtship_wing_mn_hz", "Courtship wing MNs", "c  Male fru/dsx dissociation"),
        (d_ax, fem, "receptivity_vpodn_hz", "Receptivity (vpoDN)", "d  Female fru/dsx dissociation"),
    ):
        nav0 = summ["real"]["nav_turn_sep_hz"]["mean"]
        crt0 = summ["real"][court_key]["mean"]
        y = ladder_positions(len(SEX_SWITCH_CONDS), gaps=(1, 3, 5))
        lbls = []
        for ck, clab in SEX_SWITCH_CONDS:
            fr = summ[ck]["fraction_edges_rewired"]["mean"]
            lbls.append("Real" if ck == "real" else f"{clab} ({fmt_pct_frac(fr)})")

        nav_pct = [100.0 * summ[ck]["nav_turn_sep_hz"]["mean"] / nav0 for ck, _ in SEX_SWITCH_CONDS]
        nav_sd = [100.0 * summ[ck]["nav_turn_sep_hz"]["sd"] / nav0 for ck, _ in SEX_SWITCH_CONDS]
        crt_pct = [100.0 * summ[ck][court_key]["mean"] / crt0 for ck, _ in SEX_SWITCH_CONDS]
        crt_sd = [100.0 * summ[ck][court_key]["sd"] / crt0 for ck, _ in SEX_SWITCH_CONDS]

        h = 0.33
        ax.barh(y - h / 2, nav_pct, xerr=nav_sd, height=h, color=BLUE, zorder=2, label="Navigation (PFL3)",
                error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=0))
        ax.barh(y + h / 2, crt_pct, xerr=crt_sd, height=h, color=ORANGE, zorder=2, label=court_lbl,
                error_kw=dict(ecolor=INK2, elinewidth=0.7, capsize=0))
        for yi, (nv, ns, cv, cs) in enumerate(zip(nav_pct, nav_sd, crt_pct, crt_sd)):
            if abs(nv) < 1.5:
                ax.text(max(0.0, nv + ns) + 2.2, y[yi] - h / 2, "≈0", color=BLUE, fontsize=5.5, va="center")
            if abs(cv) < 1.5:
                ax.text(max(0.0, cv + cs) + 2.2, y[yi] + h / 2, "≈0", color=ORANGE, fontsize=5.5, va="center")
        ax.axvline(100.0, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=3)
        ax.axvline(0.0, color=AXIS, lw=0.8, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(lbls, fontsize=6.4)
        ax.set_ylim(y[-1] + 1.45, -0.65)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(-8, 162)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("% of real-connectome response")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=5.9, handletextpad=0.3)

    fig.savefig(path)
    plt.close(fig)


FIGURES = {
    "fig1_wiring_specificity_hierarchy.png": fig_wiring_hierarchy,
    "fig2_embodied_steering_3d.png": fig_ladder_3d,
    "fig3_virtual_split_gal4_dn_screen.png": fig_dn_screen,
    "fig4_fru_dsx_sex_switch_and_threshold_audit.png": fig_sex_switch,
}


# ----------------------------------------------------------------------------- tables
def num(x: float, digits: int = 1) -> str:
    """Typographic minus, and no negative zero."""
    s = f"{x:.{digits}f}"
    if float(s) == 0:
        return f"{0:.{digits}f}"
    return s.replace("-", "−")


def pm(stat: Tuple[float, float], digits: int = 1) -> str:
    return f"{num(stat[0], digits)} ± {num(stat[1], digits)}"


def short_label(label: str) -> str:
    return label.replace("Random rewiring, matched", "Random, matched")


def make_tables(d: Data) -> Dict[str, Table]:
    # Table 1: Three-circuit wiring specificity hierarchy
    p1 = d.spec["pillar_1_wiring_hierarchy"]["summary"]
    rows_hier: List[Optional[List[str]]] = []
    for i, (c, lab, kept) in enumerate(HIERARCHY_CONDS):
        if i in (1, 3, 6):
            rows_hier.append(None)
        s = p1[c]
        rows_hier.append([
            lab,
            kept,
            fmt_pct_frac(s["fraction_edges_rewired"]["mean"]),
            pm((s["steer_goal_sep_hz"]["mean"], s["steer_goal_sep_hz"]["sd"]), 1),
            pm((s["steer_leg_mn_sep_hz"]["mean"], s["steer_leg_mn_sep_hz"]["sd"]), 2),
            pm((s["cx_antipodal_mod"]["mean"], s["cx_antipodal_mod"]["sd"]), 2),
            f"{num(100.0 * s['cx_decode_acc']['mean'], 1)}%",
            num(s["cx_mean_hz"]["mean"], 2),
            f"{num(100.0 * s['mb_kc_sparsity']['mean'], 1)}%",
            f"{num(100.0 * s['mb_kc_decode_acc']['mean'], 1)}%",
            f"{num(100.0 * s['mb_mbon_decode_acc']['mean'], 1)}%",
        ])

    # Table 2: Closed-loop 3D NeuroMechFly v2 walking and steering
    rows3d: List[Optional[List[str]]] = []
    for sex in ("male", "female"):
        if rows3d:
            rows3d.append(None)
        for c, lab in LADDER_3D:
            rr = d.rows3d(sex, c)
            rows3d.append([
                sex.capitalize() if c == "real" else "",
                short_label(lab),
                d.frac3d_str(c, sex),
                pm(mean_sd([r["mean_speed_mm_s"] for r in rr]), 2),
                f"{num(np.mean([r['target_progress_pct'] for r in rr]))}%",
                num(np.mean([r["mean_fwd_drive"] for r in rr]), 3),
                pm(mean_sd([r["probe_goal_separation_hz"] for r in rr]), 1),
            ])

    # Table 3: Parallel K=486 virtual Split-GAL4 DN screen
    p2 = d.spec["pillar_2_virtual_split_gal4_dn_screen"]
    by_dn = {r["dn_type"]: r for r in p2["all_dn_types"]}
    rows_dn: List[Optional[List[str]]] = []
    leg_show = [r["dn_type"] for r in p2["top_15_steering_vnc_leg_bottlenecks"][:6]] + ["DNb01"]
    wing_show = [r["dn_type"] for r in p2["top_15_courtship_vnc_wing_bottlenecks"][:6]]
    for grp_lbl, dn_list in (("Steering", leg_show), ("Courtship", wing_show)):
        if rows_dn:
            rows_dn.append(None)
        for j, tp in enumerate(dn_list):
            r = by_dn[tp]
            rows_dn.append([
                grp_lbl if j == 0 else "",
                tp,
                str(r["n_neurons"]),
                f"{int(round(r['vnc_out_syn'])):,d}",
                "yes" if r["is_fru_dsx_dimorphic"] else "no",
                pm((r["delta_vnc_leg_sep_hz"], r["delta_vnc_leg_sep_sd"]), 2),
                num(r["delta_turn_readout_hz"], 1),
                pm((r["delta_vnc_wing_mn_hz"], r["delta_vnc_wing_mn_sd"]), 2),
                num(r["type_side_null_delta_vnc_leg_sep_hz"] if grp_lbl == "Steering"
                    else r["type_side_null_delta_vnc_wing_mn_hz"], 2),
                num(r["full_null_delta_vnc_leg_sep_hz"] if grp_lbl == "Steering"
                    else r["full_null_delta_vnc_wing_mn_hz"], 2),
            ])

    # Table 4: Cross-dataset synapse threshold & recall audit
    p3 = d.spec["pillar_3_sex_switch_and_threshold_audit"]
    rows_th: List[Optional[List[str]]] = []
    for r in p3["threshold_and_recall_audit"]["threshold_curve"]:
        rows_th.append([
            f"≥ {r['min_synapses']}",
            f"{r['male_edges']:,d}",
            num(r["male_mean_out_deg"], 1),
            f"{r['female_edges']:,d}",
            num(r["female_mean_out_deg"], 1),
            f"{num(r['male_to_female_edge_ratio'], 2)}×",
        ])

    # Table 5: Symmetric within-connectome fru/dsx double dissociation
    rows_sx: List[Optional[List[str]]] = []
    msum = p3["male_within_connectome_dissociation"]["summary"]
    fsum = p3["female_within_connectome_dissociation"]["summary"]
    for i, (ck, clab) in enumerate(SEX_SWITCH_CONDS):
        if i in (1, 3, 5):
            rows_sx.append(None)
        sm = msum[ck]
        rows_sx.append([
            "Male" if i == 0 else "",
            clab,
            fmt_pct_frac(sm["fraction_edges_rewired"]["mean"]),
            pm((sm["nav_turn_sep_hz"]["mean"], sm["nav_turn_sep_hz"]["sd"]), 1),
            pm((sm["courtship_pip10_aroused_hz"]["mean"], sm["courtship_pip10_aroused_hz"]["sd"]), 1),
            pm((sm["courtship_pip10_unaroused_leak_hz"]["mean"], sm["courtship_pip10_unaroused_leak_hz"]["sd"]), 1),
            pm((sm["courtship_wing_mn_hz"]["mean"], sm["courtship_wing_mn_hz"]["sd"]), 2),
        ])
    rows_sx.append(None)
    for i, (ck, clab) in enumerate(SEX_SWITCH_CONDS):
        sf = fsum[ck]
        rows_sx.append([
            "Female" if i == 0 else "",
            clab,
            fmt_pct_frac(sf["fraction_edges_rewired"]["mean"]),
            pm((sf["nav_turn_sep_hz"]["mean"], sf["nav_turn_sep_hz"]["sd"]), 1),
            pm((sf["receptivity_vpodn_hz"]["mean"], sf["receptivity_vpodn_hz"]["sd"]), 1),
            "—",
            "—",
        ])

    return {
        "hierarchy": (
            "Multi-circuit wiring-specificity ladder in MaleCNS v1.0 (five rewire seeds × five noisy trials, "
            "20 Hz background input, uniform 20 ms membrane time constant). Steering reports both the "
            "eight-neuron-per-side descending-neuron readout and downstream VNC leg motor neurons (815 MNs). "
            "Compass metrics probe eight anatomical EPG wedges read out across 44 glomerular Delta7 and PEN "
            "instances (chance decoding: 12.5%). Olfactory metrics probe eight ALPN glomeruli across 4,064 "
            "Kenyon cells and 37 MBON cell types.",
            ["Condition", "Structure kept", "Rewired", "Steer DN (Hz)", "Leg MN (Hz)", "CX cos mod",
             "CX dec.", "CX rate", "KC active", "KC dec.", "MBON dec."],
            "lllrrrrrrrr", rows_hier,
        ),
        "ladder3d": (
            "Closed-loop 3D NeuroMechFly v2 walking and steering under the null-model ladder (single connectome, "
            "0.48 s episodes toward a 6.2 mm goal across MaleCNS v1.0 and Female BANC v888; n = 3 episodes for "
            "real and no-synapse controls, n = 9 across three rewire seeds for shuffled graphs). Walking speed "
            "and open-loop goal separation are mean ± s.d.; target progress and forward drive are episode means.",
            ["Sex", "Condition", "Rewired", "Speed (mm/s)", "Target progress", "Forward drive", "Goal signal (Hz)"],
            "llrrrrr", rows3d,
        ),
        "dnscreen": (
            "Parallel K = 486 virtual Split-GAL4 silencing screen across all 485 named descending-neuron cell "
            "types in MaleCNS v1.0 (five Common-Random-Numbers seeds; slot 0 intact wild type, slots 1–485 "
            "single-type acute silencing). Δ values report mean ± s.d. loss in downstream VNC leg-motor "
            "asymmetry (wild type: 3.06 Hz), in the eight-neuron-per-side turn readout (wild type: 60.9 Hz), "
            "and in VNC wing motor neurons (495 MNs; wild type: 3.21 Hz). The last two columns show the "
            "corresponding primary Δ (leg MN for steering, wing MN for courtship) when the screen is repeated "
            "on type + side shuffled and fully shuffled graphs.",
            ["Assay", "DN type", "n", "VNC syn.", "fru/dsx", "Δ Leg MN (Hz)", "Δ Turn DN",
             "Δ Wing MN (Hz)", "Type+side Δ", "Full null Δ"],
            "llrrlrrrrr", rows_dn,
        ),
        "threshold": (
            "Cross-dataset connection counts and mean out-degree across matched effective weight cutoffs |w| "
            "in MaleCNS v1.0 (167,565 neurons, isotropic 8 nm FIB-SEM) and Female BANC v888 (157,789 neurons, "
            "GridTape TEM).",
            ["Cutoff |w|", "MaleCNS edges", "Male out-deg.", "BANC edges", "Female out-deg.", "Male / female"],
            "lrrrrr", rows_th,
        ),
        "sexswitch": (
            "Within-connectome fru/dsx double dissociation in MaleCNS v1.0 (5,933 fru+/dsx+/dimorphic neurons, "
            "3.54% of CNS) and Female BANC v888 (2,718 neurons, 1.72% of CNS), five seeds per condition. "
            "In the male, columns report PFL3 steering separation, aroused pIP10 courtship command rate, "
            "unaroused sensory leak onto pIP10 (LC10a + VA1v without P1/pC1 input), and downstream VNC wing "
            "motor neuron rate. In the female, columns report PFL3 steering separation and vpoDN receptivity "
            "rate under pC1 + JO-B input.",
            ["Sex", "Condition", "Rewired", "Nav. steer (Hz)", "pIP10 / vpoDN (Hz)", "pIP10 leak (Hz)", "Wing MN (Hz)"],
            "lllrrrr", rows_sx,
        ),
    }


# ----------------------------------------------------------------------------- markdown
Block = tuple


def parse(md: str) -> Tuple[Dict[str, str], List[Block]]:
    meta: Dict[str, str] = {}
    body = md
    if md.startswith("---"):
        _, front, body = md.split("---", 2)
        for line in front.strip().splitlines():
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    blocks: List[Block] = []
    para: List[str] = []
    items: Optional[Tuple[str, List[str]]] = None

    def flush() -> None:
        nonlocal para, items
        if para:
            blocks.append(("p", " ".join(para)))
            para = []
        if items:
            blocks.append(items)
            items = None

    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if line.lstrip().startswith("<!--"):
            continue
        if m := re.match(r"^(#{1,2}) (.+)$", line):
            flush()
            blocks.append((f"h{len(m.group(1))}", m.group(2)))
        elif m := re.match(r"^!\[(.+)\]\((.+?)\)\{#(fig:[\w-]+)(?:\s+width=(\d+)%)?\}$", line):
            flush()
            blocks.append(("fig", m.group(2), m.group(1), m.group(3), int(m.group(4) or 100)))
        elif m := re.match(r"^\{\{table:([\w-]+)\}\}$", line):
            flush()
            blocks.append(("table", m.group(1)))
        elif m := re.match(r"^(- |\d+\. )(.+)$", line):
            kind = "ul" if m.group(1) == "- " else "ol"
            if para:
                blocks.append(("p", " ".join(para)))
                para = []
            if items is None or items[0] != kind:
                if items:
                    blocks.append(items)
                items = (kind, [])
            items[1].append(m.group(2))
        elif items and raw.startswith("  "):
            items[1][-1] += " " + line.strip()  # continuation of a list item
        else:
            if items:
                blocks.append(items)
                items = None
            para.append(line.strip())
    flush()
    return meta, blocks


def block_texts(block: Block, tables: Dict[str, Table]) -> List[str]:
    kind = block[0]
    if kind in ("h1", "h2", "p"):
        return [block[1]]
    if kind in ("ul", "ol"):
        return list(block[1])
    if kind == "fig":
        return [block[2]]
    if kind == "table":
        return [tables[block[1]][0]]
    return []


def cited_keys(blocks: List[Block], tables: Dict[str, Table]) -> List[str]:
    keys: List[str] = []
    for b in blocks:
        for text in block_texts(b, tables):
            for group in re.findall(r"\[(@[^\]]+)\]", text):
                for k in group.split(";"):
                    k = k.strip().lstrip("@")
                    if k not in keys:
                        keys.append(k)
    return keys


TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
UNICODE_TEX = {"×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$", "→": r"$\rightarrow$",
               "−": r"$-$", "≈": r"$\approx$", "τ": r"$\tau$", "Δ": r"$\Delta$", "μ": r"$\mu$", "µ": r"$\mu$",
               "ρ": r"$\rho$", "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "²": r"$^{2}$",
               "√": r"$\surd$", "÷": r"$\div$", "·": r"$\cdot$", "°": r"$^{\circ}$", "–": "--", "—": "---",
               "‘": "`", "’": "'", "“": "``", "”": "''", "…": r"\ldots{}", "ö": r"\"{o}", "Ö": r"\"{O}",
               "é": r"\'{e}", "ł": r"{\l}"}


def esc(text: str, tex: bool) -> str:
    if not tex:
        out = html.escape(text, quote=False)
        out = re.sub(r'"([^"]*)"', "“\\1”", out)
        return re.sub(r"(?<=\w)'(?=\w)", "’", out)
    out = "".join(TEX_SPECIAL.get(ch, ch) for ch in text)
    out = re.sub(r'"([^"]*)"', r"``\1''", out)
    for uni, macro in UNICODE_TEX.items():
        out = out.replace(uni, macro)
    return out


TOKEN = re.compile(
    r"(?P<code>`[^`]+`)|(?P<cite>\[@[^\]]+\])|(?P<ref>@(?:fig|tbl):[\w-]+)"
    r"|(?P<bold>\*\*.+?\*\*)|(?P<ital>(?<![\w*])\*(?!\s).+?(?<!\s)\*(?![\w*]))|(?P<url><https?://[^>\s]+>)"
)


def inline(text: str, ctx: dict, tex: bool) -> str:
    out: List[str] = []
    pos = 0
    for m in TOKEN.finditer(text):
        out.append(esc(text[pos:m.start()], tex))
        kind, tok = m.lastgroup, m.group()
        if kind == "code":
            body = esc(tok[1:-1], tex)
            out.append(rf"\texttt{{{body}}}" if tex else f"<code>{body}</code>")
        elif kind == "cite":
            keys = [k.strip().lstrip("@") for k in tok[1:-1].split(";")]
            if tex:
                out[-1] = out[-1].rstrip()
                out.append(r"~\cite{" + ",".join(keys) + "}")
            else:
                nums = sorted(ctx["cite"][k] for k in keys)
                out.append("[" + ",".join(f'<a href="#ref-{n}">{n}</a>' for n in nums) + "]")
        elif kind == "ref":
            label = tok[1:]
            is_fig = label.startswith("fig:")
            name = "Figure" if is_fig else "Table"
            num_id = (ctx["fig"] if is_fig else ctx["tbl"])[label]
            out.append(f"{name}~\\ref{{{label}}}" if tex else f'<a href="#{label}">{name} {num_id}</a>')
        elif kind == "bold":
            inner = inline(tok[2:-2], ctx, tex)
            out.append(rf"\textbf{{{inner}}}" if tex else f"<strong>{inner}</strong>")
        elif kind == "ital":
            inner = inline(tok[1:-1], ctx, tex)
            out.append(rf"\emph{{{inner}}}" if tex else f"<em>{inner}</em>")
        else:  # url
            url = tok[1:-1]
            out.append(rf"\url{{{url}}}" if tex else f'<a href="{html.escape(url)}">{html.escape(url)}</a>')
        pos = m.end()
    out.append(esc(text[pos:], tex))
    return "".join(out)


# ----------------------------------------------------------------------------- bibliography
def parse_bib(text: str) -> Dict[str, dict]:
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", text, re.S):
        fields = {f.group(1).lower(): f.group(2)
                  for f in re.finditer(r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", m.group(3))}
        entries[m.group(2)] = {"type": m.group(1).lower(), **fields}
    return entries


def detex(s: str) -> str:
    for macro, uni in ((r'{\"O}', "Ö"), (r'{\"o}', "ö"), (r'\"O', "Ö"), (r'\"o', "ö"), (r"{\l}", "ł"),
                       (r"{\'e}", "é"), (r"\'e", "é")):
        s = s.replace(macro, uni)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    return s.replace("{", "").replace("}", "").replace("--", "–")


def fmt_authors(authors: str) -> str:
    names = [n.strip() for n in authors.split(" and ") if n.strip()]
    others = any(n.lower() == "others" for n in names)
    names = [n for n in names if n.lower() != "others"]

    def short(name: str) -> str:
        if "," in name:
            last, first = (x.strip() for x in name.split(",", 1))
        else:
            parts = name.split()
            last, first = parts[-1], " ".join(parts[:-1])
        initials = "".join(p[0] for p in re.split(r"[\s.\-]+", first) if p)
        return f"{last} {initials}".strip()

    shown = ", ".join(short(n) for n in names[:3])
    return shown + (" et al." if others or len(names) > 3 else "")


def fmt_ref(e: dict) -> str:
    authors = html.escape(fmt_authors(detex(e.get("author", ""))))
    authors += "" if authors.endswith(".") else "."
    title = html.escape(detex(e.get("title", "")))
    if e["type"] == "misc":
        url = detex(e.get("howpublished", ""))
        return f'{authors} {title}. <a href="{url}">{url}</a> ({html.escape(detex(e.get("note", "")))}).'
    out = f"{authors} {title}. <i>{html.escape(detex(e.get('journal', '')))}</i>"
    if e.get("volume"):
        out += f" <b>{e['volume']}</b>"
    if e.get("pages"):
        out += f", {detex(e['pages'])}"
    out += f" ({e.get('year', '')})."
    if e.get("doi"):
        out += f' <a href="https://doi.org/{e["doi"]}">doi:{e["doi"]}</a>'
    return out


# ----------------------------------------------------------------------------- renderers
TEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.3cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}
\usepackage{amsmath,amssymb}
\usepackage{cite}
\usepackage{url}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[hidelinks]{hyperref}
"""


def tex_table(name: str, table: Table, ctx: dict) -> str:
    caption, header, align, rows = table
    lines = [r"\toprule", " & ".join(esc(h, True) for h in header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(r"\midrule" if row is None else " & ".join(inline(c, ctx, True) for c in row) + r" \\")
    lines.append(r"\bottomrule")
    return "\n".join([
        r"\begin{table}[t]", r"\centering\small",
        rf"\caption{{{inline(caption, ctx, True)}}}\label{{tbl:{name}}}",
        r"\begin{adjustbox}{max width=\textwidth}", rf"\begin{{tabular}}{{{align}}}",
        *lines, r"\end{tabular}", r"\end{adjustbox}", r"\end{table}",
    ])


def render_tex(meta: Dict[str, str], blocks: List[Block], tables: Dict[str, Table], ctx: dict) -> str:
    md = partial(inline, ctx=ctx, tex=True)
    author_lines = [rf"\textbf{{{md(meta['author'])}}}"]
    if meta.get("affiliation"):
        author_lines.append(rf"\small {md(meta['affiliation'])}")
    if meta.get("orcid"):
        oid = esc(meta["orcid"], True)
        author_lines.append(rf"\small \href{{https://orcid.org/{meta['orcid']}}}{{ORCID: {oid}}}")
    author_tex = r" \\ ".join(author_lines)
    out = [TEX_PREAMBLE, rf"\title{{{md(meta['title'])}}}", rf"\author{{{author_tex}}}",
           rf"\date{{{md(meta.get('date', ''))}}}", r"\begin{document}", r"\maketitle", ""]
    in_abstract = False
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            if in_abstract:
                out.append(r"\end{abstract}")
                in_abstract = False
            if b[1] == "Abstract":
                out.append(r"\begin{abstract}")
                in_abstract = True
            else:
                star = "*" if b[1] in UNNUMBERED else ""
                out.append(rf"\section{star}{{{md(b[1])}}}")
        elif kind == "h2":
            out.append(rf"\subsection{{{md(b[1])}}}")
        elif kind == "p":
            out.append(md(b[1]) + "\n")
        elif kind in ("ul", "ol"):
            env = "itemize" if kind == "ul" else "enumerate"
            out.append("\n".join([rf"\begin{{{env}}}", *(rf"  \item {md(i)}" for i in b[1]), rf"\end{{{env}}}"]))
        elif kind == "fig":
            _, path, caption, label, width = b
            out.append("\n".join([r"\begin{figure}[t]", r"\centering",
                                  rf"\includegraphics[width={width / 100:.2f}\textwidth]{{{path}}}",
                                  rf"\caption{{{md(caption)}}}\label{{{label}}}", r"\end{figure}"]))
        elif kind == "table":
            out.append(tex_table(b[1], tables[b[1]], ctx))
    if in_abstract:
        out.append(r"\end{abstract}")
    out += [r"\bibliographystyle{unsrt}", r"\bibliography{references}", r"\end{document}"]
    return "\n".join(out) + "\n"


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%%TITLE_TEXT%%</title>
<style>
@page { size: A4; }
html { background: #fff; }
body { margin: 0; color: #111; font: 10.3pt/1.5 Georgia, "Times New Roman", serif; font-variant-numeric: lining-nums; }
.sans, h1, h2, h3, .byline, figcaption, caption, table { font-family: system-ui, "Segoe UI", Arial, sans-serif; }
h1.title { font-size: 16.5pt; line-height: 1.28; font-weight: 650; margin: 0 0 6pt; }
.byline { font-size: 9.5pt; line-height: 1.45; color: #333; margin-bottom: 14pt; }
.byline a { color: #1c5cab; }
section.abstract { margin: 0 0 16pt; padding: 9pt 12pt; border-left: 3px solid #2a78d6; background: #f6f6f3; }
section.abstract h2 { margin: 0 0 4pt; font-size: 9pt; text-transform: uppercase; letter-spacing: 0.6pt; }
section.abstract p { margin: 0; }
h2 { font-size: 12pt; font-weight: 650; margin: 16pt 0 6pt; break-after: avoid; }
h3 { font-size: 10.5pt; font-weight: 650; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
figure { margin: 12pt auto 14pt; break-inside: avoid; }
figure img { display: block; width: 100%; }
figcaption { font-size: 8.7pt; line-height: 1.42; color: #222; margin-top: 6pt; }
table { width: 100%; border-collapse: collapse; margin: 10pt 0 14pt; font-size: 8.0pt; line-height: 1.33; break-inside: avoid; }
caption { caption-side: top; text-align: left; font-size: 8.7pt; line-height: 1.42; margin-bottom: 5pt; color: #222; }
thead th { border-top: 1.2px solid #111; border-bottom: 0.7px solid #111; font-weight: 650; padding: 3pt 4.2pt; text-align: left; vertical-align: bottom; }
td { padding: 2.1pt 4.2pt; vertical-align: top; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 1.2px solid #111; }
tr.sep td { border-top: 0.6px solid #999; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
code { font: 0.86em Consolas, "Courier New", monospace; }
a { color: inherit; text-decoration: none; }
ol.refs { padding-left: 20pt; font-size: 8.9pt; line-height: 1.4; }
ol.refs li { margin-bottom: 3pt; }
ol.refs a { color: #1c5cab; overflow-wrap: anywhere; }
</style>
</head>
<body>
<h1 class="title">%%TITLE%%</h1>
<div class="byline">%%BYLINE%%</div>
%%BODY%%
</body>
</html>
"""


def html_table(name: str, table: Table, ctx: dict) -> str:
    caption, header, align, rows = table
    num_id = ctx["tbl"][f"tbl:{name}"]

    def cls(i: int) -> str:
        return ' class="num"' if align[i] == "r" else ""

    head = "".join(f"<th{cls(i)}>{html.escape(h)}</th>" for i, h in enumerate(header))
    body, rule = [], False
    for row in rows:
        if row is None:
            rule = True
            continue
        cells = "".join(f"<td{cls(i)}>{inline(c, ctx, False)}</td>" for i, c in enumerate(row))
        attr = ' class="sep"' if rule else ""
        body.append(f"<tr{attr}>{cells}</tr>")
        rule = False
    return (f'<table id="tbl:{name}"><caption><b>Table {num_id}.</b> {inline(caption, ctx, False)}</caption>'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


def render_html(meta: Dict[str, str], blocks: List[Block], tables: Dict[str, Table], ctx: dict,
                bib: Dict[str, dict], cites: List[str]) -> str:
    md = partial(inline, ctx=ctx, tex=False)
    parts: List[str] = []
    sec = sub = 0
    in_abstract = False
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            if in_abstract:
                parts.append("</section>")
                in_abstract = False
            if b[1] == "Abstract":
                parts.append('<section class="abstract"><h2>Abstract</h2>')
                in_abstract = True
            elif b[1] in UNNUMBERED:
                parts.append(f"<h2>{md(b[1])}</h2>")
            else:
                sec, sub = sec + 1, 0
                parts.append(f"<h2>{sec}&nbsp;&nbsp;{md(b[1])}</h2>")
        elif kind == "h2":
            sub += 1
            parts.append(f"<h3>{sec}.{sub}&nbsp;&nbsp;{md(b[1])}</h3>")
        elif kind == "p":
            parts.append(f"<p>{md(b[1])}</p>")
        elif kind in ("ul", "ol"):
            parts.append(f"<{kind}>" + "".join(f"<li>{md(i)}</li>" for i in b[1]) + f"</{kind}>")
        elif kind == "fig":
            _, path, caption, label, width = b
            num_id = ctx["fig"][label]
            parts.append(f'<figure id="{label}" style="width:{width}%"><img src="{path}" alt="Figure {num_id}">'
                         f"<figcaption><b>Figure {num_id}.</b> {md(caption)}</figcaption></figure>")
        elif kind == "table":
            parts.append(html_table(b[1], tables[b[1]], ctx))
    if in_abstract:
        parts.append("</section>")
    refs = "".join(f'<li id="ref-{i + 1}">{fmt_ref(bib[k])}</li>' for i, k in enumerate(cites))
    parts.append(f'<h2>References</h2><ol class="refs">{refs}</ol>')
    by_items = [f"<strong>{md(meta['author'])}</strong>"]
    if meta.get("affiliation"):
        by_items.append(md(meta["affiliation"]))
    if meta.get("orcid"):
        oid = html.escape(meta["orcid"])
        by_items.append(f'<a href="https://orcid.org/{oid}">ORCID: {oid}</a>')
    if meta.get("date"):
        by_items.append(md(meta["date"]))
    byline = " · ".join(by_items)
    return (HTML_TEMPLATE.replace("%%TITLE_TEXT%%", html.escape(re.sub(r"[*`]", "", meta["title"])))
            .replace("%%TITLE%%", md(meta["title"])).replace("%%BYLINE%%", byline)
            .replace("%%BODY%%", "\n".join(parts)))


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    footer = ('<div style="font:8px system-ui,sans-serif;width:100%;text-align:center;color:#888">'
              '<span class="pageNumber"></span></div>')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(path=str(pdf_path), format="A4", print_background=True, display_header_footer=True,
                 header_template="<span></span>", footer_template=footer,
                 margin={"top": "17mm", "bottom": "19mm", "left": "18mm", "right": "18mm"})
        browser.close()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the manuscript from paper/manuscript.md.")
    ap.add_argument("--no-pdf", action="store_true", help="skip printing the PDF with headless Chromium")
    args = ap.parse_args()

    data = Data()
    style()
    for name, draw in FIGURES.items():
        draw(data, FIGS / name)
    tables = make_tables(data)

    meta, blocks = parse((PAPER / "manuscript.md").read_text(encoding="utf-8"))
    bib = parse_bib((PAPER / "references.bib").read_text(encoding="utf-8"))
    cites = cited_keys(blocks, tables)
    missing = [k for k in cites if k not in bib]
    if missing:
        raise SystemExit(f"cited but not in references.bib: {missing}")
    for b in blocks:
        if b[0] == "fig" and not (PAPER / b[1]).exists():
            raise SystemExit(f"missing figure: {b[1]}")
        if b[0] == "table" and b[1] not in tables:
            raise SystemExit(f"unknown table: {b[1]}")
    ctx = {
        "fig": {b[3]: i + 1 for i, b in enumerate(x for x in blocks if x[0] == "fig")},
        "tbl": {f"tbl:{b[1]}": i + 1 for i, b in enumerate(x for x in blocks if x[0] == "table")},
        "cite": {k: i + 1 for i, k in enumerate(cites)},
    }

    tex = render_tex(meta, blocks, tables, ctx)
    leftover = sorted({ch for ch in tex if ord(ch) > 127})
    if leftover:
        print(f"warning: unmapped non-ASCII characters in manuscript.tex: {leftover}")
    (PAPER / "manuscript.tex").write_text(tex, encoding="utf-8")

    html_path = PAPER / "manuscript.html"
    html_path.write_text(render_html(meta, blocks, tables, ctx, bib, cites), encoding="utf-8")
    if not args.no_pdf:
        print_pdf(html_path, PAPER / "manuscript.pdf")
    print(f"built {len(FIGURES)} figures, {len(tables)} tables, {len(cites)} references"
          f"{'' if args.no_pdf else ', manuscript.pdf'}")


if __name__ == "__main__":
    main()
