"""
Automated Invariant & Reproducibility Test Suite for Whole-CNS Drosophila Specificity
=====================================================================================
Verifies:
  1. Exact MaleCNS v1.0 (167,565 neurons, 25,623,478 edges) & Female BANC v888 (157,789 neurons, 3,036,600 edges) counts
  2. Shared-CSR memory compression across K state slots
  3. Event-driven active-row gather-scatter vs SciPy SpMM numerical parity
  4. Common-Random-Numbers (CRN) 0.000000 Hz bit-identity and Kir2.1+TNT acute silencing
  5. Nine-level Maslov-Sneppen degree/weight/sign invariants, three-circuit anatomy, 485-DN catalog & zero-norm decoder
  6. Closed-loop 3D NeuroMechFly v2 MuJoCo biomechanical coupling (815 male / 805 female VNC leg MNs)
  7. Empirical JSON invariants for all four manuscript parts
  8. Clean standalone repository gate (Figures 1-4, manuscript formats, Colab notebook, zero legacy references)
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.connectome_engine import ConnectomeDataset, WholeCNSEngine, resolve_female_connectome_dir
from src.structural_specificity import (
    CSRSnapshot,
    extract_dn_screen_catalog,
    extract_three_circuit_anatomy,
    leave_one_trial_out_cosine_metrics,
    rewire_fru_dsx_subgraph,
)
from src.vnc_neuromechfly_3d import EmbodiedNeuroMechFly3DArena, extract_vnc_motor_pools


class TestWholeCNSSpecificity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds_male = ConnectomeDataset()
        cls.ds_female = ConnectomeDataset(str(resolve_female_connectome_dir()))

    def test_01_male_and_female_connectome_exact_counts(self):
        self.assertEqual(self.ds_male.N, 167_565)
        self.assertEqual(self.ds_male.nnz, 25_623_478)
        self.assertEqual(self.ds_male.proofread_synapses, 124_174_455)

        self.assertEqual(self.ds_female.N, 157_789)
        self.assertEqual(self.ds_female.nnz, 3_036_600)
        self.assertEqual(self.ds_female.proofread_synapses, 23_553_107)

    def test_02_shared_csr_memory_compression(self):
        eng = WholeCNSEngine(
            self.ds_male, num_brains=100, device="cpu", enable_callosum=False, homogeneous_tau_ms=20.0, seed=42
        )
        mem = eng.memory_footprint_mb()
        ratio = mem["naive_unshared_equivalent_mb"] / mem["total_ram_or_vram_mb"]
        self.assertLess(mem["shared_connectome_csr_mb"], 250.0)
        self.assertGreater(ratio, 35.0)

    def test_03_event_driven_gather_scatter_vs_scipy_spmm_parity(self):
        N = self.ds_male.N
        K = 8
        W = self.ds_male.W_base_csr
        rng = np.random.default_rng(123)
        X = sparse.random(K, N, density=0.01, format="csr", random_state=rng, dtype=np.float32)

        ref = (X @ W).toarray()
        post = np.zeros((N, K), dtype=np.float32)
        X_csc = X.tocsc()
        active_u = np.flatnonzero(np.diff(X_csc.indptr) > 0)
        for u in active_u:
            s0, s1 = W.indptr[u], W.indptr[u + 1]
            if s1 > s0:
                cols = W.indices[s0:s1]
                vals = W.data[s0:s1][:, None]
                x_row = X_csc.getcol(u).toarray().ravel()[None, :]
                post[cols] += vals * x_row
        np.testing.assert_allclose(post.T, ref, rtol=1e-5, atol=1e-5)

    def test_04_crn_perturbation_engine_identity_and_silencing(self):
        eng = WholeCNSEngine(
            self.ds_male,
            num_brains=6,
            enable_callosum=False,
            homogeneous_tau_ms=20.0,
            common_random_numbers=True,
            seed=700,
        )
        bg_sub = self.ds_male.plastic_neurons[::16]
        res0 = eng.step_population(
            15,
            shared_stimuli=[(self.ds_male.idx_walk_tonic, 80.0), (self.ds_male.idx_pfl3_turnR, 250.0)],
            bg_poisson_hz=20.0,
            bg_subset_indices=bg_sub,
        )
        self.assertEqual(float(np.max(np.abs(res0["turn_hz_per_brain"] - res0["turn_hz_per_brain"][0]))), 0.0)
        self.assertEqual(float(np.max(np.abs(res0["mean_brain_hz"] - res0["mean_brain_hz"][0]))), 0.0)

        eng.set_perturbation_masks([None, self.ds_male.idx_dn_R, self.ds_male.idx_dn_L, None, None, None])
        eng.set_seed(701)
        eng.reset_state()
        res1 = eng.step_population(
            20,
            shared_stimuli=[(self.ds_male.idx_walk_tonic, 80.0), (self.ds_male.idx_pfl3_turnR, 250.0)],
            bg_poisson_hz=20.0,
            bg_subset_indices=bg_sub,
            readout_pools={"dn_R": self.ds_male.idx_dn_R},
        )
        for unpert_slot in (3, 4, 5):
            self.assertEqual(res1["turn_hz_per_brain"][unpert_slot], res1["turn_hz_per_brain"][0])
        self.assertEqual(float(res1["pool_hz"]["dn_R"][1]), 0.0)
        self.assertLess(float(res1["turn_hz_per_brain"][1]), float(res1["turn_hz_per_brain"][0]))

    def test_05_wiring_specificity_null_invariants_and_zero_norm_decoder(self):
        zero_met = leave_one_trial_out_cosine_metrics(np.zeros((5, 8, 44), dtype=np.float64))
        self.assertAlmostEqual(zero_met["decode_acc"], 0.125, places=6)
        self.assertEqual(zero_met["coding_contrast"], 0.0)

        anat = extract_three_circuit_anatomy(self.ds_male)
        self.assertEqual(len(anat.epg_wedges_8), 8)
        self.assertEqual(len(anat.cx_readout_pools), 44)
        self.assertEqual(len(anat.odor_glom_names), 8)
        self.assertEqual(len(anat.kc_indices), 4064)
        self.assertEqual(len(anat.mbon_readout_pools), 37)

        cat = extract_dn_screen_catalog(self.ds_male)
        self.assertEqual(len(cat.dn_types), 485)
        self.assertIn("pIP10", cat.dn_types)
        self.assertIn("DNa06", cat.dn_types)

        snap = CSRSnapshot.capture(self.ds_female)
        try:
            rep = rewire_fru_dsx_subgraph(self.ds_female, seed=11, mode="both")
            self.assertTrue(rep["out_degree_preserved"])
            self.assertTrue(rep["in_degree_preserved"])
            self.assertEqual(rep["per_row_sign_preserved"], 1.0)
            self.assertGreater(rep["edges_selected"], 5000)
            self.assertTrue(np.array_equal(np.sort(self.ds_female.W_base_csr.data), np.sort(snap.data)))
        finally:
            snap.restore(self.ds_female)
        self.assertTrue(np.array_equal(self.ds_female.W_base_csr.indices, snap.indices))
        self.assertTrue(np.array_equal(self.ds_female.W_base_csr.data, snap.data))

    def test_06_vnc_neuromechfly_3d_mujoco_coupling(self):
        anat_f = extract_vnc_motor_pools(self.ds_female)
        anat_m = extract_vnc_motor_pools(self.ds_male)
        self.assertEqual(anat_f.total_is_motor, 805)
        self.assertEqual(anat_f.vnc_motor_count, 805)
        self.assertEqual(len(anat_f.leg_muscle_pools), 42)
        self.assertEqual(len(anat_m.leg_muscle_pools), 42)
        self.assertGreaterEqual(anat_m.vnc_motor_count, 815)

        eng_f = WholeCNSEngine(self.ds_female, num_brains=1, enable_callosum=True, seed=413)
        ep3d = EmbodiedNeuroMechFly3DArena.run_single_fly_3d_episode(
            eng_f, target_xy=(4.5, 2.0), duration_s=0.16, frame_ms=8, capture_camera=False
        )
        self.assertGreater(ep3d["mean_speed_mm_s"], 3.0)
        self.assertGreater(ep3d["thorax_height_z_mm"], 0.6)
        self.assertLess(ep3d["roll_rms_deg"], 20.0)

    def test_07_empirical_results_json_invariants(self):
        spec = json.loads((PROJECT_ROOT / "results" / "structural_specificity_screen_results.json").read_text(encoding="utf-8"))
        r3d = json.loads((PROJECT_ROOT / "results" / "rewiring_3d_results.json").read_text(encoding="utf-8"))

        # Clean 3D results file contains only the null ladder
        self.assertEqual(list(r3d.keys()), ["ladder"])
        self.assertEqual(len(r3d["ladder"]["summary"]["male"]), 8)
        self.assertEqual(len(r3d["ladder"]["summary"]["female"]), 8)

        # Part 1: 3-tier specificity hierarchy
        p1 = spec["pillar_1_wiring_hierarchy"]["summary"]
        self.assertGreater(p1["rewired_pn_kc_random_claws"]["mb_kc_decode_acc"]["mean"], 0.60)
        self.assertGreater(p1["rewired_type_side"]["steer_leg_mn_sep_hz"]["mean"], 3.8)
        self.assertLess(p1["rewired_type"]["steer_leg_mn_sep_hz"]["mean"], 0.2)
        self.assertGreater(p1["rewired_instance"]["cx_antipodal_mod"]["mean"], 0.40)
        self.assertLessEqual(p1["rewired_type_side"]["cx_antipodal_mod"]["mean"], 0.25)
        self.assertLess(p1["real"]["cx_epg_persistence_ratio"]["mean"], 0.10)

        # Part 3: K = 486 virtual Split-GAL4 DN screen
        p2 = spec["pillar_2_virtual_split_gal4_dn_screen"]
        self.assertEqual(p2["n_dn_types_screened"], 485)
        self.assertEqual(p2["parallel_k_slots"], 486)
        self.assertEqual(p2["crn_unperturbed_max_diff_hz"], 0.0)
        self.assertEqual(p2["top_15_steering_vnc_leg_bottlenecks"][0]["dn_type"], "DNa06")
        self.assertEqual(p2["top_15_courtship_vnc_wing_bottlenecks"][0]["dn_type"], "pIP10")
        self.assertLess(abs(p2["hub_confound_audit"]["pearson_r_vnc_syn_vs_leg_impact"]), 0.15)
        self.assertLess(abs(p2["hub_confound_audit"]["pearson_r_vnc_syn_vs_wing_impact"]), 0.15)

        # Part 4: Threshold recall audit & within-connectome fru/dsx double dissociation
        p3 = spec["pillar_3_sex_switch_and_threshold_audit"]
        r3 = next(r for r in p3["threshold_and_recall_audit"]["threshold_curve"] if r["min_synapses"] == 3)
        self.assertAlmostEqual(r3["male_to_female_edge_ratio"], 3.57, places=2)
        ms = p3["male_within_connectome_dissociation"]["summary"]
        fs = p3["female_within_connectome_dissociation"]["summary"]
        self.assertGreater(ms["rewired_fru_touch"]["nav_turn_sep_hz"]["mean"], 50.0)
        self.assertLess(ms["rewired_fru_touch"]["courtship_wing_mn_hz"]["mean"], 0.8)
        self.assertLess(ms["rewired_all_isomorphic"]["nav_turn_sep_hz"]["mean"], 1.0)
        self.assertGreater(ms["rewired_all_isomorphic"]["courtship_wing_mn_hz"]["mean"], 3.0)
        self.assertGreater(fs["rewired_fru_touch"]["nav_turn_sep_hz"]["mean"], 40.0)
        self.assertLess(fs["rewired_fru_touch"]["receptivity_vpodn_hz"]["mean"], 1.5)
        self.assertLess(fs["rewired_all_isomorphic"]["nav_turn_sep_hz"]["mean"], 1.0)
        self.assertGreater(fs["rewired_all_isomorphic"]["receptivity_vpodn_hz"]["mean"], 30.0)

    def test_08_clean_standalone_repo_and_manuscript_integrity(self):
        for fig_name in (
            "fig1_wiring_specificity_hierarchy.png",
            "fig2_embodied_steering_3d.png",
            "fig3_virtual_split_gal4_dn_screen.png",
            "fig4_fru_dsx_sex_switch_and_threshold_audit.png",
        ):
            p = PROJECT_ROOT / "figures" / fig_name
            self.assertTrue(p.exists() and p.stat().st_size > 20_000, f"Missing figure: {fig_name}")

        self.assertFalse((PROJECT_ROOT / "figures" / "superseded").exists())
        self.assertFalse((PROJECT_ROOT / "dashboards").exists())

        for doc in ("manuscript.md", "manuscript.tex", "manuscript.html", "manuscript.pdf", "references.bib"):
            p = PROJECT_ROOT / "paper" / doc
            self.assertTrue(p.exists() and p.stat().st_size > 500, f"Missing paper file: {doc}")

        nb = PROJECT_ROOT / "colab" / "drosophila_whole_cns_specificity_colab.ipynb"
        self.assertTrue(nb.exists() and nb.stat().st_size > 2_000)

        forbidden = (
            "aquilon", "fossen", "maritime", "waterjet", "rudder", "usv",
            "amandeep-tridel", "malecns-100x-connectome",
            "500x", "100x", "published_renormalized", "lesion_renormalization",
            "earlier draft", "previous draft", "prior draft",
        )
        scan_files = [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "CITATION.cff",
            PROJECT_ROOT / ".zenodo.json",
            *sorted((PROJECT_ROOT / "src").glob("*.py")),
            *sorted((PROJECT_ROOT / "experiments").glob("*.py")),
            *sorted((PROJECT_ROOT / "scripts").glob("*.py")),
            *sorted((PROJECT_ROOT / "colab").glob("*.ipynb")),
            PROJECT_ROOT / "paper" / "manuscript.md",
            PROJECT_ROOT / "paper" / "manuscript.tex",
            PROJECT_ROOT / "paper" / "references.bib",
        ]
        for fpath in scan_files:
            low = fpath.read_text(encoding="utf-8").lower()
            for bad in forbidden:
                self.assertNotIn(bad, low, f"Forbidden term '{bad}' found in {fpath.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
