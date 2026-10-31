#!/usr/bin/env python3
"""Render all 16 KATv2 manuscript panels using the layout-safe V2 scripts.

Install/usage:
  1) Copy these *_v2.py files plus fig_style_v2.py and this runner into
     the existing sckat_server_package/scripts/ directory.
  2) Run: python render_all_panels_v2.py

Outputs go to <KATV2_ROOT>/figures/panels_v2 and do not overwrite the
scientifically frozen panels in figures/panels.
"""
import os
from pathlib import Path
import subprocess, sys, os, time

HERE = Path(__file__).resolve().parent
PANELS = [
    "make_fig1a_inputs_preprocessing_v2.py",
    "make_fig1b_corrected_b0_model_v2.py",
    "make_fig1c_posterior_outputs_uncertainty_v2.py",
    "make_fig1d_optional_safe_cpi_v2.py",
    "make_fig2a_model_development_timeline_v2.py",
    "make_fig2b_development_performance_v2.py",
    "make_fig2c_external_validation_summary_v2.py",
    "make_fig2d_final_model_verdict_v2.py",
    "make_fig3a_liver_success_genes_v2.py",
    "make_fig3b_dermis_failure_genes_v2.py",
    "make_fig3c_falsification_controls_v2.py",
    "make_fig3d_confounding_mechanism_schematic_v2.py",
    "make_fig4a_tabula_application_framework_v2.py",
    "make_fig4b_phase_to_sixarm_v2.py",
    "make_fig4c_clinical_translation_framework_v2.py",
    "make_fig4d_drug_target_timing_concept_v2.py",
]

# Guard against running the overlay from a detached folder where empirical
# dependencies/data are unavailable.
if not (HERE / "fig3_shared.py").exists():
    print("NOTE: fig3_shared.py is not in this overlay package.")
    print("Copy the V2 files into your existing sckat_server_package/scripts/ folder")
    print("before rendering all 16 panels. The original fig3_shared.py remains unchanged.")

failed = []
t0 = time.time()
for i, script in enumerate(PANELS, 1):
    p = HERE / script
    print(f"\n[{i:02d}/16] {script}")
    r = subprocess.run([sys.executable, str(p)], cwd=str(HERE), env=os.environ.copy())
    if r.returncode != 0:
        failed.append((script, r.returncode))

print("\n" + "=" * 72)
print(f"V2 render finished in {(time.time()-t0)/60:.1f} min")
if failed:
    print("FAILED:")
    for s, code in failed:
        print(f"  {s}: exit {code}")
    sys.exit(1)
print("ALL_16_V2_PANELS = COMPLETE")
