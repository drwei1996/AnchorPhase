"""SUPP DISCOVERY 01 — Step 2: gene-level paired Normal->Tumor harmonic remodeling analysis.

Inputs (all frozen, read-only):
  DATA/01_GENOMEWIDE_PATIENT_GENE_HARMONICS.csv.gz  (this supplement's own new fit table)
  fig4_DETERMINISTIC_COSINOR_v1/05_{Normal,Tumor,Shared}_rhythmic_gene_list.txt  (Fig4 classes)

Analyses:
  1. paired per-patient deltas: dc, ds, dA (amplitude change), ||delta||, circular phase difference
  2. eligibility: >=8 patients with detect_frac>=0.05 in BOTH conditions (predeclared family rule, mirrors V5)
  3. exact paired sign-flip test on the 2D harmonic vector (exhaustive 2^11), BH across eligible family
  4. descriptive remodeling categories (amplitude loss/gain, phase displacement) + robustness to amplitude threshold
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

import os as _os  # release patch: path de-hardcoded (see docs/port_notes.md)
ROOT = _os.environ.get('ANCHORPHASE_PROJECT_ROOT',
                        _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
G = ROOT + r'\16_FIG4_FIG5_APPLICATIONS\GSE166555'
OUT = ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_01\DATA'

rng = np.random.default_rng(20261023)
fit = pd.read_csv(OUT + r'\01_GENOMEWIDE_PATIENT_GENE_HARMONICS.csv.gz')

# --- Fig4 membership classes (frozen lists) ---
normal_set = set(open(G + r'\fig4_DETERMINISTIC_COSINOR_v1\05_Normal_rhythmic_gene_list.txt').read().split())
tumor_set = set(open(G + r'\fig4_DETERMINISTIC_COSINOR_v1\05_Tumor_rhythmic_gene_list.txt').read().split())
shared_set = set(open(G + r'\fig4_DETERMINISTIC_COSINOR_v1\05_Shared_rhythmic_gene_list.txt').read().split())


def fig4_class(gene):
    if gene in shared_set:
        return 'Shared'
    if gene in normal_set:
        return 'Normal-only'
    if gene in tumor_set:
        return 'Tumor-only'
    return 'Neither'


# --- pivot to paired patient table ---
N = fit[fit.condition == 'Normal']
T = fit[fit.condition == 'Tumor']
m = N.merge(T, on=['gene', 'patient_id'], suffixes=('_N', '_T'))
m['A_N'] = np.hypot(m.beta_cos_N, m.beta_sin_N)
m['A_T'] = np.hypot(m.beta_cos_T, m.beta_sin_T)
m['dc'] = m.beta_cos_T - m.beta_cos_N
m['ds'] = m.beta_sin_T - m.beta_sin_N
m['dA'] = m.A_T - m.A_N
m['ok'] = (m.detect_frac_N >= 0.05) & (m.detect_frac_T >= 0.05)
pairs = m[m.ok]
print('paired rows (eligible patient-gene pairs):', len(pairs), 'genes:', pairs.gene.nunique())

# --- per-gene summary + exact sign-flip test ---
SIGN_FLIPS = np.array(list(itertools.product([-1.0, 1.0], repeat=11)))  # 2048 x 11


def circdiff_h(a, b):
    """signed circular difference in hours between two phase angles given as (cos,sin) coefficient pairs"""
    phi_a = np.degrees(np.arctan2(a[1], a[0])) / 15.0
    phi_b = np.degrees(np.arctan2(b[1], b[0])) / 15.0
    d = (phi_a - phi_b + 12) % 24 - 12
    return d


rows = []
for gene, sub in pairs.groupby('gene'):
    sub = sub.sort_values('patient_id')
    n = len(sub)
    if n < 8:
        continue
    dc = sub.dc.to_numpy()
    ds = sub.ds.to_numpy()
    A_N = sub.A_N.to_numpy()
    A_T = sub.A_T.to_numpy()
    dA = sub.dA.to_numpy()
    sc, ss = dc.sum(), ds.sum()
    Tobs = np.hypot(sc, ss)
    if n == 11:
        proj = SIGN_FLIPS @ np.column_stack([dc, ds])
        Tperm = np.hypot(proj[:, 0], proj[:, 1])
        nperm = len(SIGN_FLIPS)
    else:
        fl = rng.choice([-1.0, 1.0], size=(20000, n))
        proj = fl @ np.column_stack([dc, ds])
        Tperm = np.hypot(proj[:, 0], proj[:, 1])
        nperm = 20000
    p = (1 + int((Tperm >= Tobs).sum())) / (1 + nperm)
    # circular mean phase displacement among patients with reliable amplitude BOTH conditions
    amp_min = 0.05
    rel = (A_N >= amp_min) & (A_T >= amp_min)
    if rel.sum() >= 6:
        ang = np.arctan2(ds[rel], dc[rel])
        # displacement of the harmonic vector direction (per-patient) relative to preserved-phase expectation
        # For phase displacement between conditions use each patient's fitted acrophase shift:
        phi_N = np.arctan2(sub.beta_sin_N.to_numpy()[rel], sub.beta_cos_N.to_numpy()[rel])
        phi_T = np.arctan2(sub.beta_sin_T.to_numpy()[rel], sub.beta_cos_T.to_numpy()[rel])
        dphi = np.degrees((np.angle(np.exp(1j * (phi_T - phi_N))))) / 15.0  # hours, wrapped
        mean_dphi = float(np.degrees(np.angle(np.exp(1j * np.radians(dphi * 15)))).mean() / 15.0)
        var_dphi = float(np.abs(np.exp(1j * np.radians(dphi * 15)).mean()))
        n_phase = int(rel.sum())
        frac_amp_gain = float((dA[rel] > 0).mean())
        frac_dA_pos = float((dA > 0).mean())
        mean_A_N = float(A_N.mean())
        mean_A_T = float(A_T.mean())
        mean_dA = float(dA.mean())
        mean_norm = float(np.hypot(dc, ds).mean())
        rows.append(dict(gene=gene, n_pairs=n, n_phase=n_phase, mean_A_N=mean_A_N, mean_A_T=mean_A_T,
                         mean_dA=mean_dA, frac_dA_pos=frac_dA_pos, mean_vec_norm=mean_norm,
                         vec_norm_sum=Tobs, p_signflip=p,
                         mean_dphi_h=mean_dphi, phase_conc=var_dphi,
                         fig4_class=fig4_class(gene), frac_amp_gain_reliable=frac_amp_gain))
    else:
        rows.append(dict(gene=gene, n_pairs=n, n_phase=int(rel.sum()), mean_A_N=float(A_N.mean()),
                         mean_A_T=float(A_T.mean()), mean_dA=float(dA.mean()), frac_dA_pos=float((dA > 0).mean()),
                         mean_vec_norm=float(np.hypot(dc, ds).mean()), vec_norm_sum=Tobs, p_signflip=p,
                         mean_dphi_h=np.nan, phase_conc=np.nan, fig4_class=fig4_class(gene),
                         frac_amp_gain_reliable=np.nan))

res = pd.DataFrame(rows)
# BH across eligible family
res['q_BH'] = pd.Series(res.p_signflip.values).rank(method='average').to_numpy() * np.nan
o = np.argsort(res.p_signflip.to_numpy())
p_sorted = res.p_signflip.to_numpy()[o]
mtot = len(res)
q = mtot * p_sorted / np.arange(1, mtot + 1)
q = np.minimum.accumulate(q[::-1])[::-1]
res.loc[res.index[o], 'q_BH'] = np.minimum(q, 1.0)

res.to_csv(OUT + r'\02_GENE_LEVEL_REMODELING_STATS.csv', index=False)
pairs.to_csv(OUT + r'\03_PATIENT_GENE_PAIRED_DELTAS.csv.gz', index=False, compression={'method': 'gzip', 'compresslevel': 6})

summary = {
    'family_rule': 'genes with >=8 paired patients where detect_frac>=0.05 in both conditions',
    'n_eligible_genes': int(len(res)),
    'n_q_lt_005': int((res.q_BH < 0.05).sum()),
    'n_q_lt_025': int((res.q_BH < 0.25).sum()),
    'class_counts': res.fig4_class.value_counts().to_dict(),
    'median_mean_dA_by_class': res.groupby('fig4_class').mean_dA.median().to_dict(),
    'test': 'exact paired sign-flip on summed 2D harmonic vector; exhaustive 2^11 when n=11 else 20000 MC; p=(1+#ge)/(1+nperm)',
    'multiple_testing': 'BH across the full eligible family',
    'phase_rule': 'phase displacement summarized only when per-patient amplitude>=0.05 log1p(CP10K) in BOTH conditions (>=6 such patients)',
}
with open(OUT + r'\04_ANALYSIS_SUMMARY.json', 'w') as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))
