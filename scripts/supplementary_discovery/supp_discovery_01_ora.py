"""SUPP DISCOVERY 01 — Step 3: descriptive categories, ORA of remodeling extremes, robustness stats."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import os as _os  # release patch: path de-hardcoded (see docs/port_notes.md)
ROOT = _os.environ.get('ANCHORPHASE_PROJECT_ROOT',
                        _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
G = ROOT + r'\16_FIG4_FIG5_APPLICATIONS\GSE166555'
DATA = ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_01\DATA'

res = pd.read_csv(DATA + r'\02_GENE_LEVEL_REMODELING_STATS.csv')

# ---- descriptive categories (high-amplitude subset A_N>=0.05) ----
hi = res[res.mean_A_N >= 0.05].copy()
hi['category'] = 'stable'
hi.loc[hi.mean_dA <= -0.05, 'category'] = 'amplitude loss (dA<=-0.05)'
hi.loc[hi.mean_dA >= 0.05, 'category'] = 'amplitude gain (dA>=+0.05)'
rel = hi[hi.n_phase >= 8].copy()
rel['category_p'] = 'phase preserved (|dphi|<3h)'
rel.loc[rel.mean_dphi_h.abs() >= 3, 'category_p'] = 'phase displaced (|dphi|>=3h)'

print('high-amplitude genes:', len(hi))
print(hi.category.value_counts())
print('\nreliable-phase subset:', len(rel))
print(rel.category_p.value_counts())

# ---- ORA of amplitude-loss vs amplitude-gain extremes ----
gmt_files = [(ROOT + r'\16_FIG4_FIG5_APPLICATIONS\data\h.all.v2026.1.Hs.symbols.gmt', 'Hallmark'),
             (ROOT + r'\16_FIG4_FIG5_APPLICATIONS\data\c2.cp.reactome.v2026.1.Hs.symbols.gmt', 'Reactome')]
sets = {}
for fp, coll in gmt_files:
    with open(fp) as fh:
        for line in fh:
            p = line.rstrip('\n').split('\t')
            sets[(coll, p[0])] = set(p[2:])

background = set(hi.gene)
loss_set = set(hi.loc[hi.mean_dA <= -0.05, 'gene'])
gain_set = set(hi.loc[hi.mean_dA >= 0.05, 'gene'])
print('\nORA: loss set', len(loss_set), 'gain set', len(gain_set), 'background', len(background))

rows = []
for qname, qset in [('amplitude_loss', loss_set), ('amplitude_gain', gain_set)]:
    for (coll, pw), members in sets.items():
        mem = members & background
        if len(mem) < 10 or len(mem) > 500:
            continue
        k = len(mem & qset)
        if k < 3:
            continue
        p = hypergeom.sf(k - 1, len(background), len(qset), len(mem))
        rows.append(dict(query_set=qname, collection=coll, pathway=pw, n_members_bg=len(mem), n_overlap=k,
                         expected=len(qset) * len(mem) / len(background),
                         fold_enrichment=k / (len(qset) * len(mem) / len(background)), p=p))
ora = pd.DataFrame(rows)
ora['q'] = np.nan
for q in ora['query_set'].unique():
    sel = ora['query_set'] == q
    o = np.argsort(ora.loc[sel, 'p'].to_numpy())
    ps = ora.loc[sel, 'p'].to_numpy()[o]
    m = sel.sum()
    qs = m * ps / np.arange(1, m + 1)
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    idxs = ora.index[sel][o]
    ora.loc[idxs, 'q'] = np.minimum(qs, 1.0)
ora.sort_values(['query_set', 'p']).to_csv(DATA + r'\05_REMODELING_EXTREME_ORA.csv', index=False)
print('\nTop ORA amplitude loss:')
print(ora[ora.query_set == 'amplitude_loss'].sort_values('p').head(12).to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print('\nTop ORA amplitude gain:')
print(ora[ora.query_set == 'amplitude_gain'].sort_values('p').head(12).to_string(index=False, float_format=lambda v: f'{v:.3g}'))

# ---- class-level summary + bootstrap CIs ----
rng = np.random.default_rng(20261023)
boot = []
for cls, sub in hi.groupby('fig4_class'):
    dA = sub.mean_dA.to_numpy()
    med = np.median(dA)
    bs = np.array([np.median(rng.choice(dA, len(dA))) for _ in range(2000)])
    boot.append(dict(fig4_class=cls, n=len(sub), median_dA=med, ci_low=np.quantile(bs, .025), ci_high=np.quantile(bs, .975),
                     frac_loss=float((sub.category != 'stable').mean() * (sub.mean_dA < 0).mean())))
pd.DataFrame(boot).to_csv(DATA + r'\06_CLASS_LEVEL_AMPLITUDE_CHANGE.csv', index=False)
print('\nclass-level median dA (A_N>=0.05 subset):')
print(pd.DataFrame(boot).to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# ---- robustness of 'phase preserved' to amplitude threshold ----
rob = []
for amp in (0.03, 0.05, 0.08):
    s = res[(res.mean_A_N >= amp) & (res.mean_A_T >= amp) & (res.n_phase >= 6)]
    rob.append(dict(amp_min=amp, n_genes=len(s),
                    median_abs_dphi=float(s.mean_dphi_h.abs().median()),
                    frac_abs_dphi_gt3=float((s.mean_dphi_h.abs() > 3).mean()),
                    frac_abs_dphi_gt4=float((s.mean_dphi_h.abs() > 4).mean())))
pd.DataFrame(rob).to_csv(DATA + r'\07_PHASE_PRESERVATION_ROBUSTNESS.csv', index=False)
print('\nphase-preservation robustness:')
print(pd.DataFrame(rob).to_string(index=False, float_format=lambda v: f'{v:.4f}'))

# ---- clock-gene control summary ----
core = ['ARNTL', 'BHLHE41', 'CLOCK', 'CRY1', 'CRY2', 'DBP', 'FBXO21', 'HLF', 'NFIL3', 'NPAS2', 'NR1D1', 'NR1D2', 'PER1', 'PER2', 'PER3', 'RORC', 'TEF']
cg = res[res.gene.isin(core)]
cg[['gene', 'mean_A_N', 'mean_A_T', 'mean_dA', 'mean_dphi_h', 'phase_conc', 'n_phase']].to_csv(DATA + r'\08_CORE17_CONTROL_SUMMARY.csv', index=False)
print('\ncore genes with phase data:', cg[cg.n_phase >= 6][['gene', 'mean_dphi_h', 'phase_conc']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
