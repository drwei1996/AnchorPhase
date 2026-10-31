"""SUPP DISCOVERY 02 — Figure: coordinated suppression of the phase-structured translation program
and complementarity of membership-based vs remodeling-based pathway analyses (GSE166555).

Panels:
  A  pathway-level Normal vs Tumor module amplitude (1,466 tested pathways; module sign-flip significance)
  B  per-patient paired program amplitude: cytoplasmic translation programs vs mitochondrial-translation control
  C  within translation-elongation program: gene-level paired amplitude + acrophase retention
  D  complementarity: membership-ORA-significant vs module-remodeling-significant pathways (zero overlap)

All panels rendered twice (WITH_TEXT / NO_TEXT) from the same data.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os as _os  # release patch: path de-hardcoded (see docs/port_notes.md)
ROOT = _os.environ.get('ANCHORPHASE_PROJECT_ROOT',
                        _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))
G = ROOT + r'\16_FIG4_FIG5_APPLICATIONS\GSE166555'
BASE = ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_02'
DATA = BASE + r'\DATA'
LAB = BASE + r'\PANELS_LABELED'
NOTX = BASE + r'\PANELS_NO_TEXT'
for d in (DATA, LAB, NOTX, BASE + r'\COMPOSITE', BASE + r'\RUN_MANIFEST'):
    os.makedirs(d, exist_ok=True)

N_BLUE = '#2877ac'
T_ORANGE = '#d8753a'
GREEN = '#3b9890'
GRAY = '#c3c7cb'
DK = '#222222'
RED = '#985052'

mod = pd.read_csv(G + r'\SUPP_ALL_PATHWAYS_NOMINAL_P_ONLY\01_ALL_PATHWAY_PAIRED_HARMONIC_TESTS.csv')
pairs = pd.read_csv(ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_01\DATA\03_PATIENT_GENE_PAIRED_DELTAS.csv.gz',
                    usecols=['gene', 'patient_id', 'beta_cos_N', 'beta_sin_N', 'beta_cos_T', 'beta_sin_T', 'A_N', 'A_T'])
res_gene = pd.read_csv(ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_01\DATA\02_GENE_LEVEL_REMODELING_STATS.csv')


def load_gmt():
    sets = {}
    for fp, coll in [(ROOT + r'\16_FIG4_FIG5_APPLICATIONS\data\h.all.v2026.1.Hs.symbols.gmt', 'Hallmark'),
                     (ROOT + r'\16_FIG4_FIG5_APPLICATIONS\data\c2.cp.reactome.v2026.1.Hs.symbols.gmt', 'Reactome')]:
        with open(fp) as fh:
            for line in fh:
                p = line.rstrip('\n').split('\t')
                sets[p[0]] = set(x.upper() for x in p[2:])
    return sets


GMT = load_gmt()

PROGRAMS = [
    ('REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION', 'Translation elongation'),
    ('REACTOME_EUKARYOTIC_TRANSLATION_INITIATION', 'Translation initiation'),
    ('REACTOME_NONSENSE_MEDIATED_DECAY_NMD', 'Nonsense-mediated decay'),
    ('REACTOME_RESPONSE_OF_EIF2AK4_GCN2_TO_AMINO_ACID_DEFICIENCY', 'GCN2 amino-acid response'),
    ('REACTOME_RRNA_PROCESSING', 'rRNA processing'),
    ('REACTOME_SRP_DEPENDENT_COTRANSLATIONAL_PROTEIN_TARGETING_TO_MEMBRANE', 'SRP co-transl. targeting'),
    ('REACTOME_RIBOSOME_ASSOCIATED_QUALITY_CONTROL', 'Ribosome-assoc. QC'),
    ('REACTOME_MITOCHONDRIAL_TRANSLATION_ELONGATION', 'Mitochondr. translation (control)'),
]


def save_variant(fig, outdir, stem, suffix):
    fig.savefig(os.path.join(outdir, f'{stem}_{suffix}.png'), dpi=360, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(outdir, f'{stem}_{suffix}.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(outdir, f'{stem}_{suffix}.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def panel_a(with_text=True):
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    sig = mod.paired_p < 0.05
    ax.scatter(mod.loc[~sig, 'Normal_mean_individual_amplitude'], mod.loc[~sig, 'Tumor_mean_individual_amplitude'],
               s=8, c=GRAY, alpha=.55, edgecolors='none', rasterized=True,
               label='module P ≥ 0.05' if with_text else None)
    ax.scatter(mod.loc[sig, 'Normal_mean_individual_amplitude'], mod.loc[sig, 'Tumor_mean_individual_amplitude'],
               s=12, c=T_ORANGE, alpha=.8, edgecolors='none', rasterized=True,
               label='module paired P < 0.05 (n=201)' if with_text else None)
    lim = max(mod.Normal_mean_individual_amplitude.max(), mod.Tumor_mean_individual_amplitude.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=DK, lw=.8, ls='--')
    ann = {'REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION': ('Translation elongation', 6, -12),
           'REACTOME_EUKARYOTIC_TRANSLATION_INITIATION': ('Translation initiation', 6, 6),
           'REACTOME_NONSENSE_MEDIATED_DECAY_NMD': ('NMD', 6, 6),
           'REACTOME_RESPONSE_OF_EIF2AK4_GCN2_TO_AMINO_ACID_DEFICIENCY': ('GCN2 response', 6, -14),
           'REACTOME_DIGESTION': ('Digestion', 6, 6),
           'REACTOME_PASSIVE_TRANSPORT_BY_AQUAPORINS': ('Aquaporin transport', 6, 6),
           'REACTOME_MAPK1_MAPK3_SIGNALING': ('MAPK1/3 signaling\n(Fig 4E top-4)', 6, 8),
           'REACTOME_DEGRADATION_OF_DVL': ('DVL degradation\n(Fig 4E)', 6, -20),
           'REACTOME_MITOCHONDRIAL_TRANSLATION_ELONGATION': ('mito translation (NS)', 6, 6)}
    for pw, (lab, dx, dy) in ann.items():
        r = mod[mod.pathway == pw]
        if len(r):
            x = r.Normal_mean_individual_amplitude.iloc[0]
            y = r.Tumor_mean_individual_amplitude.iloc[0]
            star = r.paired_p.iloc[0] < 0.05
            ax.scatter([x], [y], s=42, c=T_ORANGE if star else N_BLUE, edgecolors='black', linewidths=.8, zorder=6)
            if with_text:
                ax.annotate(lab, (x, y), fontsize=6.6, xytext=(dx, dy), textcoords='offset points', color=DK)
    if with_text:
        ax.set_xlabel('Module mean member-gene amplitude — Normal')
        ax.set_ylabel('Module mean member-gene amplitude — Tumor')
        ax.set_title('A | Pathway-level paired remodeling of inferred-phase programs', loc='left', fontsize=11, weight='bold')
        ax.legend(fontsize=7.5, loc='upper left')
        ax.text(0.98, 0.30, 'points below diagonal = amplitude-suppressed modules\n'
                            '(median T/N ratio among P<0.05 modules: 0.85)', transform=ax.transAxes,
                fontsize=7, ha='right', color='#555')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlim(-0.003, lim)
    ax.set_ylim(-0.003, lim)
    return fig


def program_patient_table():
    """per-patient mean member-gene amplitude for each program (genes with >=8 paired patients & detected)."""
    rows = []
    for pw, label in PROGRAMS:
        members = GMT.get(pw, set())
        sub = pairs[pairs.gene.isin(members)]
        counts = sub.groupby('gene').patient_id.nunique()
        good = set(counts[counts >= 8].index)
        sub = sub[sub.gene.isin(good)]
        per_pat = sub.groupby('patient_id').agg(A_N=('A_N', 'mean'), A_T=('A_T', 'mean'), n_genes=('gene', 'nunique')).reset_index()
        per_pat['program'] = label
        per_pat['pathway'] = pw
        per_pat['n_genes_used'] = len(good)
        rows.append(per_pat)
    return pd.concat(rows)


def panel_b(with_text=True, ppt=None):
    """Per-patient module-score amplitude (harmonic vector norm) from the frozen module-level coefficients."""
    mpat = pd.read_csv(G + r'\SUPP_ALL_PATHWAYS_NOMINAL_P_ONLY\02_PATIENT_LEVEL_HARMONIC_COEFFICIENTS.csv.gz')
    mtests = pd.read_csv(G + r'\SUPP_ALL_PATHWAYS_NOMINAL_P_ONLY\01_ALL_PATHWAY_PAIRED_HARMONIC_TESTS.csv')
    progs = [(pw, lab) for pw, lab in PROGRAMS] + [('REACTOME_MAPK1_MAPK3_SIGNALING', 'MAPK1/3 signaling (Fig 4E contrast)')]
    rng = np.random.default_rng(20261023)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ypositions = np.arange(len(progs))[::-1]
    stats = []
    for (pw, lab), yy in zip(progs, ypositions):
        d = mpat[mpat.pathway == pw].copy()
        d['A_N'] = np.hypot(d.Normal_bcos, d.Normal_bsin)
        d['A_T'] = np.hypot(d.Tumor_bcos, d.Tumor_bsin)
        R = (d.A_N - d.A_T)  # positive = module amplitude reduction
        Rm = R.mean()
        boots = np.array([rng.choice(R.to_numpy(), len(R)).mean() for _ in range(2000)])
        lo, hi = np.quantile(boots, [.025, .975])
        pv = mtests.loc[mtests.pathway == pw, 'paired_p'].iloc[0]
        stats.append(dict(program=lab, pathway=pw, n_patients=len(d), mean_reduction=Rm, ci_low=lo, ci_high=hi,
                          module_paired_p=pv))
        for r in R:
            ax.plot([r, 0], [yy, yy], color='#d0d0d0', lw=.7, alpha=.85, zorder=1)
        ax.errorbar([Rm], [yy], xerr=[[Rm - lo], [hi - Rm]], fmt='D', color=DK,
                    ecolor=DK, elinewidth=1.4, capsize=3.5, markersize=5.5, zorder=5)
        ax.scatter(R, [yy] * len(R), s=16, c=T_ORANGE, alpha=.8, edgecolors='white', linewidths=.4, zorder=4)
        if with_text:
            ax.annotate(f'P={pv:.4f}' if pv >= .001 else f'P={pv:.1e}', (1.0, yy), xycoords=('axes fraction', 'data'),
                        xytext=(6, 0), textcoords='offset points', fontsize=7,
                        color=DK if pv < .05 else '#888888', va='center')
    ax.axvline(0, color='gray', lw=.8, ls='--')
    if with_text:
        ax.set_yticks(ypositions)
        ax.set_yticklabels([s['program'] for s in stats], fontsize=8)
        ax.set_xlabel('Paired module-score amplitude reduction  Normal − Tumor')
        ax.set_title('B | Patient-level module amplitude change: translation program suppression', loc='left', fontsize=11, weight='bold')
        ax.text(0.02, 0.03, 'diamond = 11-patient mean, bootstrap 95% CI (2,000); dots = individual patients;\n'
                            'module score = member-gene mean log$_1$p(CP10K) with nuisance covariates (frozen analysis);\n'
                            'P = exact paired sign-flip on the 2D harmonic vector (nominal, unadjusted)',
                transform=ax.transAxes, fontsize=6.6, color='#555')
    else:
        ax.set_yticks([])
    return fig, pd.DataFrame(stats)


def panel_c(with_text=True):
    members = GMT['REACTOME_EUKARYOTIC_TRANSLATION_ELONGATION']
    counts = pairs[pairs.gene.isin(members)].groupby('gene').patient_id.nunique()
    good = counts[counts >= 8].index
    sub = pairs[pairs.gene.isin(good)]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={'width_ratios': [1, 1]})
    ax = axs[0]
    rng = np.random.default_rng(20261023)
    jit = rng.uniform(-.16, .16, len(sub))
    ax.scatter(sub.A_N, np.arange(len(sub)) + jit, s=10, c=N_BLUE, alpha=.6, edgecolors='none', rasterized=True)
    ax.scatter(sub.A_T, np.arange(len(sub)) + jit, s=10, c=T_ORANGE, alpha=.6, edgecolors='none', rasterized=True)
    for i, r in enumerate(sub.itertuples()):
        ax.plot([r.A_N, r.A_T], [i + jit[i], i + jit[i]], color='#e0e0e0', lw=.5, zorder=0)
    ax.set_yticks([])
    if with_text:
        ax.set_xlabel('First-harmonic amplitude (log$_1$p CP10K)')
        ax.set_ylabel(f'Genes of translation-elongation program (n={len(sub)}, paired)')
        ax.set_title('Gene-level paired amplitudes', fontsize=9)
    # right: acrophase retention
    ax = axs[1]
    rel = res_gene[(res_gene.n_phase >= 8)]
    tr = rel[rel.gene.isin(good)]
    grid = np.linspace(0, 12, 200)
    ecdf_all = np.array([(rel.mean_dphi_h.abs() <= v).mean() for v in grid])
    ecdf_tr = np.array([(tr.mean_dphi_h.abs() <= v).mean() for v in grid])
    ax.plot(grid, ecdf_all, color='#888888', lw=1.6)
    ax.plot(grid, ecdf_tr, color=GREEN, lw=1.8)
    if with_text:
        ax.set_xlabel('|Acrophase displacement| Tumor − Normal (h)')
        ax.set_ylabel('Cumulative proportion of genes')
        ax.set_title('Acrophase retention', fontsize=9)
        ax.legend([f'all reliable genes (n={len(rel)})', f'translation-elongation members (n={len(tr)})'],
                  fontsize=7.5, loc='lower right')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    if with_text:
        fig.suptitle('C | Within translation-elongation program: amplitude suppression with acrophase retention',
                     x=0.02, ha='left', fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def panel_d(with_text=True):
    ora_t = pd.read_csv(G + r'\FIG4D_UNIQUE_RHYTHMIC_ORA_v2\Tumor_only_ALL_TESTED_pathways.csv')
    ora_n = pd.read_csv(G + r'\FIG4D_UNIQUE_RHYTHMIC_ORA_v2\Normal_only_ALL_TESTED_pathways.csv')
    tum_sig = set(ora_t[ora_t.q_collection < 0.05].pathway)
    nor_sig = set(ora_n[ora_n.q_collection < 0.05].pathway)
    mod_sig = set(mod[mod.paired_p < 0.05].pathway)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    cats = ['Membership ORA\nTumor-only (Fig 4D)', 'Membership ORA\nNormal-only (Fig 4C)', 'Module remodeling\n(this figure)']
    vals = [len(tum_sig), len(nor_sig), len(mod_sig)]
    bars = ax.bar(cats, vals, color=[T_ORANGE, N_BLUE, GREEN], width=.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha='center', fontsize=10, weight='bold')
    ax.set_ylim(0, max(vals) * 1.22)
    if with_text:
        ax.set_ylabel('Pathways with nominal evidence')
        ax.set_title('D | Membership-based and remodeling-based pathway evidence are disjoint', loc='left', fontsize=11, weight='bold')
        ax.text(0.5, 0.86, 'overlap = 0 pathways\n(module P<0.05  ∩  ORA q<0.05)',
                transform=ax.transAxes, fontsize=9, ha='center', color=RED, weight='bold')
        ax.text(0.5, 0.62, 'membership ORA detects programs whose genes are\n'
                           'phase-associated in ONE condition;\n'
                           'module remodeling detects coordinated amplitude change\n'
                           'of genes phase-associated in BOTH conditions',
                transform=ax.transAxes, fontsize=7.5, ha='center', color='#555')
        ax.text(0.02, 0.02, 'nominal, exploratory thresholds; different tested families', transform=ax.transAxes, fontsize=6.5, color='#777')
    else:
        ax.set_xticklabels([])
    return fig


def composite():
    from PIL import Image
    for variant, folder in (('WITH_TEXT', LAB), ('NO_TEXT', NOTX)):
        imgs = [Image.open(os.path.join(folder, f'SUPPD2_{p}_{variant}.png'))
                for p in ['A', 'B', 'C', 'D']]
        w = max(im.width for im in imgs)
        imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
        row1_h = max(imgs[0].height, imgs[1].height)
        row2_h = max(imgs[2].height, imgs[3].height)
        canvas = Image.new('RGB', (w * 2 + 24, row1_h + row2_h + 24), 'white')
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (w + 24, 0))
        canvas.paste(imgs[2], (0, row1_h + 24))
        canvas.paste(imgs[3], (w + 24, row1_h + 24))
        out = BASE + rf'\COMPOSITE\SUPP_DISCOVERY_02_COMPOSITE_{variant}.png'
        canvas.save(out, dpi=(360, 360))
        canvas.save(out.replace('.png', '.tiff'), format='TIFF', dpi=(360, 360))
        print('composite saved', canvas.size)


if __name__ == '__main__':
    ppt = program_patient_table()
    ppt.to_csv(DATA + r'\01_PROGRAM_PATIENT_GENE_AMPLITUDES.csv', index=False)
    save_variant(panel_a(True), LAB, 'SUPPD2_A', 'WITH_TEXT')
    save_variant(panel_a(False), NOTX, 'SUPPD2_A', 'NO_TEXT')
    figb, st = panel_b(True)
    save_variant(figb, LAB, 'SUPPD2_B', 'WITH_TEXT')
    figb2, _ = panel_b(False)
    save_variant(figb2, NOTX, 'SUPPD2_B', 'NO_TEXT')
    st.to_csv(DATA + r'\02_MODULE_PROGRAM_PAIRED_STATS.csv', index=False)
    save_variant(panel_c(True), LAB, 'SUPPD2_C', 'WITH_TEXT')
    save_variant(panel_c(False), NOTX, 'SUPPD2_C', 'NO_TEXT')
    save_variant(panel_d(True), LAB, 'SUPPD2_D', 'WITH_TEXT')
    save_variant(panel_d(False), NOTX, 'SUPPD2_D', 'NO_TEXT')
    composite()
    with open(BASE + r'\RUN_MANIFEST\00_MANIFEST.json', 'w') as fh:
        json.dump({'figure': 'SUPP DISCOVERY 02 — coordinated suppression of phase-structured translation program; '
                              'membership vs remodeling pathway evidence are disjoint',
                   'dataset': 'GSE166555 (31,247 epithelial cells; 11 paired patients)',
                   'module_tests': 'SUPP_ALL_PATHWAYS_NOMINAL_P_ONLY (frozen, read-only)',
                   'phase_note': 'inferred phase, not sampling time; exploratory',
                   'seed': 20261023}, fh, indent=2)
    print('DONE')
