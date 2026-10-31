"""SUPP DISCOVERY 01 — Figure: gene-level remodeling of inferred-phase programs in paired Normal/Tumor (GSE166555).

Panels:
  A  genome-wide paired amplitude-remodeling landscape (class-stratified)
  B  phase displacement vs amplitude change (reliable-phase genes; CORE clock control)
  C  class-level median amplitude change with bootstrap 95% CI
  D  functional annotation (ORA) of the amplitude-loss extreme set

Every panel rendered twice: WITH text and NO text (same data, same layout).
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
BASE = ROOT + r'\ANCHORPHASE_MANUSCRIPT_COMPLETION_20261028\02_NEW_BIOLOGICAL_SUPPLEMENTARY_FIGURES\SUPP_DISCOVERY_01'
DATA = BASE + r'\DATA'
LAB = BASE + r'\PANELS_LABELED'
NOTX = BASE + r'\PANELS_NO_TEXT'
for d in (LAB, NOTX, BASE + r'\COMPOSITE'):
    os.makedirs(d, exist_ok=True)

N_BLUE = '#2877ac'
T_ORANGE = '#d8753a'
GREEN = '#3b9890'
PURPLE = '#7a5aa0'
GRAY = '#b8bcc0'
DK = '#222222'
RED = '#985052'

res = pd.read_csv(DATA + r'\02_GENE_LEVEL_REMODELING_STATS.csv')
ora = pd.read_csv(DATA + r'\05_REMODELING_EXTREME_ORA.csv')
cls_ci = pd.read_csv(DATA + r'\06_CLASS_LEVEL_AMPLITUDE_CHANGE.csv')

CLASS_COLOR = {'Shared': GREEN, 'Normal-only': N_BLUE, 'Tumor-only': T_ORANGE, 'Neither': '#cccccc'}
CORE17 = ['ARNTL', 'BHLHE41', 'CLOCK', 'CRY1', 'CRY2', 'DBP', 'FBXO21', 'HLF', 'NFIL3', 'NPAS2',
          'NR1D1', 'NR1D2', 'PER1', 'PER2', 'PER3', 'RORC', 'TEF']


def save_variant(fig, outdir, stem, suffix):
    fig.savefig(os.path.join(outdir, f'{stem}_{suffix}.png'), dpi=360, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(outdir, f'{stem}_{suffix}.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(outdir, f'{stem}_{suffix}.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def panel_a(with_text=True):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sub = res.copy()
    extremes = sub[(sub.mean_dA <= -0.05) | (sub.mean_dA >= 0.05)]
    plain = sub[~sub.index.isin(extremes.index)]
    for cls in ['Neither', 'Normal-only', 'Tumor-only', 'Shared']:
        d = plain[plain.fig4_class == cls]
        ax.scatter(d.mean_A_N, d.mean_dA, s=7 if cls != 'Neither' else 4, alpha=.45 if cls != 'Neither' else .35,
                   c=CLASS_COLOR[cls], edgecolors='none', label=f'{cls} (n={len(d):,})' if with_text else None, rasterized=True)
    ax.scatter(extremes.mean_A_N, extremes.mean_dA, s=14, c=[CLASS_COLOR[c] for c in extremes.fig4_class],
               edgecolors='black', linewidths=.25, zorder=5, rasterized=True)
    for g, x, y, dx, dy in [('RPL35', None, None, 3, 5), ('RPL6', None, None, 3, 5), ('RPL37', None, None, 3, 5),
                            ('SLC26A3', None, None, 3, -8), ('CEACAM7', None, None, 3, 5), ('SLC9A3', None, None, 3, -9),
                            ('GUCA2A', None, None, 3, 5), ('SELENOP', None, None, 3, -9)]:
        r = res[res.gene == g]
        if len(r):
            ax.annotate(g, (r.mean_A_N.iloc[0], r.mean_dA.iloc[0]), fontsize=6.5 if with_text else 0.01,
                        xytext=(dx, dy), textcoords='offset points', color=DK)
    ax.axhline(0, color='gray', lw=.6)
    ax.axhline(0.05, color=RED, lw=.7, ls='--')
    ax.axhline(-0.05, color=RED, lw=.7, ls='--')
    ax.set_xscale('log')
    if with_text:
        ax.set_xlabel('Normal first-harmonic amplitude  log$_1$0  (log$_1$p CP10K units)')
        ax.set_ylabel('Paired mean amplitude change  Tumor − Normal')
        ax.set_title('A | Genome-wide paired remodeling of phase-associated genes', loc='left', fontsize=11, weight='bold')
        ax.text(0.02, 0.97, '6,401 eligible genes (≥8 paired patients, detected ≥5% both conditions)',
                transform=ax.transAxes, fontsize=7.5, va='top', color='#555')
        ax.legend(fontsize=7, loc='lower right', framealpha=.9, markerscale=1.6)
        ax.text(0.98, 0.05, 'dashed: |ΔA| = 0.05', transform=ax.transAxes, fontsize=6.5, ha='right', color=RED)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlim(0.008, 1.2)
    return fig


def panel_b(with_text=True):
    rel = res[(res.n_phase >= 8)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for cls in ['Neither', 'Normal-only', 'Tumor-only', 'Shared']:
        d = rel[rel.fig4_class == cls]
        ax.scatter(d.mean_dphi_h, d.mean_dA, s=8 + 30 * d.phase_conc.fillna(0), alpha=.6,
                   c=CLASS_COLOR[cls], edgecolors='none',
                   label=f'{cls} (n={len(d)})' if with_text else None, rasterized=True)
    clock = rel[rel.gene.isin(CORE17)]
    ax.scatter(clock.mean_dphi_h, clock.mean_dA, s=46, c='white', edgecolors=DK, linewidths=1.1, zorder=6)
    for r in clock.itertuples():
        if with_text:
            ax.annotate(r.gene, (r.mean_dphi_h, r.mean_dA), fontsize=7, weight='bold',
                        xytext=(6, 4), textcoords='offset points', color=DK)
    for g, dx, dy in [('S100A11', 4, 4), ('CLDN3', 4, -10), ('JUN', 4, 4), ('IFI27', 4, 4), ('HSPA1A', -30, 6)]:
        r = rel[rel.gene == g]
        if len(r) and with_text:
            ax.annotate(g, (r.mean_dphi_h.iloc[0], r.mean_dA.iloc[0]), fontsize=6.5,
                        xytext=(dx, dy), textcoords='offset points', color='#555')
    ax.axvline(-3, color=GRAY, lw=.7, ls=':')
    ax.axvline(3, color=GRAY, lw=.7, ls=':')
    ax.axhline(0, color='gray', lw=.6)
    ax.axvline(0, color='gray', lw=.6)
    if with_text:
        ax.set_xlabel('Circular acrophase displacement  Tumor − Normal (h)')
        ax.set_ylabel('Paired mean amplitude change  Tumor − Normal')
        ax.set_title('B | Phase displacement vs amplitude change (reliable-phase genes)', loc='left', fontsize=11, weight='bold')
        ax.text(0.02, 0.97, 'n=264 genes with amplitude ≥0.05 in both conditions (≥8 patients);\n'
                             'marker size = patient-to-patient phase-displacement concentration;\n'
                             'outlined = CORE17 anchors (displacement ≈ 0–0.5 h)',
                transform=ax.transAxes, fontsize=7, va='top', color='#555')
        ax.legend(fontsize=7, loc='lower left', framealpha=.9)
        ax.text(0.98, 0.02, 'median |Δφ| = 1.51 h; only 7/264 genes |Δφ| > 4 h',
                transform=ax.transAxes, fontsize=7, ha='right', color='#555')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return fig


def panel_c(with_text=True):
    order = ['Normal-only', 'Shared', 'Neither', 'Tumor-only']
    cc = cls_ci.set_index('fig4_class').loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(cc))[::-1]
    colors = [CLASS_COLOR[c] for c in cc.fig4_class]
    ax.errorbar(cc.median_dA, y, xerr=[cc.median_dA - cc.ci_low, cc.ci_high - cc.median_dA],
                fmt='o', color=DK, ecolor=DK, elinewidth=1.2, capsize=3, markersize=0)
    ax.scatter(cc.median_dA, y, s=70, c=colors, zorder=5, edgecolors='white', linewidths=1)
    ax.axvline(0, color='gray', lw=.7, ls='--')
    if with_text:
        ax.set_yticks(y)
        ax.set_yticklabels([f'{r.fig4_class}\n(n={r.n:,})' for r in cc.itertuples()], fontsize=8.5)
        ax.set_xlabel('Median paired amplitude change  Tumor − Normal (95% bootstrap CI, 2,000 resamples)')
        ax.set_title('C | Amplitude remodeling by Fig-4 phase-association class', loc='left', fontsize=11, weight='bold')
        ax.text(0.02, 0.05, 'A$_N$ ≥ 0.05 subset', transform=ax.transAxes, fontsize=7, color='#555')
    else:
        ax.set_yticks([])
    return fig


def short(p):
    s = p.replace('REACTOME_', '').replace('HALLMARK_', '')
    return s.replace('_', ' ').title()[:52]


def panel_d(with_text=True):
    d = ora[(ora.query_set == 'amplitude_loss')].sort_values('p').head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    y = np.arange(len(d))
    sc = ax.scatter(d.fold_enrichment, y, c=-np.log10(d.q), s=40 + 9 * (d.n_overlap - 3),
                    cmap='viridis', edgecolors='black', linewidths=.4, zorder=5)
    for r, yy in zip(d.itertuples(), y):
        if with_text:
            ax.annotate(f'{r.n_overlap}/{r.n_members_bg}', (r.fold_enrichment, yy), xytext=(8, -3),
                        textcoords='offset points', fontsize=6.5, color='#555')
    ax.set_yticks(y)
    if with_text:
        ax.set_yticklabels([short(p) for p in d.pathway], fontsize=8)
        ax.set_xlabel('Fold enrichment (hypergeometric ORA)')
        ax.set_title('D | Programs enriched among strongly amplitude-losing genes', loc='left', fontsize=11, weight='bold')
        cb = fig.colorbar(sc, ax=ax, shrink=.75)
        cb.set_label('−log$_1$$_0$ BH q', fontsize=8)
        ax.text(0.98, 0.04, 'query: 117 genes with ΔA ≤ −0.05 (A$_N$≥0.05); background: 1,876 eligible genes;\n'
                            'amplitude-gain set (47 genes): no pathway BH q < 0.05 (all q ≥ 0.12)',
                transform=ax.transAxes, fontsize=6.8, ha='right', color='#555')
        ax.text(0.02, 0.96, 'label = overlap / pathway size', transform=ax.transAxes, fontsize=6.5, va='top', color='#555')
    else:
        ax.set_yticklabels([])
    return fig


def composite():
    from PIL import Image
    for variant, folder in (('WITH_TEXT', LAB), ('NO_TEXT', NOTX)):
        imgs = [Image.open(os.path.join(folder, f'SUPPD1_{p}_{variant}.png'))
                for p in ['A', 'B', 'C', 'D']]
        w = max(im.width for im in imgs)
        scale = [w / im.width for im in imgs]
        imgs = [im.resize((w, int(im.height * s))) for im, s in zip(imgs, scale)]
        row1_h = max(imgs[0].height, imgs[1].height)
        row2_h = max(imgs[2].height, imgs[3].height)
        canvas = Image.new('RGB', (w * 2 + 24, row1_h + row2_h + 24), 'white')
        canvas.paste(imgs[0], (0, 0))
        canvas.paste(imgs[1], (w + 24, 0))
        canvas.paste(imgs[2], (0, row1_h + 24))
        canvas.paste(imgs[3], (w + 24, row1_h + 24))
        out = BASE + rf'\COMPOSITE\SUPP_DISCOVERY_01_COMPOSITE_{variant}.png'
        canvas.save(out, dpi=(360, 360))
        canvas.save(out.replace('.png', '.tiff'), format='TIFF', dpi=(360, 360))
        print('composite saved', out, canvas.size)


if __name__ == '__main__':
    for fn, stem in ((panel_a, 'A'), (panel_b, 'B'), (panel_c, 'C'), (panel_d, 'D')):
        save_variant(fn(True), LAB, f'SUPPD1_{stem}', 'WITH_TEXT')
        save_variant(fn(False), NOTX, f'SUPPD1_{stem}', 'NO_TEXT')
        print('panel', stem, 'done')
    composite()
    os.makedirs(BASE + r'\RUN_MANIFEST', exist_ok=True)
    manifest = {
        'figure': 'SUPP DISCOVERY 01 — gene-level paired Normal->Tumor remodeling of inferred-phase programs',
        'dataset': 'GSE166555 (31,247 epithelial cells; 11 paired patients)',
        'phase_note': 'AnchorPhase inferred phase (frozen B0 predictions); NOT sampling time',
        'panels': ['A landscape', 'B phase displacement vs amplitude', 'C class medians', 'D ORA of loss set'],
        'stats': 'see DATA/02, 05, 06 CSVs + 04_ANALYSIS_SUMMARY.json',
        'seed': 20261023,
    }
    with open(BASE + r'\RUN_MANIFEST\00_MANIFEST.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print('DONE')
