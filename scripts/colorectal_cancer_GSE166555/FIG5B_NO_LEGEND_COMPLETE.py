#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild GSE166555 Figure 5B from V7 input tables, without legend.
Only changes from V7 Figure 5B: remove legend; preserve colors, grouping,
labels, lines, axis scales, output PNG/PDF.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REFERENCE = {
    'ABCB1': ('ENCEQUIDAR', 'Exposure / resistance'),
    'CDA': ('CEDAZURIDINE', 'Exposure / resistance'),
    'MGLL': ('ELCUBRAGISTAT', 'Direct / investigational'),
    'PLA2G2A': ('VARESPLADIB METHYL', 'Direct / multi-PLA2'),
    'SLC9A3': ('TENAPANOR', 'Non-oncology physiological'),
    'MAP2K2': ('BINIMETINIB', 'Direct / oncology pathway'),
    'LGALS3': ('BELAPECTIN', 'Direct / investigational'),
    'ODC1': ('EFLORNITHINE', 'Direct / investigational'),
    'EPCAM': ('ADECATUMUMAB', 'Surface-targeted delivery'),
}
COLORS = {
    'Direct / investigational': '#bf704b',
    'Direct / multi-PLA2': '#bf704b',
    'Direct / oncology pathway': '#bf704b',
    'Exposure / resistance': '#447ca8',
    'Surface-targeted delivery': '#3e948c',
    'Non-oncology physiological': '#9993b5',
}


def draw(root, output, max_total=12):
    v2 = root / 'FIG45_MECHANISM_PHASE_v2'
    atlas_file = v2 / '04_FIG4_DRUGGABLE_RHYTHMIC_TARGET_ATLAS.csv.gz'
    drug_file = v2 / '06_FIG5_ALL_MECHANISM_QUALIFIED_MATCHES.csv.gz'
    for f in (atlas_file, drug_file):
        if not f.exists():
            raise FileNotFoundError(f'Required V7 source table missing: {f}')
    atlas = pd.read_csv(atlas_file, low_memory=False)
    drug = pd.read_csv(drug_file, low_memory=False)
    required_atlas = {'gene', 'paired_delta_amp'}
    required_drug = {'gene', 'drug_name'}
    if not required_atlas.issubset(atlas.columns):
        raise ValueError(f'Atlas requires {required_atlas}, got {list(atlas.columns)}')
    if not required_drug.issubset(drug.columns):
        raise ValueError(f'Drug matches require {required_drug}, got {list(drug.columns)}')

    atlas['gene'] = atlas['gene'].astype(str).str.upper()
    drug['gene'] = drug['gene'].astype(str).str.upper()
    drug['drug_name'] = drug['drug_name'].astype(str).str.upper()
    drug['pref_rank'] = drug.apply(
        lambda r: 0 if r.gene in REFERENCE and r.drug_name == REFERENCE[r.gene][0] else 1,
        axis=1,
    )
    representative = (
        drug.sort_values(['gene', 'pref_rank', 'drug_name'])
            .drop_duplicates('gene')[['gene', 'drug_name']]
    )
    merged = atlas.merge(representative, on='gene', how='left', suffixes=('', '_db'))
    merged['drug_name'] = merged['drug_name'].fillna('UNMAPPED')
    merged = merged[merged['gene'].isin(REFERENCE)].copy()
    available = drug.groupby('gene')['drug_name'].agg(set).to_dict()
    merged['drug_display'] = merged.apply(
        lambda r: REFERENCE[r.gene][0]
        if REFERENCE[r.gene][0] in available.get(r.gene, set())
        else str(r.drug_name), axis=1,
    )
    merged['category'] = merged['gene'].map({k: v[1] for k, v in REFERENCE.items()})
    merged['evidence'] = pd.to_numeric(merged['paired_delta_amp'], errors='coerce')
    merged = merged.dropna(subset=['evidence']).sort_values('evidence', ascending=False).head(max_total)
    if merged.empty:
        raise ValueError('Figure 5B: no matched target evidence')
    merged = merged.sort_values(['category', 'evidence'], ascending=[True, False])

    n = len(merged)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * n + 1.5)))
    y = np.arange(n)
    for j, (_, r) in enumerate(merged.iterrows()):
        color = COLORS.get(r['category'], '#999999')
        ax.hlines(j, 0, r['evidence'], color=color, lw=2)
        ax.scatter(r['evidence'], j, s=90, color=color, zorder=3)

    ax.set_yticks(y, [f'{r.gene}   ·   {r.drug_display}' for r in merged.itertuples()], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Paired first-harmonic contrast (RNA units; not efficacy)')
    ax.set_title('Figure 5B | Pharmacological mechanism groups', fontsize=14)
    for category in merged['category'].drop_duplicates():
        locations = np.where(merged['category'].to_numpy() == category)[0]
        ax.axhspan(locations.min() - 0.48, locations.max() + 0.48,
                   color=COLORS.get(category, '#999999'), alpha=0.05)

    # Intentional: NO legend (V7 previously put it at lower right).
    ax.grid(axis='x', alpha=0.12)
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    stem = 'FIG5B_MECHANISM_LOLLIPOP_NO_LEGEND'
    fig.savefig(output / f'{stem}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output / f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(output / f'{stem}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    merged[['gene', 'drug_display', 'category', 'evidence']].to_csv(
        output / f'{stem}_DATA.csv', index=False)
    print(f'SUCCESS: {output / stem}.[png|pdf|svg]')


def main():
    parser = argparse.ArgumentParser(description='Figure 5B, no lower-right legend')
    parser.add_argument('--root', default='16_FIG4_FIG5_APPLICATIONS/GSE166555')
    parser.add_argument('--output', default=None)
    parser.add_argument('--max-total', type=int, default=12)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / 'FIG5B_NO_LEGEND'
    draw(root, output, args.max_total)


if __name__ == '__main__':
    main()
