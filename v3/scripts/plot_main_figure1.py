#!/usr/bin/env python3
"""
Figure 1: Four-Panel Representational-Causal Profile
=====================================================
Demonstrating the dissociation between representation accessibility
(linear decodability) and causal leverage across layers and generation stages.

Panels:
(a) Layerwise Linear Decodability (D_l) across all 28 layers (MLP, ATTN, RESID)
(b) Prompt-Time Local Causal Recovery (S_l) under matched substitution (~0% everywhere)
(c) Generation-Time Local Causal Recovery (G_{l,t}) across focused representative layers (surges to 53.5% in late residual)
(d) Peak-Site Contrast & Conceptual Dissociation Summary (L15 MLP vs L24 RESID)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set professional publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
})

# Color palette: distinct, accessible, publication-grade
COLORS = {
    'mlp': '#2b5c8f',      # Deep Blue
    'attn': '#d95f02',     # Vibrant Vermilion
    'resid': '#1b9e77',    # Emerald Green
    'contrast': '#7570b3', # Slate Purple
    'accent': '#e7298a',   # Magenta Accent
}

def load_data():
    base_dir = Path("/mnt/nas/home/hiromi/src/emo/v3/results")
    
    # Load 28-layer sweep data
    sweep_df = pd.read_csv(base_dir / "causal_localization_sweep_joint_ot.csv")
    
    # Load 39-pair focused sweep data
    focused_df = pd.read_csv(base_dir / "focused_causal_sweep_39pairs.csv")
    
    return sweep_df, focused_df

def plot_figure1():
    sweep_df, focused_df = load_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.25)
    
    # -------------------------------------------------------------
    # Panel (a): Layerwise Linear Decodability (D_l)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = sweep_df[sweep_df['component'] == comp].sort_values('layer')
        ax_a.plot(sub['layer'], sub['probe_r2'], marker='o', label=label, color=COLORS[comp])
        
    # Annotate peaks
    mlp_peak_l = sweep_df[sweep_df['component'] == 'mlp'].loc[sweep_df[sweep_df['component'] == 'mlp']['probe_r2'].idxmax()]
    attn_peak_l = sweep_df[sweep_df['component'] == 'attn'].loc[sweep_df[sweep_df['component'] == 'attn']['probe_r2'].idxmax()]
    resid_peak_l = sweep_df[sweep_df['component'] == 'resid'].loc[sweep_df[sweep_df['component'] == 'resid']['probe_r2'].idxmax()]
    
    ax_a.annotate(f"MLP Peak: L15\n($R^2 = 0.561$)", 
                  xy=(15, mlp_peak_l['probe_r2']), xytext=(12, 0.61),
                  arrowprops=dict(arrowstyle="->", color=COLORS['mlp'], lw=1.2),
                  fontsize=8.5, fontweight='bold', color=COLORS['mlp'],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['mlp'], alpha=0.9))
    
    ax_a.annotate(f"Attention Peak: L18 ($R^2 = 0.550$)", 
                  xy=(18, attn_peak_l['probe_r2']), xytext=(10, 0.22),
                  arrowprops=dict(arrowstyle="->", color=COLORS['attn'], lw=1.2),
                  fontsize=8.5, color=COLORS['attn'])

    ax_a.set_title("(a) Prompt-Time Linear Decodability ($D_\\ell$)", fontweight='bold', loc='left')
    ax_a.set_xlabel("Layer Index (0–27)")
    ax_a.set_ylabel("Linear Probe $R^2$ (Pseudo-$R^2$)")
    ax_a.set_xlim(-0.5, 27.5)
    ax_a.set_ylim(0.15, 0.68)
    ax_a.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # -------------------------------------------------------------
    # Panel (b): Prompt-Time Local Causal Recovery (S_l)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    for comp, label in [('mlp', 'MLP output'), ('attn', 'Attention output'), ('resid', 'Residual stream')]:
        sub = sweep_df[sweep_df['component'] == comp].sort_values('layer')
        ax_b.plot(sub['layer'], sub['mean_ot_recovery'], marker='s', label=label, color=COLORS[comp], alpha=0.6, linestyle=':')

    # Overlay Focused 39-pair points
    for comp, marker in [('mlp', 'o'), ('attn', '^'), ('resid', 'D')]:
        sub_f = focused_df[focused_df['component'] == comp].sort_values('layer')
        ax_b.plot(sub_f['layer'], sub_f['prompt_mean_ot_recovery'], marker=marker, 
                  color=COLORS[comp], lw=2.2, label=f"{comp.upper()} (N=39 focused)")

    ax_b.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
    ax_b.annotate("L15 MLP (Probe Peak):\n$S_{\\mathrm{L15,MLP}} = 0.06\\%$", 
                  xy=(15, 0.055), xytext=(13, 2.5),
                  arrowprops=dict(arrowstyle="->", color=COLORS['mlp'], lw=1.2),
                  fontsize=8.5, fontweight='bold', color=COLORS['mlp'],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['mlp'], alpha=0.9))

    ax_b.set_title("(b) Prompt-Time Local Causal Recovery ($S_\\ell$)", fontweight='bold', loc='left')
    ax_b.set_xlabel("Layer Index (0–27)")
    ax_b.set_ylabel("2D Joint OT Recovery (%)")
    ax_b.set_xlim(-0.5, 27.5)
    ax_b.set_ylim(-4.5, 5.0)
    ax_b.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=8)

    # -------------------------------------------------------------
    # Panel (c): Generation-Time Local Causal Recovery (G_{l,t})
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    
    # 95% CIs precomputed from N=39 pairs
    ci_dict = {
        ('resid', 10): (-2.5, -0.1), ('resid', 14): (-1.7, 1.9), ('resid', 15): (4.5, 10.2),
        ('resid', 18): (35.2, 47.7), ('resid', 20): (42.6, 57.8), ('resid', 24): (46.0, 60.9),
        ('mlp', 10): (-1.8, 1.5), ('mlp', 14): (-2.7, -0.3), ('mlp', 15): (-0.5, 3.3),
        ('mlp', 18): (5.2, 9.6), ('mlp', 20): (6.1, 13.0), ('mlp', 24): (11.0, 19.4),
        ('attn', 10): (-0.3, 1.7), ('attn', 14): (-2.1, 0.0), ('attn', 15): (-0.1, 6.0),
        ('attn', 18): (-9.8, -5.7), ('attn', 20): (-0.8, 2.8), ('attn', 24): (-0.6, 0.5),
    }

    for comp, label in [('resid', 'Residual stream'), ('mlp', 'MLP output'), ('attn', 'Attention output')]:
        sub_f = focused_df[focused_df['component'] == comp].sort_values('layer')
        layers = sub_f['layer'].values
        means = sub_f['gen_mean_ot_recovery'].values
        
        # Calculate error bars
        yerr_lower = [means[i] - ci_dict.get((comp, l), (means[i]-2, means[i]+2))[0] for i, l in enumerate(layers)]
        yerr_upper = [ci_dict.get((comp, l), (means[i]-2, means[i]+2))[1] - means[i] for i, l in enumerate(layers)]
        
        ax_c.errorbar(layers, means, yerr=[yerr_lower, yerr_upper], 
                     marker='o' if comp=='resid' else ('s' if comp=='mlp' else '^'),
                     label=label, color=COLORS[comp], lw=2.2, capsize=4, capthick=1.2)

    ax_c.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)
    ax_c.annotate("Late Residual Surge:\nL24: 53.48% (Med: 59.4%)\nL20: 50.22% (Med: 57.9%)\nL18: 41.48% (Med: 48.5%)", 
                  xy=(24, 53.48), xytext=(11, 40.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS['resid'], lw=1.4),
                  fontsize=8.5, fontweight='bold', color=COLORS['resid'],
                  bbox=dict(boxstyle="round,pad=0.3", fc="#eefaf5", ec=COLORS['resid'], alpha=0.95))

    ax_c.annotate("L15 MLP (Probe Peak):\n$G = 1.39\\%$", 
                  xy=(15, 1.39), xytext=(11, -8.0),
                  arrowprops=dict(arrowstyle="->", color=COLORS['mlp'], lw=1.2),
                  fontsize=8.5, color=COLORS['mlp'],
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS['mlp'], alpha=0.9))

    ax_c.set_title("(c) Generation-Time Local Causal Recovery ($G_{\\ell,t}$)", fontweight='bold', loc='left')
    ax_c.set_xlabel("Focused Representative Layer (N=39 pairs)")
    ax_c.set_ylabel("Generation 2D Joint OT Recovery (%)")
    ax_c.set_xticks([10, 14, 15, 18, 20, 24])
    ax_c.set_ylim(-15, 65)
    ax_c.legend(loc='upper left', frameon=True, framealpha=0.9)

    # -------------------------------------------------------------
    # Panel (d): Peak-Site Contrast & Conceptual Dissociation Summary
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    
    # Bar chart comparing L15 MLP vs L24 Residual
    categories = ['Linear Decodability\n($D_\\ell: R^2 \\times 100$)', 
                  'Prompt-Time Recovery\n($S_\\ell: \\%$)', 
                  'Generation-Time Recovery\n($G_{\\ell,t}: \\%$)']
    
    l15_vals = [0.5610 * 100, 0.0556, 1.3948]
    l24_vals = [0.1470 * 100, 0.2005, 53.4835]
    
    x = np.arange(len(categories))
    width = 0.36
    
    rects1 = ax_d.bar(x - width/2, l15_vals, width, label='Layer 15 MLP (Probe Peak)', 
                      color=COLORS['mlp'], alpha=0.9, edgecolor='black', lw=0.8)
    rects2 = ax_d.bar(x + width/2, l24_vals, width, label='Layer 24 Residual (Late Leverage)', 
                      color=COLORS['resid'], alpha=0.9, edgecolor='black', lw=0.8)
    
    # Value labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax_d.annotate(f'{height:.1f}%' if height > 1 else f'{height:.2f}%',
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=COLORS['mlp'])
    for rect in rects2:
        height = rect.get_height()
        ax_d.annotate(f'{height:.1f}%' if height > 1 else f'{height:.2f}%',
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3), textcoords="offset points",
                      ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=COLORS['resid'])
        
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(categories, fontsize=9.5)
    ax_d.set_ylabel("Metric Magnitude (%)")
    ax_d.set_ylim(-2, 72)
    ax_d.set_title("(d) Peak-Site Contrast (L15 MLP vs. L24 Residual)", fontweight='bold', loc='left')
    ax_d.legend(loc='upper right', frameon=True, framealpha=0.9)
    
    # Central Thesis Banner inside Panel (d)
    banner_text = (
        "Peak-Site Contrast:\n"
        "$\\Delta G = G_{\\mathrm{L24,RESID}} - G_{\\mathrm{L15,MLP}} = +52.09\\%$\n"
        "$95\\%\\text{ bootstrap CI: } [+44.4\\%, +59.7\\%]\n\n"
        "\"Where information is readable \\neq\n"
        " where it becomes causally effective\""
    )
    ax_d.text(0.32, 0.48, banner_text, transform=ax_d.transAxes,
              fontsize=9.5, fontweight='bold', ha='center', va='center',
              bbox=dict(boxstyle="round,pad=0.5", fc='#fff2f2', ec='#d95f02', lw=1.5, alpha=0.95))

    # Save high-res figures
    out_dir = Path("/mnt/nas/home/hiromi/src/emo/v3/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "figure1_four_panel_dissociation.png"
    pdf_path = out_dir / "figure1_four_panel_dissociation.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated Figure 1:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

if __name__ == "__main__":
    plot_figure1()
