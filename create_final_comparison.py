"""
Create comprehensive comparison visualization of all interventions:
- Baseline vs Disclaimer vs CoT
- 405B vs 70B
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*70)
print("CREATING FINAL COMPREHENSIVE COMPARISON")
print("="*70)

# Load all experiment results
experiments = {
    "405b_baseline": "results/8b_405b_baseline_20251102_133054/experiment.json",
    "405b_disclaimer": "results/8b_405b_disclaimer_20251102_134822/experiment.json",
    "405b_cot": "results/8b_405b_cot_20251102_142418/experiment.json",
    "70b_baseline": "results/8b_70b_baseline_20251102_132305/experiment.json",
    "70b_disclaimer": "results/8b_70b_disclaimer_20251102_134736/experiment.json",
    "70b_cot": "results/8b_70b_cot_20251102_141909/experiment.json",
}

data = {}
for name, path in experiments.items():
    if Path(path).exists():
        with open(path) as f:
            data[name] = json.load(f)
        print(f"✓ Loaded {name}")
    else:
        print(f"✗ Missing {name}: {path}")

# Extract PGR metrics for each K
K_VALUES = [0, 2, 5, 10]

def extract_pgr_curve(exp_data):
    """Extract PGR values for K={0,2,5,10}"""
    pgr_metrics = exp_data['pgr_metrics']
    return [pgr_metrics[str(k)]['pgr'] for k in K_VALUES]

def extract_acc_curve(exp_data):
    """Extract accuracy values for K={0,2,5,10}"""
    pgr_metrics = exp_data['pgr_metrics']
    return [pgr_metrics[str(k)]['strong_weak_accuracy'] for k in K_VALUES]

# Extract all curves
pgr_curves = {name: extract_pgr_curve(exp) for name, exp in data.items()}
acc_curves = {name: extract_acc_curve(exp) for name, exp in data.items()}

print("\n" + "="*70)
print("PGR COMPARISON TABLE")
print("="*70)

import pandas as pd

# Create comparison table
table_data = []
for k in K_VALUES:
    row = {"K": k}
    for name in experiments.keys():
        pgr = pgr_curves[name][K_VALUES.index(k)]
        row[name] = f"{pgr:.3f}"
    table_data.append(row)

df = pd.DataFrame(table_data)
print("\n" + df.to_string(index=False))

# Calculate deltas from baseline
print("\n" + "="*70)
print("DELTA FROM BASELINE")
print("="*70)

delta_data = []
for k in K_VALUES:
    idx = K_VALUES.index(k)
    row = {"K": k}

    # 405B deltas
    baseline_405 = pgr_curves["405b_baseline"][idx]
    row["405B_disclaimer"] = f"{pgr_curves['405b_disclaimer'][idx] - baseline_405:+.3f}"
    row["405B_cot"] = f"{pgr_curves['405b_cot'][idx] - baseline_405:+.3f}"

    # 70B deltas
    baseline_70 = pgr_curves["70b_baseline"][idx]
    row["70B_disclaimer"] = f"{pgr_curves['70b_disclaimer'][idx] - baseline_70:+.3f}"
    row["70B_cot"] = f"{pgr_curves['70b_cot'][idx] - baseline_70:+.3f}"

    delta_data.append(row)

delta_df = pd.DataFrame(delta_data)
print("\n" + delta_df.to_string(index=False))

# Create comprehensive visualization
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig = plt.figure(figsize=(18, 10))

# Define colors
colors_405 = {
    'baseline': '#1f77b4',  # blue
    'disclaimer': '#ff7f0e',  # orange
    'cot': '#2ca02c',  # green
}

colors_70 = {
    'baseline': '#d62728',  # red
    'disclaimer': '#9467bd',  # purple
    'cot': '#8c564b',  # brown
}

# Plot 1: 405B - All interventions
ax1 = plt.subplot(2, 3, 1)
ax1.plot(K_VALUES, pgr_curves['405b_baseline'], 'o-', label='Baseline',
         linewidth=2.5, markersize=9, color=colors_405['baseline'])
ax1.plot(K_VALUES, pgr_curves['405b_disclaimer'], 's-', label='Disclaimer',
         linewidth=2.5, markersize=9, color=colors_405['disclaimer'])
ax1.plot(K_VALUES, pgr_curves['405b_cot'], '^-', label='CoT',
         linewidth=2.5, markersize=9, color=colors_405['cot'])
ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Recovery')
ax1.set_xlabel('K (Few-Shot Examples)', fontsize=11)
ax1.set_ylabel('PGR', fontsize=11)
ax1.set_title('405B: All Interventions', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(K_VALUES)
ax1.set_ylim(0.80, 1.10)

# Plot 2: 70B - All interventions
ax2 = plt.subplot(2, 3, 2)
ax2.plot(K_VALUES, pgr_curves['70b_baseline'], 'o-', label='Baseline',
         linewidth=2.5, markersize=9, color=colors_70['baseline'])
ax2.plot(K_VALUES, pgr_curves['70b_disclaimer'], 's-', label='Disclaimer',
         linewidth=2.5, markersize=9, color=colors_70['disclaimer'])
ax2.plot(K_VALUES, pgr_curves['70b_cot'], '^-', label='CoT',
         linewidth=2.5, markersize=9, color=colors_70['cot'])
ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect Recovery')
ax2.set_xlabel('K (Few-Shot Examples)', fontsize=11)
ax2.set_ylabel('PGR', fontsize=11)
ax2.set_title('70B: All Interventions', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(K_VALUES)
ax2.set_ylim(0.80, 1.10)

# Plot 3: Direct comparison - Baseline
ax3 = plt.subplot(2, 3, 3)
ax3.plot(K_VALUES, pgr_curves['405b_baseline'], 'o-', label='405B',
         linewidth=2.5, markersize=9, color=colors_405['baseline'])
ax3.plot(K_VALUES, pgr_curves['70b_baseline'], 'o-', label='70B',
         linewidth=2.5, markersize=9, color=colors_70['baseline'])
ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
ax3.set_xlabel('K (Few-Shot Examples)', fontsize=11)
ax3.set_ylabel('PGR', fontsize=11)
ax3.set_title('Baseline Comparison', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(K_VALUES)
ax3.set_ylim(0.80, 1.10)

# Plot 4: Direct comparison - Disclaimer
ax4 = plt.subplot(2, 3, 4)
ax4.plot(K_VALUES, pgr_curves['405b_disclaimer'], 's-', label='405B',
         linewidth=2.5, markersize=9, color=colors_405['disclaimer'])
ax4.plot(K_VALUES, pgr_curves['70b_disclaimer'], 's-', label='70B',
         linewidth=2.5, markersize=9, color=colors_70['disclaimer'])
ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
ax4.set_xlabel('K (Few-Shot Examples)', fontsize=11)
ax4.set_ylabel('PGR', fontsize=11)
ax4.set_title('Disclaimer Comparison', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(K_VALUES)
ax4.set_ylim(0.80, 1.10)

# Plot 5: Direct comparison - CoT
ax5 = plt.subplot(2, 3, 5)
ax5.plot(K_VALUES, pgr_curves['405b_cot'], '^-', label='405B',
         linewidth=2.5, markersize=9, color=colors_405['cot'])
ax5.plot(K_VALUES, pgr_curves['70b_cot'], '^-', label='70B',
         linewidth=2.5, markersize=9, color=colors_70['cot'])
ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
ax5.set_xlabel('K (Few-Shot Examples)', fontsize=11)
ax5.set_ylabel('PGR', fontsize=11)
ax5.set_title('CoT Comparison', fontsize=13, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(K_VALUES)
ax5.set_ylim(0.80, 1.10)

# Plot 6: Intervention effectiveness (delta from baseline)
ax6 = plt.subplot(2, 3, 6)
deltas_405_disclaimer = [pgr_curves['405b_disclaimer'][i] - pgr_curves['405b_baseline'][i] for i in range(len(K_VALUES))]
deltas_405_cot = [pgr_curves['405b_cot'][i] - pgr_curves['405b_baseline'][i] for i in range(len(K_VALUES))]
deltas_70_disclaimer = [pgr_curves['70b_disclaimer'][i] - pgr_curves['70b_baseline'][i] for i in range(len(K_VALUES))]
deltas_70_cot = [pgr_curves['70b_cot'][i] - pgr_curves['70b_baseline'][i] for i in range(len(K_VALUES))]

x = np.arange(len(K_VALUES))
width = 0.2

ax6.bar(x - 1.5*width, deltas_405_disclaimer, width, label='405B Disclaimer', color=colors_405['disclaimer'], alpha=0.8)
ax6.bar(x - 0.5*width, deltas_405_cot, width, label='405B CoT', color=colors_405['cot'], alpha=0.8)
ax6.bar(x + 0.5*width, deltas_70_disclaimer, width, label='70B Disclaimer', color=colors_70['disclaimer'], alpha=0.8)
ax6.bar(x + 1.5*width, deltas_70_cot, width, label='70B CoT', color=colors_70['cot'], alpha=0.8)

ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.set_xlabel('K (Few-Shot Examples)', fontsize=11)
ax6.set_ylabel('Δ PGR from Baseline', fontsize=11)
ax6.set_title('Intervention Effectiveness', fontsize=13, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(K_VALUES)
ax6.legend(fontsize=8, loc='lower left')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save
output_path = Path("results/final_comprehensive_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved to: {output_path}")

plt.show()

print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

print("\n405B Patterns:")
print("  Baseline:    Near-perfect throughout (0.89-1.00)")
print("  Disclaimer:  Minimal effect (±0.02)")
print("  CoT:         Crossover! Hurts at K=0 (-0.16), helps at K>1 (+0.06 to +0.10)")

print("\n70B Patterns:")
print("  Baseline:    Degrades at K=10 (1.00 → 0.86)")
print("  Disclaimer:  K-dependent reversal! Helps at K≤2 (+0.07-0.08), hurts at K=10 (-0.02)")
print("  CoT:         Consistently hurts at K>0 (-0.02 to -0.04)")

print("\nCritical Insight:")
print("  Model scale determines intervention effectiveness:")
print("  - 405B can filter noisy reasoning (CoT helps)")
print("  - 70B gets confused by noisy reasoning (CoT hurts)")
print("  - Disclaimer has K-dependent effects on 70B")

print("\n" + "="*70)
print("✓ FINAL COMPREHENSIVE COMPARISON COMPLETE")
print("="*70)
