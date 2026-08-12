import matplotlib
matplotlib.use('Agg')  # ← Fix: saves to file without needing a display
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# YOUR RESULTS — edit these values as needed
# ─────────────────────────────────────────────
metrics = ['Accuracy', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
vqax_base = [79.5, 0.596, 0.433, 0.309, 0.222, 0.205, 0.469, 0.868]
vqax_lora = [70.8, 0.523, 0.349, 0.231, 0.156, 0.169, 0.402, 0.575]

# Normalize Accuracy (%) to 0–1 scale for fair comparison
base_norm = [v / 100 if i == 0 else v for i, v in enumerate(vqax_base)]
lora_norm = [v / 100 if i == 0 else v for i, v in enumerate(vqax_lora)]


# ─────────────────────────────────────────────
# CHART 1: Grouped Bar Chart
# ─────────────────────────────────────────────
def plot_grouped_bar():
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bars1 = ax.bar(x - width / 2, base_norm, width,
                   label='vqaX Base', color='#4CC9F0', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, lora_norm, width,
                   label='vqaX LoRA', color='#F72585', edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars1, vqax_base):
        label = f'{val}%' if val > 1 else f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                label, ha='center', fontsize=8.5, color='#4CC9F0', fontweight='bold')

    for bar, val in zip(bars2, vqax_lora):
        label = f'{val}%' if val > 1 else f'{val:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                label, ha='center', fontsize=8.5, color='#F72585', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, color='white')
    ax.set_ylabel('Score (Accuracy normalized to 0–1)', fontsize=11, color='white')
    ax.set_title('actX Results: vqaX Base vs vqaX LoRA — Full Metric Comparison',
                 fontsize=14, color='white', fontweight='bold', pad=20)
    ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    ax.set_ylim(0, 1.15)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('grouped_bar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()  # ← Fix: close after saving
    print("✅ Chart 1 saved: grouped_bar_comparison.png")


# ─────────────────────────────────────────────
# CHART 2: Performance Drop Bar Chart
# ─────────────────────────────────────────────
def plot_performance_drop():
    diff = [b - l for b, l in zip(base_norm, lora_norm)]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    colors = ['#FF6B6B' if d > 0 else '#4CC9F0' for d in diff]
    bars = ax.bar(metrics, diff, color=colors, edgecolor='white', linewidth=0.5)

    for bar, d in zip(bars, diff):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003 if d >= 0 else bar.get_height() - 0.013,
                f'{d:.3f}', ha='center', fontsize=9, color='white', fontweight='bold')

    ax.axhline(0, color='white', linewidth=1, linestyle='--')
    ax.set_ylabel('Base − LoRA  (positive = Base is better)', fontsize=11, color='white')
    ax.set_title('Performance Drop: vqaX Base vs vqaX LoRA',
                 fontsize=13, color='white', fontweight='bold', pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('performance_drop.png', dpi=150, bbox_inches='tight')
    plt.close()  # ← Fix: close after saving
    print("✅ Chart 2 saved: performance_drop.png")


# ─────────────────────────────────────────────
# CHART 3: Radar Chart (explanation metrics only)
# ─────────────────────────────────────────────
def plot_radar():
    radar_metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    base_vals = [0.596, 0.433, 0.309, 0.222, 0.205, 0.469, 0.868]
    lora_vals = [0.523, 0.349, 0.231, 0.156, 0.169, 0.402, 0.575]

    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

    # Close the radar loop
    base_plot = base_vals + base_vals[:1]
    lora_plot = lora_vals + lora_vals[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    ax.plot(angles_plot, base_plot, 'o-', linewidth=2, color='#4CC9F0', label='vqaX Base')
    ax.fill(angles_plot, base_plot, alpha=0.25, color='#4CC9F0')

    ax.plot(angles_plot, lora_plot, 'o-', linewidth=2, color='#F72585', label='vqaX LoRA')
    ax.fill(angles_plot, lora_plot, alpha=0.25, color='#F72585')

    ax.set_xticks(angles)
    ax.set_xticklabels(radar_metrics, color='white', fontsize=11)
    ax.set_yticklabels([])
    ax.tick_params(colors='white')
    ax.spines['polar'].set_color('#444')
    ax.yaxis.grid(True, color='#333')
    ax.xaxis.grid(True, color='#444')
    ax.set_title('Explanation Metrics Radar\nvqaX Base vs LoRA',
                 fontsize=13, color='white', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              facecolor='#1a1a2e', edgecolor='white', labelcolor='white', fontsize=11)

    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()  # ← Fix: close after saving
    print("✅ Chart 3 saved: radar_chart.png")


# ─────────────────────────────────────────────
# RUN ALL CHARTS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    plot_grouped_bar()
    plot_performance_drop()
    plot_radar()
    print("\n🎉 All charts generated successfully!")
    print("📁 Find your charts in the same folder as this script:")
    print("   → grouped_bar_comparison.png")
    print("   → performance_drop.png")
    print("   → radar_chart.png")