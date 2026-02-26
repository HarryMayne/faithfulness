# %%
import pandas as pd
import matplotlib.pyplot as plt
import re

# %%
standard = pd.read_csv("../experiments/cot/predictions_normal_simulatability_averaged.csv")
cot = pd.read_csv("../experiments/cot/predictions_cot_simulatability_averaged.csv")

# %%
def extract_model_size(model_name: str) -> float:
    """Extract model size in billions of parameters from model name."""
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
    if match:
        return float(match.group(1))
    return None

def plot_normalized_gain_vs_params(dfs: dict, show_ci: bool = True, fontsize: int = 15, savepath: str = None):
    """
    Plot normalized simulatability gain vs parameter count.

    Args:
        dfs: Dict mapping label to DataFrame
        show_ci: Whether to show confidence interval shading (default True)
        fontsize: Font size for all text elements (default 12)
        savepath: Path to save figure (default None, no save)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#082b5a', '#b33f00']  # Even darker blue, even darker orange

    all_sizes = []
    for (label, df), color in zip(dfs.items(), colors):
        df = df.copy()
        df['model_size_b'] = df['model'].apply(extract_model_size)
        df_plot = df[df['model_size_b'].notna()].sort_values('model_size_b')

        if len(df_plot) == 0:
            continue

        x = df_plot['model_size_b'].values
        y = df_plot['normalized_gain'].values
        all_sizes.extend(x)

        if show_ci:
            ci_lower = df_plot['norm_gain_ci_lower'].values
            ci_upper = df_plot['norm_gain_ci_upper'].values
            ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

        # Plot with markers but exclude markers from legend
        line, = ax.plot(x, y, marker='x', linestyle='-', color=color, markersize=8)
        # Add legend entry with line only
        ax.plot([], [], linestyle='-', color=color, label=label)

    #ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Parameter count (B)', fontsize=fontsize)
    ax.set_ylabel('Normalized simulatability gain (%)', fontsize=fontsize)
    ax.set_ylim(bottom=0)

    # Set x-ticks only at actual model sizes, remove minor ticks
    unique_sizes = sorted(set(all_sizes))
    ax.set_xticks(unique_sizes)
    ax.set_xticklabels([f'{s}' if s == int(s) else f'{s}B' for s in unique_sizes], fontsize=fontsize)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(axis='y', labelsize=fontsize)

    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

# %%
plot_normalized_gain_vs_params({
    'User-facing explanations': standard,
    'Chain-of-thought': cot,
}, show_ci=True, savepath='../figures/cot_vs_standard_normalized_gain.pdf')

# %%
