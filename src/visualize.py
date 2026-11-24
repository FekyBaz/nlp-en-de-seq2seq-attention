"""
Visualization routines for training loss curves, attention heatmaps,
and length-stratified comparative analysis.
"""

from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_loss_curves(
    baseline_history: Dict[str, List[float]],
    attention_history: Dict[str, List[float]],
    save_path: str = "outputs/figures/loss_curves.png",
):
    """
    Plots Training and Validation Loss convergence curves for both models side-by-side.
    """
    epochs_base = range(1, len(baseline_history["train_loss"]) + 1)
    epochs_attn = range(1, len(attention_history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: Training Loss Comparison
    axes[0].plot(epochs_base, baseline_history["train_loss"], label="Baseline Train Loss", color="#1f77b4", lw=2.2, linestyle="--")
    axes[0].plot(epochs_attn, attention_history["train_loss"], label="Attention Train Loss", color="#2ca02c", lw=2.5)
    axes[0].set_title("Training Loss Convergence", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].legend(frameon=True, facecolor="white", framealpha=0.9)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Validation Loss Comparison
    axes[1].plot(epochs_base, baseline_history["val_loss"], label="Baseline Val Loss", color="#d62728", lw=2.2, linestyle="--")
    axes[1].plot(epochs_attn, attention_history["val_loss"], label="Attention Val Loss", color="#9467bd", lw=2.5)
    axes[1].set_title("Validation Loss Comparison", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[1].legend(frameon=True, facecolor="white", framealpha=0.9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[*] Loss curves successfully saved to {save_path}")


def plot_attention_heatmap(
    src_tokens: List[str],
    trg_tokens: List[str],
    attention_matrix: np.ndarray,
    save_path: str,
    title: Optional[str] = None,
):
    """
    Renders 2D Attention Heatmap (source tokens on x-axis, predicted target tokens on y-axis).
    """
    # Truncate matrix if dimensions mismatch
    matrix = attention_matrix[: len(trg_tokens), : len(src_tokens)]

    fig = plt.figure(figsize=(max(8, len(src_tokens) * 0.7), max(6, len(trg_tokens) * 0.55)), dpi=300)
    ax = fig.add_subplot(111)

    cax = ax.matshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04, label="Attention Weight (alpha)")

    # Set axes labels
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(trg_tokens)))

    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(trg_tokens, fontsize=11)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Grid between cells
    ax.set_xticks(np.arange(-0.5, len(src_tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(trg_tokens), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlabel("Source Sentence Tokens (English)", fontsize=12, fontweight="bold", labelpad=12)
    ax.set_ylabel("Generated Target Tokens (German)", fontsize=12, fontweight="bold", labelpad=12)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[*] Attention heatmap saved to {save_path}")


def plot_length_analysis(
    bucket_results: Dict[str, Dict[str, Dict[str, float]]],
    save_path: str = "outputs/figures/length_analysis.png",
):
    """
    Bar plot comparing BLEU-4 and ROUGE-L across sentence length buckets
    between Baseline Seq2Seq and Seq2Seq with Attention.
    """
    categories = list(bucket_results.keys())
    base_bleu = [bucket_results[c]["Baseline"]["BLEU-4"] for c in categories]
    attn_bleu = [bucket_results[c]["Attention"]["BLEU-4"] for c in categories]

    base_rouge = [bucket_results[c]["Baseline"]["ROUGE-L"] for c in categories]
    attn_rouge = [bucket_results[c]["Attention"]["ROUGE-L"] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # BLEU-4 comparison
    axes[0].bar(x - width / 2, base_bleu, width, label="Baseline LSTM", color="#1f77b4", alpha=0.85)
    axes[0].bar(x + width / 2, attn_bleu, width, label="Seq2Seq + Attention", color="#2ca02c", alpha=0.85)
    axes[0].set_title("BLEU-4 by Source Sentence Length", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_ylabel("BLEU-4 Score", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=10)
    axes[0].legend(frameon=True, facecolor="white")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Annotate values
    for i in range(len(categories)):
        axes[0].text(x[i] - width / 2, base_bleu[i] + 0.5, f"{base_bleu[i]:.1f}", ha="center", fontsize=9)
        axes[0].text(x[i] + width / 2, attn_bleu[i] + 0.5, f"{attn_bleu[i]:.1f}", ha="center", fontsize=9, fontweight="bold")

    # ROUGE-L comparison
    axes[1].bar(x - width / 2, base_rouge, width, label="Baseline LSTM", color="#d62728", alpha=0.85)
    axes[1].bar(x + width / 2, attn_rouge, width, label="Seq2Seq + Attention", color="#9467bd", alpha=0.85)
    axes[1].set_title("ROUGE-L by Source Sentence Length", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_ylabel("ROUGE-L Score", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories, fontsize=10)
    axes[1].legend(frameon=True, facecolor="white")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    for i in range(len(categories)):
        axes[1].text(x[i] - width / 2, base_rouge[i] + 0.8, f"{base_rouge[i]:.1f}", ha="center", fontsize=9)
        axes[1].text(x[i] + width / 2, attn_rouge[i] + 0.8, f"{attn_rouge[i]:.1f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[*] Length analysis plot saved to {save_path}")
