"""Shared figure style for the report: colours per model, matplotlib settings, saving."""
import matplotlib.pyplot as plt

# Fixed identity colours (validated palette: first three categorical slots).
COLORS = {"resnet_A": "#2a78d6", "resnet_B": "#eb6834", "hog_svm": "#1baf7a"}
LABELS = {"resnet_A": "ResNet-18 A (original images)", "resnet_B": "ResNet-18 B (background swap, p = 0.5)",
          "hog_svm": "HOG + SVM"}
INK, INK2, RULE, SURFACE = "#0b0b0b", "#52514e", "#d6d5d0", "#fcfcfb"


def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": RULE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ebeae6", "grid.linewidth": 0.8,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "lines.linewidth": 2, "legend.frameon": False,
    })


def saver(fig_dir):
    def save(fig, name):
        fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.show()
    return save
