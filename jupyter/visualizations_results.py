import os
from pathlib import Path
import numpy as np
from pyddeeg.classification.visualize import plot_auc_clusters, plot_coeff_heat, plot_decision_heat, plot_feature_sel, plot_fold_box
import matplotlib.pyplot as plt

ROOT = Path("/home/mariopasc/Python/Results/EEG/2hz_stim_delta/data")

RESULTS = {
    electrode.split("_")[0]: 
    {
        "cv_results": os.path.join(ROOT, electrode, "cv_results.npz"),
        "stats": os.path.join(ROOT, electrode, "stats.npz")
    } 
    for i, electrode in enumerate(os.listdir(ROOT))}
WINDOW_SIZE = os.listdir(ROOT)[0].split("_")[2]
DIRECTION = os.listdir(ROOT)[0].split("_")[-1]
print("Window Size: (ms)", WINDOW_SIZE)
print("Direction:", DIRECTION)
electrode_try = "T7"
results_cv = np.load(RESULTS[electrode_try]["cv_results"], allow_pickle=True)
stats = np.load(RESULTS[electrode_try]["stats"], allow_pickle=True)
print("CV Results:", results_cv.files)
print("Stats:", stats.files)

ROOT = Path("/home/mariopasc/Python/Results/EEG/2hz_stim_delta/data")
for electrode, paths in RESULTS.items():
    out_dir = Path("/home/mariopasc/Python/Results/EEG/2hz_stim_delta/results") / electrode
    out_dir.mkdir(exist_ok=True)

    fig = plot_auc_clusters(
        cv_path=paths["cv_results"],
        stats_path=paths["stats"],
        title=f"2Hz Stimulus Delta - {electrode} - {WINDOW_SIZE}ms - {DIRECTION}",
    )
    fig.savefig(out_dir / "auc_clusters.png", dpi=300)
    plt.close(fig)
    
    fig = plot_decision_heat(
        cv_path=paths["cv_results"],
    )
    fig.savefig(out_dir / "decision_heat.png", dpi=300)
    plt.close(fig)

    fig = plot_fold_box(
        cv_path=paths["cv_results"],
        stats_path=paths["stats"],
    )
    fig.savefig(out_dir / "fold_box.png", dpi=300)
    plt.close(fig)

    fig = plot_coeff_heat(
        cv_path=paths["cv_results"],
        stats_path=paths["stats"],
    )
    fig.savefig(out_dir / "coeff_heat.png", dpi=300)
    plt.close(fig)

    fig = plot_feature_sel(
        stats_path=paths["stats"],
        cv_path=paths["cv_results"],
    )
    fig.savefig(out_dir / "feature_sel.png", dpi=300)
    plt.close(fig)
    print(f"Plots saved for {electrode} in {out_dir}")