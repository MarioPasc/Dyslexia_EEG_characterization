import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from typing import List, Tuple, Dict, Any
import json
import pandas as pd
import os
import time
from datetime import datetime
import dask
from dask import delayed
from dask.distributed import Client, progress, performance_report
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import necessary functions from the provided module
from pyddeeg.utils.synthetic_signal_generator.synth_signal import synthetic_signal_gen

try:
    from pyunicorn.timeseries import RecurrencePlot
except ImportError:
    RecurrencePlot = None
    print("[DEBUG] pyunicorn not installed. Install via: pip install pyunicorn")


# Import the functions from the module
def compute_rqa_metrics_for_window(
    window_signal: np.ndarray,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    min_diagonal_line: int = 2,
    min_vertical_line: int = 2,
    min_white_vertical_line: int = 1,
) -> dict:
    """
    Compute comprehensive RQA metrics from a single window of the raw EEG signal
    using pyunicorn's RecurrencePlot.
    """
    if RecurrencePlot is None:
        raise ImportError(
            "pyunicorn is not available. Install with pip install pyunicorn."
        )

    if distance_metric.lower() == "meandist":
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist

        distances = pdist(window_signal.reshape(-1, 1), metric="euclidean")
        mean_dist = np.mean(distances)
        # Normalize radius by mean distance
        normalized_radius = radius / mean_dist
        actual_radius = normalized_radius
        distance_metric = "euclidean"
    else:
        actual_radius = radius

    rp = RecurrencePlot(
        time_series=window_signal,
        dim=embedding_dim,
        tau=time_delay,
        metric=distance_metric,
        threshold=actual_radius,
        silence_level=2,
    )

    # Compute all available RQA metrics
    metrics = {
        "RR": rp.recurrence_rate(),
        "DET": rp.determinism(l_min=min_diagonal_line),
        "L_max": rp.max_diaglength(),
        "L_mean": rp.average_diaglength(l_min=min_diagonal_line),
        "ENT": rp.diag_entropy(l_min=min_diagonal_line),
        "LAM": rp.laminarity(v_min=min_vertical_line),
        "TT": rp.trapping_time(v_min=min_vertical_line),
        # Additional line-based measures
        "V_max": rp.max_vertlength(),
        "V_mean": rp.average_vertlength(v_min=min_vertical_line),
        "V_ENT": rp.vert_entropy(v_min=min_vertical_line),
        # White vertical line measures
        "W_max": rp.max_white_vertlength(),
        "W_mean": rp.average_white_vertlength(w_min=min_white_vertical_line),
        "W_ENT": rp.white_vert_entropy(w_min=min_white_vertical_line),
        # Complexity measures
        "CLEAR": rp.complexity_entropy() if hasattr(rp, "complexity_entropy") else None,
        "PERM_ENT": (
            rp.permutation_entropy() if hasattr(rp, "permutation_entropy") else None
        ),
    }

    return metrics


def compute_rqa_time_series(
    signal: np.ndarray,
    raw_signal_window_size: int,
    embedding_dim: int,
    time_delay: int,
    radius: float,
    distance_metric: str,
    min_diagonal_line: int,
    min_vertical_line: int,
    min_white_vertical_line: int,
    metrics_to_use: List[str],
    stride: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Slide a small window of size 'raw_signal_window_size' over the raw signal, compute RQA
    metrics for each position, and return them as a 2D array of shape (T, L).

    Returns:
        Tuple of (rqa_matrix, all_metrics)
        - rqa_matrix: 2D array of shape (num_positions, len(metrics_to_use))
        - all_metrics: List of complete metric dictionaries for each window position
    """
    n = len(signal)
    num_positions = (n - raw_signal_window_size) // stride + 1
    rqa_results = []
    all_metrics = []  # Store all metrics for each window

    for i in range(num_positions):
        start_idx = i * stride
        window_data = signal[start_idx : start_idx + raw_signal_window_size]
        rqa_vals = compute_rqa_metrics_for_window(
            window_signal=window_data,
            embedding_dim=embedding_dim,
            time_delay=time_delay,
            radius=radius,
            distance_metric=distance_metric,
            min_diagonal_line=min_diagonal_line,
            min_vertical_line=min_vertical_line,
            min_white_vertical_line=min_white_vertical_line,
        )

        # Store complete metrics dictionary
        all_metrics.append(rqa_vals)

        # Extract only the metrics we asked for
        values_of_interest = []
        for key in metrics_to_use:
            values_of_interest.append(rqa_vals.get(key))
        rqa_results.append(values_of_interest)

    return np.array(rqa_results), all_metrics


def compute_rqe_and_correlation(rqa_matrix: np.ndarray, rqa_space_window_size: int):
    """
    Compute two measures for each 'big window' in the RQA time series:
      1) RQE = product of (1 + |rho|)
      2) Corr = average of |rho|
    """
    T, L = rqa_matrix.shape
    if rqa_space_window_size > T:
        rqa_space_window_size = T
        print(f"[WARNING] Adjusted rqa_space_window_size to {T}")

    Q = T - rqa_space_window_size + 1
    rqe_list = []
    corr_list = []
    all_rho_values = []  # Store all correlation values for later analysis

    for start_idx in range(Q):
        sub_block = rqa_matrix[start_idx : start_idx + rqa_space_window_size, :]
        # shape = (rqa_space_window_size, L)
        # We'll treat each column as a sub-series of length rqa_space_window_size
        sub_block_T = sub_block.T

        # Collect pairwise correlations
        pairwise_rho = []
        window_rho_values = {}  # Store by pairs for this window

        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                rho, p_value = stats.spearmanr(sub_block_T[l1, :], sub_block_T[l2, :])
                pairwise_rho.append(abs(rho))
                # Store pair info
                pair_key = f"{l1}_{l2}"
                window_rho_values[pair_key] = {
                    "rho": rho,
                    "abs_rho": abs(rho),
                    "p_value": p_value,
                }

        all_rho_values.append(window_rho_values)

        if len(pairwise_rho) == 0:
            # If L < 2
            rqe_val = 1.0
            corr_val = 0.0
        else:
            # RQE = product(1 + |rho|)
            product_val = 1.0
            for c in pairwise_rho:
                product_val *= 1.0 + c
            rqe_val = product_val

            # Corr = mean(|rho|)
            corr_val = np.mean(pairwise_rho)

        rqe_list.append(rqe_val)
        corr_list.append(corr_val)

    return np.array(rqe_list), np.array(corr_list), all_rho_values


def min_max_normalize(array: np.ndarray) -> np.ndarray:
    """
    Scale the values of 'array' into [0, 1] by min-max normalization.
    If all values are the same, the result is an array of zeros.
    """
    array = np.nan_to_num(x=array)

    min_val = np.min(array)
    max_val = np.max(array)

    if max_val > min_val:
        return (array - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(array)


def generate_synthetic_signal(length=700):
    """Generate a synthetic EEG signal with known characteristics"""
    characteristics = {
        (0, 100): {"mean": 0, "std": 1},
        (101, 140): {"mean": 1, "std": 1},
        (141, 200): {"mean": 1, "std": 4},
        (201, 280): {"mean": 4, "std": 4},
        (281, 300): {"mean": 4, "std": 6},
        (301, 400): {"mean": -5, "std": 6},
        (401, 420): {"mean": 2, "std": 6},
        (421, 500): {"mean": 2, "std": 1},
        (501, 560): {"mean": 0, "std": 1},
        (561, 600): {"mean": 0, "std": 3},
        (601, 700): {"mean": 1, "std": 3},
    }

    # Generate synthetic signal
    t, orig_sig, raw_signal, results = synthetic_signal_gen(
        characteristics=characteristics,
        base_mean=0,
        base_std=0.5,
        pdf_function=stats.norm,
    )

    return t, raw_signal


@delayed
def process_grid_cell(
    raw_signal: np.ndarray,
    t: np.ndarray,
    raw_window: int,
    rqa_window: int,
    embedding_dim: int,
    radius: float,
    time_delay: int,
    distance_metric: str,
    min_diagonal_line: int,
    min_vertical_line: int,
    min_white_vertical_line: int,
    metrics_to_use: List[str],
    stride: int,
    signal_length: int,
) -> Dict:
    """
    Process a single grid cell for the visualization.
    Returns a dictionary with all necessary data for plotting and analysis.
    """
    # Compute RQA time series
    rqa_matrix, all_metrics = compute_rqa_time_series(
        signal=raw_signal,
        raw_signal_window_size=raw_window,
        embedding_dim=embedding_dim,
        time_delay=time_delay,
        radius=radius,
        distance_metric=distance_metric,
        min_diagonal_line=min_diagonal_line,
        min_vertical_line=min_vertical_line,
        min_white_vertical_line=min_white_vertical_line,
        metrics_to_use=metrics_to_use,
        stride=stride,
    )

    # Make sure rqa_window is not larger than T
    T, L = rqa_matrix.shape
    adjusted_rqa_window = min(rqa_window, T)

    # Compute RQE and correlation
    rqe_vals, corr_vals, all_rho_values = compute_rqe_and_correlation(
        rqa_matrix, rqa_space_window_size=adjusted_rqa_window
    )

    # Normalize values
    rqe_norm = min_max_normalize(rqe_vals)
    corr_norm = min_max_normalize(corr_vals)

    # Create time axis for RQE/corr values
    Q = len(rqe_vals)
    big_half = adjusted_rqa_window / 2.0
    rqe_time = []
    for idx in range(Q):
        center_rqa = idx + big_half
        # Approx mapping to raw-signal time index
        approx_raw = center_rqa + raw_window / 2.0
        if approx_raw < signal_length:
            rqe_time.append(approx_raw)
        else:
            rqe_time.append(signal_length - 1)
    rqe_time = np.array(rqe_time)

    # Prepare detailed metrics for saving
    detailed_metrics = {
        "window_indices": list(range(Q)),
        "rqe_time": rqe_time.tolist(),
        "raw_rqe": rqe_vals.tolist(),
        "norm_rqe": rqe_norm.tolist(),
        "raw_corr": corr_vals.tolist(),
        "norm_corr": corr_norm.tolist(),
        "all_rho_values": all_rho_values,
        "all_rqa_metrics": [
            {
                k: float(v) if v is not None and not np.isnan(v) else None
                for k, v in m.items()
            }
            for m in all_metrics
        ],
    }

    return {
        "raw_window": raw_window,
        "rqa_window": adjusted_rqa_window,
        "rqe_norm": rqe_norm,
        "corr_norm": corr_norm,
        "rqe_time": rqe_time,
        "raw_rqe": rqe_vals,
        "raw_corr": corr_vals,
        "detailed_metrics": detailed_metrics,
    }


def save_results_to_csv(results_grid, output_dir, timestamp):
    """Save results to CSV files in a structured way"""
    for i, j, result in results_grid:
        raw_win = result["raw_window"]
        rqa_win = result["rqa_window"]

        # Create subdirectory for this parameter combination
        param_dir = os.path.join(output_dir, f"raw{raw_win}_rqa{rqa_win}")
        os.makedirs(param_dir, exist_ok=True)

        # Save main results
        main_df = pd.DataFrame(
            {
                "time": result["rqe_time"],
                "rqe": result["raw_rqe"],
                "rqe_norm": result["rqe_norm"],
                "corr": result["raw_corr"],
                "corr_norm": result["corr_norm"],
            }
        )
        main_df.to_csv(os.path.join(param_dir, "main_results.csv"), index=False)

        # Save all metrics to CSV (flattened structure)
        metrics_data = []
        for idx, metrics in enumerate(result["detailed_metrics"]["all_rqa_metrics"]):
            row = {"window_idx": idx}
            row.update(metrics)
        metrics_data.append(row)

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(param_dir, "detailed_metrics.csv"), index=False)

        # Save correlation data
        corr_data = []
        for win_idx, window_corrs in enumerate(
            result["detailed_metrics"]["all_rho_values"]
        ):
            for pair_key, pair_data in window_corrs.items():
                row = {"window_idx": win_idx, "pair": pair_key}
                row.update(pair_data)
                corr_data.append(row)

        if corr_data:  # Only create if there's data
            corr_df = pd.DataFrame(corr_data)
            corr_df.to_csv(os.path.join(param_dir, "correlation_data.csv"), index=False)


def save_results_to_json(results_grid, output_dir, timestamp):
    """Save detailed results in JSON format"""
    for i, j, result in results_grid:
        raw_win = result["raw_window"]
        rqa_win = result["rqa_window"]

        # Create subdirectory for this parameter combination
        param_dir = os.path.join(output_dir, f"raw{raw_win}_rqa{rqa_win}")
        os.makedirs(param_dir, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        json_result = {
            "raw_window": int(raw_win),
            "rqa_window": int(rqa_win),
            "detailed_metrics": result["detailed_metrics"],
        }

        # Save as JSON
        with open(os.path.join(param_dir, "full_results.json"), "w") as f:
            json.dump(json_result, f, indent=2)


def create_rqe_grid_visualization_parallel(
    raw_window_sizes,
    rqa_window_sizes,
    embedding_dim=10,
    radius=70.0,
    time_delay=1,
    distance_metric="meandist",
    min_diagonal_line=5,
    min_vertical_line=1,
    min_white_vertical_line=1,
    metrics_to_use=["RR", "DET", "L_max", "ENT", "LAM", "TT"],
    stride=1,
    signal_length=700,
    n_workers=4,
    output_dir="./results",
):
    """
    Create a grid visualization showing RQE and correlation values for different
    combinations of raw_signal_window_size and rqa_space_window_size.
    Uses Dask for parallel processing and saves comprehensive results.

    Parameters:
    - raw_window_sizes: List of values for raw_signal_window_size
    - rqa_window_sizes: List of values for rqa_space_window_size
    - n_workers: Number of parallel workers (default=4)
    - output_dir: Directory to save all outputs
    """
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory structure
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save run parameters
    params = {
        "timestamp": timestamp,
        "raw_window_sizes": raw_window_sizes,
        "rqa_window_sizes": rqa_window_sizes,
        "embedding_dim": embedding_dim,
        "radius": radius,
        "time_delay": time_delay,
        "distance_metric": distance_metric,
        "min_diagonal_line": min_diagonal_line,
        "min_vertical_line": min_vertical_line,
        "min_white_vertical_line": min_white_vertical_line,
        "metrics_to_use": metrics_to_use,
        "stride": stride,
        "signal_length": signal_length,
        "n_workers": n_workers,
    }

    with open(os.path.join(run_dir, "parameters.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Start Dask client
    client = Client(n_workers=n_workers, threads_per_worker=1)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    # Generate synthetic signal (only once)
    t, raw_signal = generate_synthetic_signal(signal_length)

    # Save the raw signal for reference
    raw_df = pd.DataFrame({"time": t, "signal": raw_signal})
    raw_df.to_csv(os.path.join(run_dir, "raw_signal.csv"), index=False)

    N = len(raw_window_sizes)
    M = len(rqa_window_sizes)

    # Compute actual radius based on signal range
    signal_range = np.max(raw_signal) - np.min(raw_signal)
    actual_radius = (radius / 100.0) * signal_range

    # Create a list of delayed tasks for each grid cell
    tasks = []
    for i, raw_window in enumerate(raw_window_sizes):
        for j, rqa_window in enumerate(rqa_window_sizes):
            task = process_grid_cell(
                raw_signal=raw_signal,
                t=t,
                raw_window=raw_window,
                rqa_window=rqa_window,
                embedding_dim=embedding_dim,
                radius=actual_radius,
                time_delay=time_delay,
                distance_metric=distance_metric,
                min_diagonal_line=min_diagonal_line,
                min_vertical_line=min_vertical_line,
                min_white_vertical_line=min_white_vertical_line,
                metrics_to_use=metrics_to_use,
                stride=stride,
                signal_length=signal_length,
            )
            tasks.append((i, j, task))

    # Create performance report
    with performance_report(filename=os.path.join(run_dir, "dask-report.html")):
        # Compute all tasks in parallel
        start_time = time.time()
        print("Starting parallel computation...")
        results_with_indices = []
        for i, j, task in tasks:
            results_with_indices.append((i, j, dask.compute(task)[0]))

        end_time = time.time()
        print(f"Parallel computation completed in {end_time - start_time:.2f} seconds!")

    # Save all results to CSV and JSON
    print("Saving results to CSV files...")
    save_results_to_csv(results_with_indices, run_dir, timestamp)

    print("Saving results to JSON files...")
    save_results_to_json(results_with_indices, run_dir, timestamp)

    # Create figure with grid
    fig = plt.figure(figsize=(5 * M, 4 * N))
    gs = gridspec.GridSpec(N, M)

    # Create the visualization from results
    # Loop over the computed results and plot them
    for i, j, result in results_with_indices:
        ax = fig.add_subplot(gs[i, j])  # Create subplot in the grid

        # Extract the relevant data
        rqe_time = result["rqe_time"]
        rqe_norm = result["rqe_norm"]
        corr_norm = result["corr_norm"]

        # Plot RQE (blue) and correlation (orange) over time
        ax.plot(rqe_time, rqe_norm, label="RQE", color="blue")
        ax.plot(rqe_time, corr_norm, label="|Corr|", color="orange")

        # Label and legend
        ax.set_title(f"RawWin={result['raw_window']} | RQAWin={result['rqa_window']}")
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Normalized Values")
        ax.legend(loc="best")

    # Adjust layout and display
    plt.tight_layout()

    # Save plot as SVG and PNG
    svg_file = os.path.join(run_dir, "rqe_window_analysis.svg")
    png_file = os.path.join(run_dir, "rqe_window_analysis.png")
    plt.savefig(svg_file)
    plt.savefig(png_file, dpi=300)

    print(f"Results saved in: {run_dir}")

    # Shut down the client
    client.close()

    return run_dir


# Example usage
if __name__ == "__main__":
    # Define lists of window sizes to test
    raw_window_sizes = [10, 50, 100, 150]
    rqa_window_sizes = [10, 20, 30, 40]

    # Create grid visualization with parallel processing
    output_dir = create_rqe_grid_visualization_parallel(
        raw_window_sizes=raw_window_sizes,
        rqa_window_sizes=rqa_window_sizes,
        embedding_dim=10,
        radius=70.0,
        time_delay=1,
        distance_metric="meandist",
        min_diagonal_line=5,
        stride=1,
        n_workers=4,  # Specify number of parallel workers
        output_dir="./scripts/rqe/results",
    )

    print(f"Analysis complete! Results available in: {output_dir}")
