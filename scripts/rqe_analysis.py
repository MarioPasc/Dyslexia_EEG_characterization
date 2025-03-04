from pyddeeg.utils.synthetic_signal_generator import synth_eeg, synth_signal
from pyddeeg.signal_processing.rqa_toolbox import rqe_correlation_index, rqe_analysis

import scipy.stats as stats
import numpy as np

import matplotlib.pyplot as plt
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])


if __name__ == "__main__":
    # Define the characteristics: time intervals and corresponding distribution parameters.
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

    # Generate the synthetic signal using scipy.stats.norm as the PDF.
    timestamps, orig_sig, signal, results = synth_signal.synthetic_signal_gen(
        characteristics=characteristics,
        base_mean=0,
        base_std=0.5,
        pdf_function=stats.norm,
    )

    synth_signal.plot_piecewise_mean_std(
        t=timestamps, original_sig=orig_sig, results=results, save="./scripts/panel.svg"
    )

    # Parameters from the paper
    window_size = 50
    shift = 1
    lag = 1
    embedding = 10
    radius = 80
    min_line_length = 5
    distance = "euclidean"

    # Perform RQE analysis
    epochs_timestamps, rqa_measures = rqe_analysis(
        signal,
        timestamps,
        window_size,
        shift,
        lag,
        embedding,
        radius,
        min_line_length,
        distance,
    )

    # Calculate RQE correlation index
    corr_timestamps, correlation_index, _ = rqe_correlation_index(
        signal,
        timestamps,
        window_size,
        shift,
        lag,
        embedding,
        radius,
        min_line_length,
        distance,
    )

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, signal)
    plt.title("Original Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.ylim([10, -15])

    plt.subplot(3, 1, 2)
    plt.plot(epochs_timestamps, rqa_measures["DET"], label="DET")
    plt.plot(epochs_timestamps, rqa_measures["LAM"], label="LAM")
    plt.plot(epochs_timestamps, rqa_measures["RR"], label="RR")
    plt.title("RQA Measures")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(corr_timestamps, correlation_index)
    plt.title("RQE Correlation Index")
    plt.xlabel("Time")
    plt.ylabel("Index Value")

    plt.tight_layout()
    plt.savefig("./scripts/rqe_analysis.svg", format="svg")
    plt.show()
