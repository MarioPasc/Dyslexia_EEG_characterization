# Dyslexia EEG Characterization (pyddeeg)
## Overview

Developmental dyslexia (DD) manifests as a reading and learning difficulty despite normal intelligence and education. Research has shown that atypical neural dynamics—in particular, changes in the temporal organisation and non‑linear dynamics of electroencephalographic (EEG) rhythms—can contribute to this disorder. pyddeeg is a Python package that implements a complete pipeline for characterising EEG signals recorded from children with and without dyslexia. The library performs:

- Pre‑processing of raw EEG time‑series: zero‑lag band‑pass filtering and standardisation by frequency band to produce band‑limited, time–frequency tensors. The filter employed is a two‑pass, zero‑phase FIR design that removes phase distortion by filtering forwards and backwards.
- Extraction of non‑linear features: Recurrence Quantification Analysis (RQA) metrics are computed on sliding windows of the pre‑processed signals. Optional optimisation of the embedding delay (τ) and dimension (m) is performed using mutual information and false‑nearest‑neighbour criteria.
- Classification of dyslexic vs control subjects: a nested k×h cross‑validation pipeline with window‑specific hyper‑parameter tuning yields time‑resolved decision scores and feature‑importance measures. Hyper‑parameters are selected with Optuna; feature selection is performed using SelectKBest with F‑statistics (selector.yaml) and separate models are fitted for each time window.
- Statistical evaluation: cluster‑based permutation tests and per‑window Area Under the Curve (AUC) analyses assess whether classification performance at any time window is significantly above chance.

The package is modular; each step can be run independently or orchestrated as a complete workflow.
## Installation

1. Clone the repository and install the package in editable mode (Python ≥ 3.10):
```bash
    git clone https://.com/MarioPasc/Dyslexia_EEG_characterization.git
    cd Dyslexia_EEG_characterization
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e .  # installs pyddeeg and its dependencies
```
2. Dependencies. pyddeeg depends on the scientific Python stack (numpy, scipy, pandas), mne for EEG handling, scikit‑learn for machine learning, pyunicorn/nolitsa for recurrence analysis, dask for parallel processing and optuna for hyper‑parameter optimisation. These are declared in pyproject.toml and will be installed automatically.

3. Data. Raw EEG data are not distributed with this repository. The preprocessing stage expects .npz files containing four tensors of shape (subjects, channels, time, condition) for control up, control down, dyslexic up and dyslexic down conditions. See zerolag_config.yaml for an example directory layout and channel order.

## Pre‑processing pipeline
1. Zero‑lag filtering and standardisation. The first stage reads raw time‑series, standardises each subject’s channels and applies a two‑pass zero‑phase FIR band‑pass filter across multiple frequency bands (Delta–Gamma). The filter design uses a least‑squares FIR with symmetric forward and backward filtering to achieve zero‑phase response. An example configuration is provided in zerolag_config.yaml. Run the stage directly with `python src/pyddeeg/preprocessing/pipelines/zerolag_preprocessing.py --config path/to/zerolag_config.yaml` Four files are produced: CT_UP_preprocess_<stim>.npz, CT_DOWN_preprocess_<stim>.npz, DD_UP_preprocess_<stim>.npz and DD_DOWN_preprocess_<stim>.npz. Each contains a 4‑D tensor (subjects, channels, time, bands) of band‑filtered data.
2. Recurrence Quantification Analysis (RQA). Recurrence analysis quantifies the dynamics of a time series in a reconstructed phase‑space. For each sliding time window (size and stride configurable), the signal is embedded using Takens’ method. If optimise_takens is set, the optimal delay (τ) is chosen as the first local minimum of the mutual information curve and the optimal dimension (m) is selected when the fraction of false nearest neighbours drops below a threshold. The recurrence threshold (radius) is then selected to achieve a target recurrence rate. RQA metrics—including recurrence rate (RR), determinism (DET), laminarity (LAM), trapping time (TT), entropy measures and vertical/white line statistics—are computed on each window. The RQA step can operate in parallel using Dask and supports Hilbert transforms and multiple window sizes. A sample configuration is provided in pipeline_config.yaml lines 65–125. To run RQA on the output of the zero‑lag stage, execute `python src/pyddeeg/preprocessing/pipelines/rqa_windows_picasso.py --config path/to/rqa_config.yaml` (or run the unified pipeline described below).
3. Dataset reorganisation. After computing RQA metrics, the per‑window results are reorganised into a dataset directory for classification. The script reorganize_per_window_results groups RQA metrics into DD_UP_metrics.npz and CT_UP_metrics.npz (or DOWN equivalents) with shape (subjects, features, windows).
4. Unified pipeline. `preprocessing.py` orchestrates all three stages using a master YAML. It deep‑merges user overrides with per‑stage templates and skips stages when inputs are unchanged. Typical usage is
```
python src/pyddeeg/preprocessing/pipelines/preprocessing.py \
    --config src/pyddeeg/preprocessing/cfgs/pipeline_config.yaml \
    --stim 20 --channel Fz --do-rqa
```
This command performs zero‑lag filtering, RQA extraction and reorganisation for the specified stimulus and electrode.
## Classification pipeline

The classification module uses RQA features to discriminate dyslexic from control subjects. The main entry point is tune_hyperparams_electrode.py, which performs:
1. Nested cross‑validation: k outer folds and h inner folds are created using stratified group splits so that each subject forms a group. For each outer fold the inner folds are used to tune hyper‑parameters.
2. Per‑window hyper‑parameter tuning: for each time window, Optuna explores a user‑defined search space for the classifier and feature selector. The default selector is SelectKBest with k between 2 and 15 and F‑test scoring (see selector.yaml)
3. Multi‑window estimation: after tuning, a MultiWindowEstimator fits a separate pipeline (scaler → selector → model) to each time window and aggregates predictions across windows.
4.Performance metrics and statistics: the function nested_evaluate returns decision scores, per‑fold AUCs, tuned parameters, selector diagnostics (feature scores, p‑values, selection masks) and model coefficients. It also performs a cluster‑based permutation test to identify contiguous windows where dyslexic and control groups are separable beyond chance.

To run the classification for a single electrode, create a settings.yaml pointing to your dataset root, window (e.g. window_750 for 750 ms windows), classification direction (up or down) and the paths of the hyper‑parameter, optuna and selector configuration files. Example fields are provided in the sample settings.yaml. Then run:
```
python src/pyddeeg/classification/tests/tune_hyperparams_electrode.py \
    --settings path/to/settings.yaml --electrode Fz --threads 8
```
This produces cv_results.npz and stats.npz inside output_dir/<electrode>_<window>_<direction>. Use analyze.py to run additional statistical tests:
```
python src/pyddeeg/classification/statistics/analyze.py cv_results.npz --out stats.npz --n_perm 10000 --alpha 0.05
```
Configuring models and hyper‑parameters
- Model configuration (model.yaml): specify the dotted path to a scikit‑learn estimator and define search spaces for its hyper‑parameters using Optuna’s format (low/high or categorical choices). For example, logistic regression with search over C and penalty.
- Optuna configuration (optuna.yaml): choose the sampler and pruner, number of trials and whether to persist studies to disk.
- Selector configuration (selector.yaml): choose a feature selector (e.g. SelectKBest) and the range of k to explore.

## Examples and outputs

After running classification, inspection of the output arrays reveals when RQA features discriminate dyslexic subjects. The fold_auc array has shape (outer folds, 2, windows) where index 0 is ROC AUC and index 1 is average precision. The feature_importance dict provides window‑wise averages of feature scores, p‑values and selection frequency. Significant clusters identified by the permutation test highlight temporal regions where group differences are robust.
Recurrence Quantification Epoch (RQE) and visualisation

In addition to the standard RQA analysis, pyddeeg provides utilities for computing an Recurrence Quantification Epoch (RQE) index, which measures the coherence among RQA metrics over a second sliding window. After computing a matrix of RQA metrics for a time‑windowed signal, compute_rqe_batch multiplies the absolute Spearman correlations between all metric pairs within a larger “RQA‑space” window to quantify temporal coherence. The function returns both the RQE values and the mean pairwise correlation, facilitating assessment of dynamical consistency. High RQE values imply that RQA metrics co‑vary over time, indicating organised dynamics, whereas low values suggest more independent or complex behaviour.

Visualisation scripts under `src/pyddeeg/utils/visualize/rqe/` allow users to overlay RQE curves on band‑filtered EEG signals, inspect RQA matrices for each frequency band and patient, and compare RQE across dyslexic and control groups. For example, `rqa_matrix_per_band.py` plots recurrence plots and computes metrics such as recurrence rate, determinism, laminarity and trapping time for each band. `rqe_per_bandwidth.py` computes RQE for each band and overlays it on the time‑series to highlight epochs of dynamical coherence. These tools support exploratory analyses and may inform the selection of informative time intervals for classification.
Citing this work

If you use pyddeeg in your research, please cite the following description in your paper or thesis:

```
M. Pascual-González., I. Rodríguez-Rodríguez. pyddeeg: time‑resolved EEG classification with windowed non‑linear recurrence metrics for developmental dyslexia research. University of Málaga, 2025.
```
## License

The project is licensed under the MIT License. See LICENSE for details.
