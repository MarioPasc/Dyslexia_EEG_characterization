# Preprocessing submodule

The preprocessing pipeline takes an experiment npz data corresponding to an specific stimulus frequency (2, 8, 20), and outputs four files:

- CT_UP_preprocess.npz (Control patients, UP condition)
- CT_DOWN_preprocess.npz (Control patients, DOWN condition)
- DD_UP_preprocess.npz (Dyslexia patients, UP condition)
- DD_DOWN_preprocess.npz (Dyslexia patients, DOWN condition)

These files underwent a `StandardScaler` processing and a Two-way zero-phase lag finite impulse response (FIR) Least-Squares filter for the specified interest bands.
The following code block represents a normal execution using the fast mode:

```bash
(pyddeeg) (base) mariopasc@mariopasc-System-Product-Name:~/Python/Projects/Dyslexia_EEG_characterization$ python src/pyddeeg/signal_processing/preprocessing/pipeline.py --config src/pyddeeg/signal_processing/preprocessing/assets/config_example.yaml 
2025-03-13 17:47:51,067 - eeg_preprocessing - INFO - Loading configuration from src/pyddeeg/signal_processing/preprocessing/assets/config_example.yaml
2025-03-13 17:47:51,069 - eeg_preprocessing - INFO - Starting EEG preprocessing pipeline for age=7, stim=2
2025-03-13 17:47:51,069 - eeg_preprocessing - INFO - Processing mode: fast
2025-03-13 17:47:51,069 - eeg_preprocessing - INFO - Loading data from /home/mariopasc/Python/Datasets/EEG/timeseries/original/data_timeseries_7_2.npz
2025-03-13 17:47:52,116 - eeg_preprocessing - INFO - Data loaded successfully with shapes: CT_UP=(34, 32, 68000), CT_DOWN=(34, 32, 68000), DD_UP=(15, 32, 68000), DD_DOWN=(15, 32, 68000)
2025-03-13 17:47:52,116 - eeg_preprocessing - INFO - Preprocessing Control UP data
2025-03-13 17:47:52,116 - eeg_preprocessing - INFO - Starting data preprocessing
2025-03-13 17:47:52,116 - eeg_preprocessing - INFO - Using accelerated processing with dask
[########################################] | 100% Completed | 16.72 s
2025-03-13 17:48:10,142 - eeg_preprocessing - INFO - Preprocessing Control DOWN data
2025-03-13 17:48:10,142 - eeg_preprocessing - INFO - Starting data preprocessing
2025-03-13 17:48:10,142 - eeg_preprocessing - INFO - Using accelerated processing with dask
[########################################] | 100% Completed | 16.72 s
2025-03-13 17:48:28,501 - eeg_preprocessing - INFO - Preprocessing Dyslexia UP data
2025-03-13 17:48:28,501 - eeg_preprocessing - INFO - Starting data preprocessing
2025-03-13 17:48:28,501 - eeg_preprocessing - INFO - Using accelerated processing with dask
[########################################] | 100% Completed | 6.69 ss
2025-03-13 17:48:36,079 - eeg_preprocessing - INFO - Preprocessing Dyslexia DOWN data
2025-03-13 17:48:36,079 - eeg_preprocessing - INFO - Starting data preprocessing
2025-03-13 17:48:36,079 - eeg_preprocessing - INFO - Using accelerated processing with dask
[########################################] | 100% Completed | 6.70 ss
2025-03-13 17:48:43,204 - eeg_preprocessing - INFO - Saving preprocessed data to /home/mariopasc/Python/Datasets/EEG/timeseries/processed
2025-03-13 17:48:47,213 - eeg_preprocessing - INFO - Preprocessing completed successfully
```
