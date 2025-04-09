# Pipelines

- `zerolag_preprocessing.py`. Applied zerolag preprocessing to raw EEG data to generate a tensor of (patients, channels, data) to (patients, channels, data, bands)
- `rqe_preproc_picasso.py`. Given a zerolag preprocessed tensor, apply RQE preprocessing to generate a tensor of (patients, channels, data, bandwidths, rqa_metrics). This preprocessing is applied to all bands, all patients, and all channels.
- `stimuli_rqa_preproc.py`. Given a zerolag processed tensor, and a target bandwidth, as well as a list of possible window sizes in the temporal dimension, apply RQA preprocessing to generate a tensor of (patients, channels, data, 1,rqa_metrics, windows).