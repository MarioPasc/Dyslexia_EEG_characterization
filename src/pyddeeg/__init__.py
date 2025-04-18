# src/pyddeeg/__init__.py


# Dictionary mapping channel names to their indices
EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", 
    "FC6", "T7", "C3", "C4", "T8", "TP9", "CP5", "CP1", "CP2", "CP6", 
    "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", 
    "PO10", "Cz"
]

# Create a mapping from channel name to index
CHANNEL_NAME_TO_INDEX = {name: idx for idx, name in enumerate(EEG_CHANNELS)}

# Dictionary mapping metric names to their indices
RQA_METRICS = [
"RR",
"DET",
"L_max",
"L_mean",
"ENT",
"LAM",
"TT",
"V_max",
"V_mean",
"V_ENT",
"W_max",
"W_mean",
"W_ENT",
"CLEAR",
"PERM_ENT"
]

# Create a mapping from metric name to index
METRIC_NAME_TO_INDEX = {name: idx for idx, name in enumerate(RQA_METRICS)}