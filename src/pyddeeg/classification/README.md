## Methodology for Time-Resolved Classification Analysis

### 1. Data and Participants
- **Participants**: Two groups—children with developmental dyslexia (DD) and controls (CT).
- **EEG Recordings**: 32 electrodes at 500 Hz sampling, with a 40 Hz auditory stimulus paradigm.
- **Preprocessed Data**: Gamma-band (30–80 Hz) EEG extracted, resulting in tensors of shape:
  - CT: 34 × 32 × 68000
  - DD: 15 × 32 × 68000

### 2. Windowing and Feature Extraction
1. **Windowing**:
   - Each 68000-point signal (per electrode) is segmented into 100 ms windows (50% overlap ⇒ 50 ms stride).
   - Thus, each electrode’s time series is converted to multiple overlapping windows.
2. **Recurrence Quantification Analysis (RQA)**:
   - For each window, compute 15 RQA metrics (e.g., RR, DET, LAM, ENTR, etc.).
   - Final shape (per participant × electrode) becomes (number_of_windows × 15 features).

### 3. Classification Setup
1. **Per-Electrode Classification**:
   - Each electrode is analyzed separately to capture electrode‐specific differences.
   - Label for each window = participant group (DD or CT).
2. **Cross‐Validation**:
   - Use a 5‐fold stratified cross‐validation **by participant**:
     - Entire participants appear exclusively in train or test fold (no overlap).
     - Stratification ensures similar DD/CT ratios in each fold.
   - Within each fold:
     1. Train on all windows from train‐fold participants.
     2. Evaluate on all windows from test‐fold participants.

### 4. Hyperparameter Tuning and Feature Selection
- **Hyperparameter Tuning**:
  - Within each fold, perform feature selection and classifier hyperparameter tuning (e.g., grid search) using only the training participants’ windows.
  - This avoids data leakage and overfitting.
- **Permutation Tests**:
  - After final model training, use a participant‐level label shuffle to evaluate the significance of the observed classification metrics.

### 5. Time‐Resolved Performance and AUC Curves
1. **Window‐Level Predictions**:
   - For each test set participant in a fold, run every time window (for a given electrode) through the trained classifier.
   - Collect predicted probabilities (or decision function values) for DD vs. CT.
2. **Compute Per‐Window AUC**:
   - **Across participants**, aggregate the predictions for the *same time window index* and calculate the ROC‐AUC (or PR‐AUC).
   - Repeat for each time window index to obtain an AUC value as a function of time.
3. **Visualization**:
   - **Temporal Evolution**:
     - Plot AUC vs. time to visualize how classification accuracy evolves.  
   - **Topographic Distribution**:
     - For selected time points (e.g., suspected moments of neural adaptation), map the electrode‐wise AUC values onto a 2D scalp layout (topomap). 
     - Highlights where (which electrodes) the classifier best separates DD vs. CT at those time windows.

### 6. Final Participant‐Level Performance (Optional)
- To get an overall participant‐level measure:
  - Aggregate window predictions (e.g., average predicted probability) for each participant.
  - Compute ROC‐AUC across participants for the final classification measure.
  - This complements the time‐resolved AUC and confirms the discriminative power at a subject level.

---

**Summary**:  
By splitting data at the participant level, extracting RQA features per 100 ms window, and classifying windows per electrode, we can compute a time-resolved AUC curve that reveals *when* (in the time domain) each electrode best distinguishes DD from CT. Topographic maps at specific windows further highlight *where* on the scalp the most discriminative signals occur, reflecting potential neural adaptation differences between groups.
