# Recurrence Quantification Analysis (RQA)

Recurrence Quantification Analysis (RQA) is a nonlinear data analysis method used to quantify the patterns and dynamics of a system by studying its recurrences in phase space. In simpler terms, it examines when and how often the state of a system (like an EEG signal) returns to a similar state over time.

---

# RQE (Recurrence‐Quantification‐based Correlation Index)

This document provides an overview of the RQE algorithm, originally introduced in the context of **Recurrence Quantification Analysis (RQA)** of time series signals (e.g., EEG). 

> **RQE** stands for **Recurrence‐Quantification‐based Correlation Index**, and it measures how multiple RQA metrics (derived from a signal) correlate across rolling windows of time.

- **Stage 1**: For each window \(\mathcal{W}_{\mathrm{raw}}(i)\) of size `epoch_size` in the **raw signal**, build a recurrence plot and compute RQA metrics:
   \[
     \text{RQA}(i) 
       \;=\; \bigl(\mathrm{RR}(i),\; \mathrm{DET}(i),\; \mathrm{LAM}(i),\;\dots\bigr).
   \]
   This yields \(L\) RQA measures across \(\tilde{n}\) windows (\(\tilde{n} = n - \text{epoch\_size}+1\)).

- **Stage 2**: Take the RQA measure time series \(\bigl\{ \mathrm{RQA}(1), \mathrm{RQA}(2), \dots, \mathrm{RQA}(\tilde{n})\bigr\}\).  
   - Define a **big window** \(\mathcal{I}_{k,i}\) of length `big_window_size = k`.  
   - For each measure \(\ell\in\{1,\dots,L\}\), the sub‐series is
     \(\; \mathrm{RQA}^{(\ell)}[i : i + k - 1 ]\).  
   - Compute pairwise **Spearman** correlations \(\rho_{k,i}^{(\ell,m)}\) among these sub‐series.  
   - **RQE**:
     \[
       \mathrm{RQE}_{k,i} 
         \;=\; 
         \prod_{\ell \neq m} \Bigl( 1 \;+\; \bigl|\rho_{k,i}^{(\ell,m)}\bigr| \Bigr).
     \]


---

## 1. Motivation

In **RQA**, one often computes several features (e.g., Recurrence Rate, Determinism, Laminarity, Trapping Time) that describe the dynamics of a signal. The **RQE** index aggregates **pairwise correlations** of these RQA features over a specific segment (window) of time. By doing so, it provides a **single scalar** that reflects how these RQA measures co‐vary.

---

## 2. RQE Definition

Let \( S^{(\ell)} = \{ S_t^{(\ell)} \mid t = 1, \dots, n \} \) be the **\(\ell\)-th RQA time series**, for \(\ell = 1, 2, \dots, L\).  

We define a rolling window \(\mathcal{I}_{k,i}\) of **size** \(k\) (in the **RQA time series domain**, not necessarily the raw‐signal domain), **starting** at index \(i\). Formally:
\[
  \mathcal{I}_{k,i} = \{i, i+1, \ldots, i + k - 1\}.
\]
Within that window, for each pair \((\ell, m)\), we compute the **Spearman correlation** 
\[
   \rho_{k,i}^{(\ell,m)} 
   \;=\; \text{SpearmanCorr}\bigl( S_{k,i}^{(\ell)},\, S_{k,i}^{(m)} \bigr),
\]
where 
\[
  S_{k,i}^{(\ell)} 
  = \bigl\{ S_{t}^{(\ell)} \mid t \in \mathcal{I}_{k,i} \bigr\}
\]
is the sub‐series of measure \(\ell\) restricted to the indices \(\mathcal{I}_{k,i}\).

**RQE** is then defined as the product of \(\bigl(1 + \lvert \rho \rvert\bigr)\) over **all distinct measure pairs** \(\ell \neq m\). Symbolically:

\[
\boxed{
  \mathrm{RQE}_{k,i}
    \;=\;
    \prod_{\ell \neq m}
    \Bigl(
      1 \;+\; \bigl|\rho_{k,i}^{(\ell,m)}\bigr|
    \Bigr).
}
\]

- If you have \(L\) RQA measures, there are \(\binom{L}{2}\) distinct pairs \((\ell,m)\).  
- If \(L<2\), there are no pairs, so by convention \(\mathrm{RQE} = 1\).  
- If any correlation is \(\pm 1\), then \(\bigl(1 + |\rho|\bigr) = 2\). Hence \(\mathrm{RQE}\) is bounded in \(\bigl[1,\, 2^{\binom{L}{2}}\bigr]\).

---

## 3. Typical Two‐Stage RQA + RQE Workflow

1. **Compute RQA Measures**  
   From the raw signal, slide a small “epoch window” (e.g., 50 samples). For **each** small window in the raw signal:
   - Build a **Recurrence Plot** (e.g., with embedding dimension, radius, etc.).  
   - Extract RQA metrics (Recurrence Rate, Determinism, Laminarity, etc.).  
   - Each metric forms **one time series** across all these small windows.

2. **Compute RQE**  
   In the resulting **RQA time‐series domain**, define a larger rolling window of length \(k\). For each position, gather the sub‐segments of each RQA measure. Compute pairwise **Spearman** correlations among those sub‐segments, then multiply \(\left(1+|\rho|\right)\) over all pairs to get \(\mathrm{RQE}_{k,i}\).

This **two‐stage** approach ensures each RQA measure is not just a single scalar per big window, but a **sequence** of values that can be correlated meaningfully.

---

## 4. Explanation of Key Parameters

Below is a typical function signature for computing RQE from a single‐channel EEG signal, referencing the **paper’s Table 3** parameters:

```python
def main(
    embedding_dim: int = 10,    # "Embedding" dimension for RQA
    radius: float = 80.0,       # "Radius" threshold for recurrences
    min_line_length: int = 5,   # "Line": minimal line length for DET, LAM
    time_delay: int = 1,        # "Shift": time lag in phase-space
    epoch_size: int = 50,       # "Epoch": size of the small RQA window in the raw signal
    distance_metric: str = "euclidean",  # "Distance": type of distance
    big_window_size: int = 80,  # 'k': length of rolling window in the RQA domain
    simulate_length: int = 700  # total length of synthetic EEG
):
    ...
```

Here’s what each parameter means:

1. **`embedding_dim`** (\(\dim\))  
   - The **phase‐space embedding dimension** used in RQA. For example, \(\dim=10\) means each point is represented in a 10‐dimensional reconstructed state space.

2. **`radius`** (\(\epsilon\))  
   - The **threshold** for deciding if two embedded points are “recurrent” (distance \(\le \epsilon\)). Large `radius` \(\to\) more recurrences, possibly saturating the recurrence matrix. Very small `radius` \(\to\) few recurrences.  
   - In Table 3, “Radius=80” might need to be adapted if your signal amplitude is small or large.

3. **`min_line_length`**  
   - Minimum diagonal (or vertical) line length in the recurrence plot when computing **Determinism (DET)** or **Laminarity (LAM)**. In Table 3, “Line=5” means lines shorter than 5 points are not counted for DET or LAM.

4. **`time_delay`** (\(\tau\))  
   - The “lag” or “shift” in phase‐space reconstruction. In Table 3, “Shift=1.”  
   - If you embed with \(\dim=10\) and \(\tau=1\), each embedded vector is \((x_t, x_{t+1}, \dots, x_{t+9})\).

5. **`epoch_size`**  
   - Size of the **small rolling window** in the raw signal from which you compute each RQA measure. In Table 3, “Epoch=50,” so each RQA metric is derived from a 50‐sample segment of the raw signal.

6. **`distance_metric`**  
   - For the recurrence plot, you typically use “euclidean.” If the paper says “Meandist, Euclidean,” you might compute the mean distance in the window and multiply by a factor to get an adaptive `radius`.

7. **`big_window_size`** (\(k\))  
   - The length of the **rolling window** in the RQA‐measure domain for computing the RQE index. The RQA time series has length \(T = n - \text{epoch\_size} + 1\). If you choose `big_window_size=80`, that means each sub‐segment is 80 points in each RQA measure, from which you compute the Spearman correlations \(\rho\).

8. **`simulate_length`**  
   - Total length of your **synthetic** or real EEG data. If you want to replicate the exact number of epochs from the paper (e.g., “No. of epochs=642”), pick your signal length accordingly.

---

## 5. Common Pitfalls

1. **Threshold Too Large or Too Small**  
   - Leads to recurrence matrices that are all 1’s or all 0’s. Then all RQA measures become constant (e.g., RR\(\approx1\)), causing \(\mathrm{RQE} \to \mathrm{NaN}\) from undefined correlations.

2. **Insufficient Window Sizes**  
   - If `epoch_size` is too short for a high `embedding_dim`, you might have too few points in phase space.

3. **No Variation in RQA**  
   - If the RQA time series for each measure is constant over time, the Spearman correlation is undefined (NaN).

4. **Single RQA Measure**  
   - If \(L=1\), there are no pairs \((\ell,m)\). \(\mathrm{RQE}\equiv1\).

If you see \(\mathrm{RQE}=1\) or \(\mathrm{NaN}\), check that your threshold parameter, embedding dimension, or window sizes are appropriate for your signal’s amplitude and variability.
