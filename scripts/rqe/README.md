# Core Idea: Recurrence Plots

- **Recurrence Matrix (R)**:  
  A binary \(N \times N\) matrix where each entry \(R_{ij}\) is 1 if the distance between state vectors \(\mathbf{x}_i\) and \(\mathbf{x}_j\) is below some threshold, and 0 otherwise. It visually and quantitatively reveals when and how often states of the system recur in phase space.

- **Embedding**:  
  For a *scalar* time series, you can reconstruct its “phase space” by specifying embedding dimension (\(\dim\)) and delay (\(\tau\)). This transforms a 1D series into multi-dimensional vectors. Recurrence analysis often requires embedding to capture the system’s underlying dynamics.

- **Distance Metrics**:  
  Supported metrics include `"manhattan"`, `"euclidean"`, and `"supremum"` norms (plus a derived `"meandist"` approach for adjusting the threshold by the mean pairwise distance). These metrics define how “closeness” is measured in phase space.

- **Threshold / Recurrence Rate**:  
  Instead of using a fixed threshold, you can fix the recurrence rate or local recurrence rate. The library can automatically compute the threshold that yields a specific fraction of recurrences.

---

## Recurrence Quantification Analysis (RQA) Measures

1. **Recurrence Rate \(\mathrm{RR}\)**  
   - *Definition*: Fraction (or percentage) of points that are recurrent (i.e., how dense the black points are in the recurrence matrix).  
   - *Interpretation/Use*: Gives a global measure of how often the system revisits similar states. Often used as a baseline RQA measure.

2. **Determinism \(\mathrm{DET}\)**  
   - *Definition*: The fraction of recurrence points forming diagonal lines of at least length \(l_{\min}\).  
   - *Interpretation/Use*: Indicates the predictability or deterministic structure in the system’s dynamics. Highly deterministic systems often have longer diagonal structures.

3. **Maximum Diagonal Line Length \(\mathrm{L_{\max}}\)**  
   - *Definition*: The length of the longest diagonal line segment in the recurrence plot.  
   - *Interpretation/Use*: Connected to how long the system remains in a particular “flow” before diverging; often associated with Lyapunov exponents.

4. **Average (Mean) Diagonal Line Length \(\mathrm{L_{mean}}\)**  
   - *Definition*: The mean length of diagonal lines of at least length \(l_{\min}\).  
   - *Interpretation/Use*: Another measure of how long trajectories stay close, relating to predictability and correlation time.

5. **Diagonal Line Entropy \(\mathrm{ENT}\)**  
   - *Definition*: The Shannon entropy of the distribution of diagonal line lengths.  
   - *Interpretation/Use*: Reflects the complexity or richness of the diagonal structures. Higher entropy means a wider variety of diagonal lengths.

6. **Laminarity \(\mathrm{LAM}\)**  
   - *Definition*: Fraction of recurrence points forming vertical lines of at least length \(v_{\min}\).  
   - *Interpretation/Use*: Measures how often the system gets “stuck” in particular states or “laminar” phases. High laminarity often indicates intermittency or laminar flow.

7. **Trapping Time \(\mathrm{TT}\)**  
   - *Definition*: The average length of vertical lines (of at least \(v_{\min}\)); often the same as `average_vertlength()`.  
   - *Interpretation/Use*: Indicates how long the system remains in certain states (sometimes viewed as the mean residence time).

8. **Maximum Vertical Line Length \(\mathrm{V_{\max}}\)**  
   - *Definition*: The length of the longest vertical line segment.  
   - *Interpretation/Use*: Like \(L_{\max}\), but captures vertical structures (states that don’t change quickly or remain close in time).

9. **Average Vertical Line Length \(\mathrm{V_{mean}}\)**  
   - *Definition*: The average length of vertical lines of at least \(v_{\min}\).  
   - *Interpretation/Use*: Another measure of how persistently the system hovers near certain states in consecutive time steps.

10. **Vertical Line Entropy \(\mathrm{V_{ENT}}\)**  
    - *Definition*: The entropy of the distribution of vertical line lengths.  
    - *Interpretation/Use*: Reflects the complexity in how the system hovers in certain states for variable durations.

11. **Mean (Average) White Vertical Line Length \(\mathrm{W_{mean}}\)**  
    - *Definition*: Mean length of “white vertical lines” (gaps in the recurrence matrix) of at least \(w_{\min}\). Also called “mean recurrence time.”  
    - *Interpretation/Use*: Tells you, on average, how long it takes to return to a previously visited state.

12. **Maximum White Vertical Line Length \(\mathrm{W_{\max}}\)**  
    - *Definition*: The length of the longest white vertical gap in the recurrence matrix.  
    - *Interpretation/Use*: The maximum observed recurrence time—the system’s slowest return to an earlier state.

13. **White Vertical Line Entropy \(\mathrm{W_{ENT}}\)**  
    - *Definition*: Entropy of the distribution of white vertical line lengths.  
    - *Interpretation/Use*: Indicates complexity in the distribution of return times.

14. **Complexity Entropy \(\mathrm{CLEAR}\)**  
    - *Definition*: A measure from [Ribeiro2011] evaluating how “complex” the embedded dynamics are.  
    - *Interpretation/Use*: Provides an alternative view of the system’s entropy that factors in the system’s underlying structure.

15. **Permutation Entropy \(\mathrm{PERM\_ENT}\)**  
    - *Definition*: Entropy measure derived by comparing the order patterns (permutations) in the embedded time series [Bandt2002].  
    - *Interpretation/Use*: Often used for nonlinear time‐series analysis; robust measure of complexity based on ordinal patterns.

16. **Recurrence Probability**  
    - *Definition*: Probability that the trajectory is recurrent after some time lag. Typically `lag=0` checks immediate recurrences.  
    - *Interpretation/Use*: Variation of the recurrence rate that can be examined across different lags to see how recurrences evolve over time.

17. **Frequency Distributions**  
    - **Diagonal line distribution (`diagline_dist`)**: A histogram of how many diagonal lines exist for each length.  
    - **Vertical line distribution (`vertline_dist`)**: Likewise, how many vertical lines exist for each length.  
    - **White vertical line distribution (`white_vertline_dist`)**: Same idea, but for the gaps in the recurrence matrix.  
    - *Interpretation/Use*: These distributions are used to compute the RQA measures (like DET, ENT, etc.) and to see the range of line lengths in more detail.

18. **Resampling Methods**  
    - **`resample_diagline_dist(M)`** and **`resample_vertline_dist(M)`**: Monte Carlo–type resampling to estimate confidence intervals on diagonal/vertical distributions, useful for statistical significance testing.  
    - *Interpretation/Use*: Helps you quantify the statistical reliability of your RQA measures (e.g., how they vary if you bootstrap the line distributions).

19. **`rqa_summary(l_min, v_min)`**  
    - *Definition*: A quick dictionary of commonly cited RQA measures: \(\mathrm{RR}\), \(\mathrm{DET}\), \(\mathrm{L_{mean}}\), \(\mathrm{LAM}\).  
    - *Interpretation/Use*: Handy “snapshot” to see the main metrics for a given time series.

---

## Configuration / Helper Methods

- **`set_fixed_threshold(threshold)`** / **`set_fixed_threshold_std(threshold_std)`**:  
  Set the threshold directly (in absolute terms or in units of the time series’ standard deviation).

- **`set_fixed_recurrence_rate(rr)`** / **`set_fixed_local_recurrence_rate(rr_local)`**:  
  Instruct the class to find a threshold that yields a *global* or *local* fraction of recurrences.

- **`legendre_coordinates`**, **`embed_time_series`**, etc.:  
  Methods for embedding or reconstructing the phase space from a scalar series (e.g., time-delay embedding or advanced expansions).

- **Distance Matrices**:  
  - `euclidean_distance_matrix()`, `manhattan_distance_matrix()`, `supremum_distance_matrix()` – convenience methods to compute distances in phase space directly.  
  - Possibly used if you want to further customize threshold selection or create specialized analyses.

- **Surrogates** (`twin_surrogates`, etc.):  
  Generate “twin” or surrogate time series that preserve certain properties of the original. Useful for hypothesis testing of nonlinearity.

---

### Takeaway

In summary, **RecurrencePlot** gives you:

1. **Recurrence Matrix Construction**  
   (choose embedding dimension, distance metric, threshold or recurrence rate).

2. **RQA Metrics**  
   (like Recurrence Rate, Determinism, Laminarity, line lengths, entropies).

3. **Distributions**  
   (diagonal/vertical line length histograms, white vertical lines).

4. **Confidence/Statistical Tools**  
   (resampling of line distributions, surrogate data methods).

5. **Embedding**  
   Tools to reconstruct multi-dimensional phase‐space from one‐dimensional signals.

All these tools are used to quantify and interpret the dynamical structure, predictability, and complexity of time‐series data—from biological signals (EEG, heart rates) to any system with nonlinear, potentially chaotic behavior.
