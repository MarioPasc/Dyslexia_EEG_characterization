# Report week 4/5

The main objective of this stage of the project is to make a metric selection among the 15 RQA metrics in order to, either:

- Train a Machine Learning model whose inputs are the selected features.
- Train a Machine Learning model whose input is the RQE correlation between the selected features.

In order to perform these test, we must remove features that are highly correlated, and give more importante to metrics that allow us to distinguish between our two groups.

---

| **Step** | **Goal**                                                     | **Key Questions**                                                                                                                                                                                                                                                           | **Statistical Models/Tools**                                                                                                                                | **Considerations for Robustness**                                                                                                                                                                          |
|----------|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Check Normality and Homoscedasticity** | Determine appropriate parametric or nonparametric tests                | - Does each metric follow a normal distribution (pooled or by group)?- Are variances similar across groups (CT vs. DD)?- If not, which nonparametric tests are appropriate?                                                                                   | - Shapiro-Wilk or Kolmogorov-Smirnov tests (normality)- Levene’s or Bartlett’s tests (homogeneity of variance)                                        | - EEG-derived metrics often violate normality assumptions- Smaller sample sizes (CT=34, DD=15) warrant careful choice of test                                                                          |
| **Examine Redundancy Among Metrics** | Identify highly correlated or redundant RQA metrics                    | - Which metrics are highly correlated or effectively interchangeable?- Do any metrics cluster together, indicating redundancy?- Do I need dimension reduction?                                                                                                     | - Spearman’s correlation matrix- PCA or Factor Analysis (optional)                                                                                     | - Use multiple-comparison corrections (FDR or Bonferroni) for significance of correlations- Theoretical importance may justify retaining correlated metrics                                             |
| **Identify Metrics That Discriminate Between Groups** | Determine which metrics differ most between CT and DD groups           | - Which metrics show significant differences across groups?- What effect sizes do we observe (practical vs. statistical significance)?- Should we treat channels/bands as repeated measures or analyze them separately?                                           | - Mann-Whitney U / Welch’s t-tests (for each metric)- Repeated-measures ANOVA or Friedman/Aligned Rank Transform (for multi-level data)- Effect sizes (Cohen’s d, Cliff’s delta) | - Apply multiple-comparisons correction- Factor in smaller sample sizes for DD group- Interpret p-values alongside effect sizes to gauge practical importance                                      |



## Objectives for this week

- [X] Execute full RQA metric computation for all channels and patients
- [ ] Build a RQA metric selection based off correlation analysis 

## Weekly Project Time Tracking

| End Date       | Task Description            | Hours Spent | Category/Type | Notes |
|------------|----------------------------|------------|--------------|-------|
| 2025-03-03 |       |         |  |  |
