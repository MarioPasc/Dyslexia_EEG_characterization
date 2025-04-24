# pyddeeg/classification/engine

How WindowParamEstimator works:

1. **Global counter**  
   *MNE’s* `SlidingEstimator` first clones the *base* estimator and then calls
   `.fit()` on each clone, one per window.  
   A module-level counter (`itertools.count()`) guarantees that the *first*
   call to `.fit()` receives `window_idx == 0`, the second `1`, … even though
   each call happens on an independent clone whose own `__init__` was unaware
   of the order.

2. **Window-specific hyper-parameters**  
   Inside `.fit()` the class:
   * grabs the parameter dict for *its* window,
   * removes the special key `"k"` (number of features),
   * builds a real `Pipeline` (`SelectKBest -> model(**params)`),
   * fits the pipeline and exports `.classes_` for compatibility.

3. **Delegation**  
   Prediction methods simply delegate to `self.fitted_` once the estimator is
   fitted.

Now, it can be called like:

```python
frozen_estimator = WindowParamEstimator(
    base_cls          = type(base_estimator),
    params_per_window = best_params,
)

results = evaluate_frozen_models(      # <-- new helper in trainer.py
    elec              = elec,
    dataset           = dataset,
    params_per_window = best_params,
    base_estimator_cls= type(base_estimator),
    n_jobs            = cfg.get("cv_jobs", -1),
)
np.savez_compressed(run_dir / "cv_results.npz", **results)
```
