# Session 1 — local time-series analysis

## Quick start

MATH840 final project: forecast the next 48 hours of electricity generation with 95% prediction intervals. The assignment is in [docs/MATH840_final_project.pdf](docs/MATH840_final_project.pdf).

| Path | Contents |
|---|---|
| `train.csv`, `weather_forecast.csv` | Supplied data |
| `NAZARENKO_project.csv` | Submission file, identical to `results_task/selected_model_forecast.csv` |
| `session1_*.py` | Experiments, one script per step |
| `notebooks/` | Numbered notebooks, one per step, in the order the work was done |
| `results*/` | Saved predictions, scores and forecasts |

Notebooks run from `notebooks/` or from the repository root; their first code cell sets the working directory.

| Notebook | Topic |
|---|---|
| `00_assignment_template` | Original assignment template |
| `01_eda` | Data checks and exploration |
| `02_ets_arima_sarima` | ETS, ARIMA and SARIMA baselines |
| `03_more_models` | Ridge, boosting, trees and weather sensitivity |
| `04_robust_models` | SARIMA variants, weather corrections, blends |
| `05_chronos_pretrained` | Chronos-Bolt pretrained models |
| `06_stacking` | Ensembles and chronological stacking |
| `07_timestral` | Timestral with future weather |
| `08_task_review` | Assignment metrics and interval methods |
| `09_baseline_win_rate` | Share of windows beating the benchmark |
| `10_extra_trees_snaive` | Extra Trees + SNaive |

Saved results are in the repository, so the notebooks open without refitting. To rerun everything, set up both environments (see below) and run the steps in this order, because later steps read the outputs of earlier ones:

```bash
.venv/bin/python session1_models.py
.venv/bin/python session1_more_models.py
.venv/bin/python session1_robust_models.py
.venv-hf/bin/python session1_hf_inference.py --download-only
HF_HUB_OFFLINE=1 .venv-hf/bin/python session1_hf_inference.py
.venv/bin/python session1_hf_evaluate.py
.venv/bin/python session1_stacking.py
.venv/bin/python session1_timestral_evaluate.py --prepare
.venv-hf/bin/python session1_timestral_inference.py --download-only
HF_HUB_OFFLINE=1 .venv-hf/bin/python session1_timestral_inference.py
.venv/bin/python session1_timestral_evaluate.py
.venv-hf/bin/python session1_task_inference.py --download-only
HF_HUB_OFFLINE=1 .venv-hf/bin/python session1_task_inference.py
.venv/bin/python session1_task_evaluate.py
.venv/bin/python session1_snaive_ensembles.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python session1_snaive_search.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python session1_extra_snaive.py
```

## SNaive ensembles

Open `notebooks/10_extra_trees_snaive.ipynb` for an executed Extra Trees + SNaive experiment, weight curves, window comparisons, prior-ensemble comparison, and the final forecast. Use the `Time Series (.venv)` kernel; Run All reruns evaluation with cached model predictions, while `REFIT=True` retrains the three Extra Trees configurations.

For Extra Trees specifically, run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python session1_extra_snaive.py`. This fits 120-tree weather models on 21/60/120-day histories using only available target lags, and tests SNaive weights 5–95% in 5% steps: 61 candidates including standalone models and baseline. It uses the common six initial calibration windows, eleven selection windows, latest audit, and origin-based weather-stress seeds. Point predictions are cached in `results_extra_snaive/`; remove the corresponding cache if changing the data or model configuration. Earlier Extra Trees results in `results_more/` used different windows and were not part of the 66,845-combination search.

The best Extra Trees blend is 60% Extra Trees (60-day training) + 40% SNaive, with selection RMSE 271.86 versus SNaive 318.14. It does not beat the prior Timestral/Ridge/SNaive blend (232.55). Scores, window wins, all candidate forecasts, and `selected_model_forecast.csv` are in `results_extra_snaive/`. Intervals use expanding earlier errors; the latest audit does not select weights. Validation reuse and hypothetical weather errors remain limitations. Codex implemented, ran, and verified this experiment.

For the large search, run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python session1_snaive_search.py`. It loads 50 unique point models from the common twelve saved rolling windows and tests 66,845 candidates: single-model controls, every SNaive/model pair at 5% weight increments, separate day-one/day-two pair weights at 10% increments, every SNaive/two-model triple on a 10% simplex grid, and 20,000 seeded random blends with 3, 4, 5, or 8 members. Random blends allocate at least 5% to SNaive. This is a broad finite search, not every possible continuous weighting. No model refits or downloads are needed.

`results_snaive_search/` contains the top 100, best per family, point RMSE/MAPE details, per-window results for the winner and references, and chronological grid-selection checks. The full ranked table (`all_combinations.csv`) and loss arrays (`search_arrays.npz`) are not stored because of their size; rerunning the script recreates them without refitting any model. The latter choose weights using only earlier completed windows, starting at fold 7, so their ten selection windows differ from the eleven used for the main ranking. All candidates share the same weather scenarios; the latest audit is excluded from selection. Duplicate point models are removed and scoring is checked against direct hourly arithmetic.

The large-search winner is 10% SNaive + 20% hour-weather Ridge (60-day training) + 70% Timestral-28M with weather: selection RMSE 232.55, day-two RMSE 249.17, and 10/11 wins against SNaive. Latest audit RMSE is 179.85 versus SNaive's 120.40. The selected 48-hour forecast is `results_snaive_search/selected_model_forecast.csv`. Final intervals use all completed historical errors; this search optimizes point RMSE, not Winkler, and does not independently evaluate those intervals. Searching tens of thousands of combinations on reused validation windows increases selection optimism. Codex implemented and ran this experiment.

Run `.venv/bin/python session1_snaive_ensembles.py` after the Timestral experiments. This reuses saved forward predictions to compare 13 new ensembles: 10/25/50/75% daily SNaive mixed with Timestral-28M or the previous five-model mean; a three-model mean and median; and expanding, recent-four-window, or daywise convex weights over SNaive, Timestral, and the previous mean. Learned weights use only completed earlier folds. Intervals use expanding errors by forecast day (from fold 3 for learned ensembles); SNaive retains its official intervals.

Results are in `results_snaive/`, including full-horizon/day-two scores, window comparisons, learned weights, and 48-row forecasts. Selection uses weather-stress RMSE on the same eleven selection windows and excludes the latest audit. The selected 90% Timestral + 10% SNaive blend has RMSE 238.40 versus Timestral 240.79 and SNaive 318.14, winning 9/11 windows against SNaive. Its latest audit RMSE improves from Timestral's 180.14 to 159.55 but remains above SNaive's 120.40. Selection day-two RMSE slightly worsens versus Timestral (261.73 vs 260.66). Repeated validation reuse and hypothetical weather stress limit conclusions about hidden performance. Codex implemented, ran, and checked this comparison; the selected forecast is `results_snaive/selected_model_forecast.csv`.

From this directory:

```bash
source .venv/bin/activate
python -m jupyter lab
```

Open `notebooks/01_eda.ipynb` and select `Time Series (.venv)` (in VS Code: Select Kernel → Python Environments → `.venv/bin/python`). Run the cells in order. The EDA notebook does not require your surname or produce a submission.

`notebooks/00_assignment_template.ipynb` is the original assignment notebook. Set `SURNAME` in its setup cell before running it to generate the benchmark submission. Upload your submission yourself according to the assignment instructions.

To recreate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m ipykernel install --sys-prefix --name time-serias --display-name "Time Series (.venv)"
```

`requirements-lock.txt` records the installed versions once setup is complete. All CSV paths are relative to this folder. Use chronological validation; the final 48 training hours are reserved for validation in the EDA. Historical weather is observed, whereas future weather is forecast, so weather-based validation may be optimistic.

LLM use: Codex helped set up the local Python environment and create the initial EDA notebook, including data checks, plots, and the supplied seasonal-naive validation benchmark. Only the provided assignment data was used.

## ETS, ARIMA, and seasonal ARIMA comparison

Open `notebooks/02_ets_arima_sarima.ipynb` for baseline explanations, comparison tables, diagnostics, and the next 48-hour forecast. It reads saved experiment results so opening it does not refit every model.

To reproduce all fits and forecasts:

```bash
source .venv/bin/activate
python session1_models.py
```

To also write the required Moodle filename, substitute your surname:

```bash
python session1_models.py --surname YOUR_SURNAME
```

The script compares seven ETS/ARIMA/SARIMA configurations and daily seasonal naive. Six historical 48-hour blocks calibrate candidate intervals, five later blocks select the lowest pooled RMSE, and the last block is reported separately. The last block was previously inspected during EDA and an initial trial, so it is not an untouched test set. Each fit uses only data before its forecast origin; recent training-window sizes are part of the candidate configurations. The baseline remains eligible if no candidate improves selection RMSE.

Outputs in `results/` include per-window scores, hourly validation predictions, the comparison table, optimizer diagnostics, and `selected_model_forecast.csv` (48 rows in assignment format). Candidate intervals use earlier rolling forecast errors, separately by forecast day; this empirical method does not guarantee future 95% coverage. Weather is not used in these models. Hidden future performance cannot be measured locally.

Codex also implemented and ran this model comparison and helped document its interpretation. Review the notebook and model choices so you can explain them in Session 2.

## Additional models and weather sensitivity

Run `python session1_more_models.py`, then open `notebooks/03_more_models.ipynb`. This compares seasonal means, calendar Ridge, weather Ridge, histogram boosting, Extra Trees, Random Forest, the previous SARIMA, and seasonal naive. Target lags are at least 48 hours, so they are available for the entire forecast horizon. Training uses recent windows and no future targets.

`results_more/` contains separate results and a new candidate forecast; the previous SARIMA outputs remain in `results/`. The notebook's final cell can save the new candidate using your surname. Weather models are evaluated with observed weather and three illustrative perturbed-weather scenarios. The latter are sensitivity assumptions, not measured weather forecast errors. The lowest pooled stress-scenario selection RMSE chooses the candidate; MAPE and interval scores may favor another model. Reusing selection windows can introduce model-selection optimism. The historical audit is not an untouched test set.

Codex implemented and ran these additional comparisons and weather sensitivity checks.

The additional-model notebook now includes dated window-by-window RMSE tables, baseline ratios, a heatmap, and a comparison plot. Change `SCENARIO` to compare observed weather or weather stress. `results_more/window_comparison.csv` contains all metrics by window; `window_consistency.csv` reports selection-window wins, ranks, and best/worst ratios. Audit results are shown separately and excluded from win counts. These CSVs are regenerated by `session1_more_models.py`; chart PNGs are saved when running the notebook.

## Broader search: SARIMA variants, weather corrections, and blends

Run `python session1_robust_models.py`, then open `notebooks/04_robust_models.ipynb`. This tests 21 candidates, including seasonal-MA SARIMA variants, hour-specific weather Ridge, weather corrections anchored to the last day, residual boosting, and fixed-weight ensembles. It uses six calibration windows, eleven selection windows, and the previously seen latest audit. The lowest pooled stress-scenario 48-hour RMSE selects the candidate; the baseline is eligible.

The larger selection period adds six earlier windows but does not turn previously examined data into an untouched test set. Weather perturbations are illustrative, and this run ties their seeds to forecast origins. Do not directly attribute differences between runs solely to model changes: validation periods and stress draws differ. Every candidate within this run receives the same windows and scenarios.

`results_robust/` contains model/window comparisons, day-two scores, fit diagnostics, point forecasts by model, and `selected_model_forecast.csv` with prediction intervals. Day-two historical results are reported because the final private leaderboard uses the second day. Set `SURNAME` in the notebook's last cell to save the required submission filename. All older forecasts remain available. No hidden targets or external data are accessed.

Codex implemented and ran this expanded search and its leakage/format checks. The models and interval methods should be reviewed and understood before presenting them.

Expanded-run finding: the 50/50 SARIMA/hour-weather Ridge blend had stress-scenario pooled RMSE 277.10 versus baseline 318.14 across eleven selection windows (12.9% reduction), winning eight windows. Day-two RMSE was 290.91 versus 349.50. The latest historical audit favored the baseline: 120.40 versus the blend's 162.47. Calibration-only validation coverage for the blend was 88.8%, below nominal 95%; the future forecast refreshes interval calibration using all completed historical windows, but that refreshed interval has no independent coverage test. The candidate is evidence of average improvement, not a guarantee of hidden-leaderboard success.

## Hugging Face: local Chronos-Bolt experiments

`session1_hf_inference.py` runs frozen [Chronos-Bolt Tiny and Small](https://huggingface.co/amazon/chronos-bolt-small) on CPU using only local generation history, without fine-tuning or weather inputs. It tests 168, 504, and 1536 hours of context. The native median is the point forecast; nominal 95% intervals are calibrated from historical rolling errors, not inferred from native 10–90% quantiles. Model revisions are recorded in `results_hf/model_versions.json`. Pretraining-data overlap with this disguised series cannot be independently verified; no source-identification search is performed.

Chronos requires newer Python, so it uses a separate `.venv-hf` (Python 3.12), while evaluation and notebooks use the original `.venv`:

```bash
python3.12 -m venv .venv-hf
.venv-hf/bin/python -m pip install -r requirements-hf-lock.txt
.venv-hf/bin/python session1_hf_inference.py --download-only
HF_HUB_OFFLINE=1 .venv-hf/bin/python session1_hf_inference.py
.venv/bin/python session1_hf_evaluate.py
```

The download step retrieves public weights/configuration into `.cache/huggingface`; prediction runs locally and can run offline afterward. It does not send the project CSV data to an inference service. The recorded model revisions document the evaluated artifacts; the download step reuses recorded revisions when the manifest exists.

Open `notebooks/05_chronos_pretrained.ipynb` with the original `Time Series (.venv)` kernel to review results. `results_hf/` includes individual model forecasts, `best_hf_forecast.csv`, and `selected_model_forecast.csv` (best among previous candidates and HF candidates, using the same eleven selection windows). The latest audit is excluded from selection. The notebook can save the selected file under your surname. More models on reused windows can increase selection optimism; model size and pretraining do not guarantee hidden-leaderboard improvement.

Codex set up this isolated environment, implemented the inference/evaluation scripts, and ran the local pretrained-model comparisons.

HF-run finding: Chronos-Bolt Small with 504 hours of history achieved pooled selection RMSE 307.95; its 50/50 SARIMA blend achieved 298.69, versus baseline 318.14. The prior SARIMA/hour-weather blend remains selected at 277.10. The best HF blend's latest audit RMSE was 144.98 versus baseline 120.40. Larger pretrained models did not automatically improve the chosen metric; the run records these unsuccessful and partially successful alternatives for the report.

## Percentage of windows beating baseline

Open `notebooks/09_baseline_win_rate.ipynb` with the original kernel. It calculates `100 × winning windows / total windows`, where a win means strictly lower RMSE than seasonal naive (numerical ties are reported separately). It compares the 29 candidates with available results on the same eleven selection windows; older ETS/ARIMA runs on five windows are excluded from this ranking. The baseline ties itself and is omitted from the win-rate chart. The latest audit is separate. Change `SCENARIO` to compare observed weather and weather stress.

The notebook includes a percentage bar chart, a window win/loss matrix, pooled RMSE reduction, and audit results. It exports tables and PNGs to `results_win_rate/`. A win rate is a historical frequency, not a probability of winning the hidden leaderboard; win counts do not reflect the size of each win or loss.

## Ensembles and chronological stacking

Run `.venv/bin/python session1_stacking.py`, then open `notebooks/06_stacking.ipynb`. This uses saved broad-search/HF artifacts and reconstructs the six initial base-forecast windows. It compares equal mean, median, expanding convex weights, recent-four-window convex weights, Ridge stacking, separate day-one/day-two weights, and weights shrunk toward the prior fixed blend.

Meta models learn only from earlier out-of-fold forecasts and targets. At each new window they predict first, then update after its actuals become available. The first three windows initialize stacking; forward errors from window 3 onward calibrate expanding prediction intervals. The eleven selection windows and separate audit are unchanged. Earlier models retain their previous interval methods, so interval-score comparisons also reflect calibration-policy differences. Prior feature/model selection still introduces selection optimism.

`results_stacking/` includes comparisons, per-window weights, candidate forecasts, and the overall selected forecast. The simple equal mean of SARIMA, hour-weather Ridge, weather Ridge, weather correction, and Chronos-Bolt Small achieved selection RMSE 265.86 versus baseline 318.14 (16.4% reduction), winning 9/11 windows (81.8%). It lost on the latest audit: 172.21 versus baseline 120.40. Learned stacking did not beat this simple average. All past forecasts remain available.

`notebooks/09_baseline_win_rate.ipynb` now uses the latest stacking comparison when available, including all 36 candidates evaluated on the same eleven windows. Codex implemented, ran, and checked the ensemble experiments and documented the limitations.

## Timestral: pretrained forecasting with future weather

`notebooks/07_timestral.ipynb` compares Mistral Timestral 5M and 28M, each with 504 hours of history, with and without seven weather covariates. It also tests a fixed 50/50 blend of Timestral-28M with weather and the previous equal-weight ensemble. Larger Timestral sizes are not tested. The same eleven selection windows and illustrative weather-stress draws are used; the latest previously seen audit remains separate.

Reproduce after generating the prior robust/HF/stacking artifacts:

```bash
.venv-hf/bin/python -m pip install -r requirements-hf-lock.txt
.venv/bin/python session1_timestral_evaluate.py --prepare
.venv-hf/bin/python session1_timestral_inference.py --download-only
HF_HUB_OFFLINE=1 .venv-hf/bin/python session1_timestral_inference.py
.venv/bin/python session1_timestral_evaluate.py
```

Package commit and model revisions are pinned/recorded. Downloads only fetch public model artifacts; project inputs are processed locally. Future target values are excluded from the history and covariate arrays. Weather validation is optimistic with observed weather; stress scenarios are hypothetical. Nominal 95% intervals use earlier forecast errors, not mislabeled native 5th/95th quantiles. Prior ensemble intervals retain their earlier calibration policy.

Outputs are in `results_timestral/`, including the best Timestral candidate and the overall selected forecast. Codex implemented and ran the comparison. Reused validation data and uncertain pretraining overlap limit conclusions about hidden-leaderboard performance.

Timestral-run finding: Timestral-28M with weather achieved weather-stress selection RMSE 240.79 versus baseline 318.14 (24.3% reduction), winning 9/11 windows. It improves on the earlier equal-weight ensemble's 265.86. However, the latest audit RMSE was 180.14 versus baseline 120.40, and fixed-calibration validation interval coverage was 90.3%, below nominal 95%. The selected candidate is saved in `results_timestral/selected_model_forecast.csv`; future intervals refresh calibration using all completed historical windows without an independent test of that refreshed coverage. `notebooks/09_baseline_win_rate.ipynb` now includes all 41 candidates from the common eleven-window evaluation and prefers the latest Timestral results when available.

## Assignment-focused review: point accuracy and intervals

Open `notebooks/08_task_review.ipynb`. This explicitly separates Session 1's full-48-hour RMSE objective from the final private day-two rank combination (30% RMSE, 30% MAPE, 40% Winkler). It tests shorter Timestral context, an hour-of-day covariate, Timestral-123M, causal half-strength recent level/bias adjustments, and SARIMA blends. It also compares expanding absolute, scaled, and daylight-specific asymmetric empirical intervals.

Reproduce after the Timestral and previous experiment prerequisites:

```bash
.venv-hf/bin/python session1_task_inference.py --download-only
HF_HUB_OFFLINE=1 .venv-hf/bin/python session1_task_inference.py
.venv/bin/python session1_task_evaluate.py
```

Outputs go to `results_task/`. Session 1 selection minimizes full-48-hour weather-stress RMSE; the interval for that point model minimizes selection Winkler. Latest audit results do not select the model. Adaptation uses only the preceding completed block. The validation data are reused, interval methods are also selected on validation, and hypothetical weather errors may not represent actual forecast errors. These results cannot guarantee hidden-leaderboard success. Codex implemented, ran, and checked this assignment-focused comparison.

Task-review finding: the extra point-model variants did not beat the existing Timestral-28M weather forecast. It remains selected at full-48-hour selection RMSE 240.79 versus baseline 318.14, with 9/11 window wins. With causal expanding absolute-error intervals, selection Winkler is 1124.68 versus baseline 2258.12 and coverage is 92.8%. On selection day-two hours, RMSE is 260.66 vs 349.50, MAPE is 168.13% vs 294.00%, and Winkler is 1142.90 vs 2367.15. These are historical metrics, not classroom ranks or hidden scores. The latest audit remains worse than baseline (full-48-hour RMSE 180.14 vs 120.40). The new task-review CSV has the same selected point forecast as before, with only negligible numerical differences in final interval calculation. The additional tests confirm the prior point-model choice rather than establish a newly improved forecast.
