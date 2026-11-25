# Artifact readiness and preservation audit

This audit records the local checkout after research and before dashboard work. The [baseline inventory](consolidation_baseline.json) records commit `483a8d10301933fd36a7b0c05cbe5199216b3642`, a clean initial tracked/untracked status, the ignored status and SHA-256 hashes for 405 pre-existing files. It covers source, tests, notebooks, historical reports, data, models, figures, result tables, protocols and checkpoints.

## Data and descriptive outputs

The cached raw CSV and source manifest are present. Their hashes agree, and the processed classifier export matches the SHA-256 in the original model-selection record. The cleaned all-flight export, classifier export and separate missing-departure-label export are present. The preparation report, data dictionary and monthly coverage are also present. [Raw source manifest](../data/raw/source.json) · [Model selection record](../outputs/models/prediction_selection.json) · [Data quality report](data_quality.md)

The standalone descriptive exports were missing at the consolidation baseline. Run `descriptive_recovery_001` regenerated them from isolated copies of the existing notebooks and the unchanged `src/patterns.py`. Dashboard consumers can use the tables, figures and model directly from the recovery directory. [Recovery directory](../outputs/experiments/descriptive_recovery/descriptive_recovery_001/) · [Artifact index with schemas and populations](../outputs/experiments/descriptive_recovery/descriptive_recovery_001/artifact_index.json)

| Descriptive exports | Recovery location |
| --- | --- |
| 18 EDA tables | The run’s `tables/` directory has `daily.csv`, the coverage and delay summaries, the weekday-hour table and all follow-up comparisons. |
| 12 EDA figures | The run’s `figures/` directory has the notebook’s EDA and follow-up charts. |
| 10 daily-pattern, cluster and anomaly tables | The run’s `tables/` directory has the summaries, assignments, cluster metrics, flags, sensitivity results, overlap and case studies. |
| 3 cluster and anomaly figures | The run’s `figures/` directory has cluster selection, the cluster timeline and past-only anomaly charts. |
| Descriptive model and settings | The run’s `models/` directory has `daily_clusters.joblib` and `daily_patterns_config.json`. |

The three notebook-generated report copies match their archived counterparts byte for byte. Validation confirmed 365 dates, 363 days eligible for clustering, clusters of 279 and 84 days, 29 delay flags and 21 missing-label flags. The recovery files are regenerated exports. Full-network daily clusters and flags retain their NYC-wide meaning when a dashboard filters to an airline or airport. Display unknown assignments and flags as unknown. Keep the full-year outcomes out of pre-departure classifier inputs. [Numerical validation](../outputs/experiments/descriptive_recovery/descriptive_recovery_001/validation.json) · [Completion and preservation record](../outputs/experiments/descriptive_recovery/descriptive_recovery_001/completion.json)

## Prediction artifacts

| Study | Saved row-level scores | Saved fitted objects |
| --- | --- | --- |
| Original four-model experiment | Yes. `outputs/tables/prediction_predictions.csv` covers validation and original historical assessment for all four models. | Four fitted pipelines and the selected logistic regression alias `outputs/models/flight_delay_pipeline.joblib`. Its cutoff, feature contract and environment versions are in `prediction_selection.json`. |
| Improvement study | Yes. `validation_predictions.csv` has September–October scores for seven shortlisted pipelines. Internal folds have aggregate metrics. | Seven shortlisted pipelines and one separate fitted calibration bundle. |
| Independent audit | Yes. `development_predictions.csv` has fold assessment scores. `tuning_*.csv` has per-fold tuning scores. Historical development diagnostics also have row-level files. | 28 models covering seven candidates and four folds. |
| Chronological calibration | Yes. `raw_period_scores.csv` and `assessment_predictions.csv`. | Intercept parameters saved as JSON. The run uses frozen audit bases, with their identities recorded in `base_integrity.csv`. |
| Chronological schedule grid | Assessment metrics and threshold-selection curves only. | Candidate checkpoints contain metrics and curves. |
| Departure and arrival replay | Historical metrics, cutoff-selection metrics and waiting-rule agreement counts only. | The run used in-memory models and saved no fitted pipelines. |

The 41 existing predictive Joblib files deserialized as fitted objects in the verified environment. The original saved model can support a schedule-only prediction view using its declared feature order and cutoff. The grid and replay support aggregate comparisons. Flight-level replay detail or inference would require separately available fitted models and a point-in-time operating data source. [Original models](../outputs/models/) · [Improvement outputs](../outputs/experiments/modeling_study/) · [Audit outputs](../outputs/experiments/independent_audit/stage1_audit_001/) · [Calibration outputs](../outputs/experiments/calibration_study/stage2_calibration_001/) · [Grid outputs](../outputs/experiments/chronological_grid_search/) · [Replay outputs](../outputs/experiments/flight_delay_replay/)

## Validation

- All 41 existing unit tests passed. The suite includes fixture checks and the original saved-model smoke test on two rows. The research validation itself used persisted scores and aggregate tables.
- Recomputed confusion and rank metrics from saved scores for 8 original assessment groups, 7 improvement validation groups, 68 audit assessment groups and 40 calibration assessment groups. Verified the original selection, the 14-configuration ranking against 42 internal fold rows, 56 audit cutoffs and 16 calibration cutoffs. The calibration rows reproduce equal raw/intercept ROC-AUC and AP. All ten saved equivalent-threshold comparisons record zero mismatches.
- Verified the grid’s 216 assessment rows against confusion counts and checkpoints, its 39-candidate ranking against equal-weight fold means, and its applied thresholds against the selection curves and frozen shortlist.
- Verified the replay’s 432 assessment rows and 123 cutoff-selection rows against confusion counts, frozen cutoffs and winner flags. Checked the labeled counts against the raw snapshot and reconciled the waiting-rule agreement counts. The grid and replay did not save row-level scores, so their ROC-AUC and AP are carried through from the archived computations. The replay’s cutoff optimality can be evaluated only to the extent of the recorded selection metrics.
- Verified that the [current results JSON](current_results.json) matches the [saved-table generator](../scripts/summarize_saved_results.py).

## Verified environment

The checks used CPython 3.13.13 on macOS 15.6.1 arm64. `python -m pip check` found no broken requirements. The original pins in [requirements.txt](../requirements.txt) are byte-identical to the baseline. [Experiment requirements](../requirements-experiments.txt) adds exact versions for direct dependencies and timezone data from the verified environment. Linux and Windows setup commands are documented as portable entry points. Verification was performed in the existing macOS environment.

| Packages from verified environment | Versions |
| --- | --- |
| pandas, scikit-learn, matplotlib | 2.3.1, 1.7.2, 3.10.7 |
| numpy, scipy, joblib, threadpoolctl | 2.5.5, 1.18.1, 1.6.0, 3.8.0 |
| notebook, ipykernel, IPython | 7.4.7, 7.1.0, 9.18.1 |
| nbformat, nbclient, tzdata | 5.11.1, 0.11.0, 2026.6 |

The [improvement environment record](../outputs/experiments/modeling_study/environment.json), archived protocols, audit reports and calibration reports retain their original environment information and historical paths.

## Preservation and intentional edits

The post-cleanup check found 404 byte-identical files out of the 405 in the baseline. The planned exception is `README.md`, rewritten as the project entry point. Its original bytes remain in the recorded Git commit. The before SHA-256 is `750370fbad32066213f53e666a5e184e237e3d4d8d2c23c462c426e8949e1479` and the after SHA-256 is `54fe7b24f7071d7837e92e9c355d11fba5025a1a718ac8e5aea6eedae68891df`.

The new files are the current findings, generated results JSON, saved-table generator, artifact audit, baseline inventory and experiment requirements. The only removals were `src/__pycache__/prepare.cpython-313.pyc`, `src/__pycache__/__init__.cpython-313.pyc` and their emptied cache directory. Personal IDE configuration was retained. Run locks, phase markers and other provenance files remain in place. Shared-export exclusions are documented in the README.

The improvement preservation manifest’s 17 paths still match. The older independent audit and calibration manifests already differed from this cleanup’s starting checkout for `README.md`, `notebooks/analysis.ipynb` and `notebooks/daily_patterns.ipynb`. The audit’s 7 recorded source hashes and the calibration study’s 5 source and 12 input hashes still match. The historical guards compare against their original snapshots. Their manifests and guards are preserved byte-for-byte, and the current baseline records the new documentation changes separately. [Improvement preservation record](../outputs/experiments/modeling_study/preservation_check.json) · [Independent audit protocol](../outputs/experiments/independent_audit/stage1_audit_001/protocol.json) · [Calibration protocol](../outputs/experiments/calibration_study/stage2_calibration_001/protocol.json)
