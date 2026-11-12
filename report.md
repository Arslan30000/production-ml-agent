# Technical report: Production ML Agent

## System and permissions

The system uses DVC to track synthetic equipment data, MLflow to record training runs, Airflow 3 to orchestrate validation through a quality gate, and a FastAPI model service packaged for Kubernetes. The natural-language agent exposes structured operations across the four services. The architecture and each module's permissions are listed in [architecture.md](architecture.md). Training triggers and deployment restarts require separate CLI approval flags. Kubernetes operations are restricted to mlops-agent.

## Recorded results

| Check | Observed result |
| --- | --- |
| Dataset versions | data-v1 and data-v2 were committed and pushed as Git tags. The restored data-v1 CSV SHA-256 was ed16efff36a8916b235b446bd7b73a9f64bd385360ef57526c92703dc8b659c2; dvc status was current. |
| Version comparison | The DVC diff reported changed hashes for data/raw/equipment.csv, artifacts/model.json, and reports/metrics.json. |
| Baseline experiments | MLflow experiment 1 recorded three finished runs. The highest F1 was 0.800 for seed 44, run 3c6ecf539a474e9f9e628b1108d23a31. All three runs recorded DVC MD5 57c399e79e05a3e6e0fe747af2ad372d. |
| Normal workflow | Airflow run manual__2026-11-12T13:33:11.028062+00:00 completed success. The validation, preprocessing, training, MLflow evaluation, and quality gate tasks ran. |
| Injected validation failure | Run manual__2026-11-12T13:33:12.546233+00:00 completed failed. The validation task raised the injected error on three attempts; downstream tasks did not run. |
| Injected quality gate failure | Run manual__2026-11-12T13:33:12.769980+00:00 completed failed. Evaluation logged a model, then quality_gate raised the injected error on three attempts; its recorded F1 was 0.789. |
| Automated checks | 34 tests passed in WSL after installing Airflow and FastAPI. The final GitHub Actions run passed Ruff and pytest: 40 passed, 1 skipped, and 3 subtests passed. The skipped API test needs the DVC model artifact, which is kept in local remote storage. |
| Kubernetes deployment | Manifests and mocked diagnostics are present. A live cluster rollout, Pod status, and API response have not been supplied or verified. |
| LLM conversations | The local tool-calling loop and mocked four-service selection test run. Five real provider-backed conversations have not been captured because no API key was supplied in this project session. |

## Failure investigations

1. **Validation test:** The validation task deliberately raised an exception before preprocessing. Airflow retried twice, marked the run failed, and prevented later tasks. Resolution: trigger without inject_failure after confirming the dataset is current.
2. **Quality gate test:** The quality_gate task deliberately failed after MLflow evaluation. The F1 value of 0.789 exceeded the configured 0.750 minimum; the failure was injected, not caused by a low score. Resolution: trigger without inject_failure. Neither test constitutes evidence of model degradation.
3. **Tracking service interruption:** An early normal run could not contact MLflow on port 5000 and retried the evaluation task. Restarting the MLflow server allowed that run to complete successfully. This was a service availability issue, not a data or model quality failure.

## Demonstration scenarios

| Scenario | Natural-language prompt | Verified state |
| --- | --- | --- |
| Dataset history | "List tracked datasets, compare data-v1 and data-v2, and give the commands to reproduce data-v1." | The DVC diff and isolated data-v1 restoration were run locally. |
| Experiment selection | "Compare recent F1 runs, show the best candidate and its recorded dataset reference." | Three MLflow runs and their provenance were queried locally. |
| Workflow failure | "Find the failed validation and quality gate runs, inspect failed tasks, and explain what happened." | Two injected Airflow runs failed at their intended tasks; a normal run succeeded. |
| Deployment diagnosis | "Check the inference Deployment, Pod health, and deployed image; explain any restart or readiness issue." | Kubernetes tool responses are covered by mocked tests. No live cluster result was recorded. |
| Integrated decision | "Find the best recent model, identify its dataset version, check deployment health and Airflow status, and assess retraining." | The four-service tool selection and deterministic audit passed tests. A provider-backed conversation was not captured. |

These prompts can be submitted through `python -m agent.main 'PROMPT'` when the provider and services are available. The corresponding direct checks are:

```bash
python -m agent.tools.dvc_tools version
python -m agent.mlflow_cli best 1 --metric f1
python -m agent.airflow_cli runs
python -m agent.kubernetes_cli diagnose
python -m agent.main "Find the best recent model, identify its dataset version, check deployment health and Airflow status, and assess retraining"
python -m agent.main --audit --experiment 1
```

The audit reports known facts and uncertainties and never starts retraining. A high F1 score selects a candidate for review; it does not approve production use.

## Limitations and improvements

The DVC remote is a local folder and cannot be used from another computer without transferring its objects or configuring shared storage. The tracking and Airflow backends run locally. The image does not carry an MLflow run ID, so deployment health cannot establish exact run identity. A shared artifact store, immutable model image digest with run provenance, production authentication, and an automated promotion process would be needed for a production environment.

