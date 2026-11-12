# Production ML Agent

An MLOps agent for checking dataset provenance, comparing experiments, running an approved training workflow, and diagnosing a Kubernetes inference service. Operational answers include tool results. Training and deployment restarts require explicit CLI approval.

The equipment readings are synthetic demonstration data. The inference service is a baseline example, not a real equipment safety system.

## Project layout

- training/ generates and trains the deterministic logistic baseline.
- agent/tools/ contains fixed DVC, MLflow, Airflow, and Kubernetes operations.
- agent/main.py runs natural-language tool calling or a deterministic four-service audit.
- dags/training_pipeline.py defines the Airflow 3 training DAG.
- api/main.py serves the model; Dockerfile and kubernetes/ define its deployment.
- tests/ contains unit and mocked integration checks; .github/workflows/ci.yml runs lint and tests on pull requests.

## Install in WSL

Run from the existing repository at ~/MLOPS/A3:

~~~bash
cd ~/MLOPS/A3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,airflow,api]'
python -m pytest -q
python -m ruff check --select E4,E7,E9,F agent api dags scripts tests training
~~~

Python 3.11 or newer is required. The normal CI job does not start external services; service responses are mocked in its tests.

## Dataset and DVC

The generated CSV is versioned by DVC and excluded from Git. A local DVC remote lives beside this checkout, at ../production-ml-agent-dvc-storage. A fresh computer needs a copy of that storage or a separately configured shared remote before dvc pull can restore the data.

~~~bash
dvc pull
dvc status
python -m agent.tools.dvc_tools list
python -m agent.tools.dvc_tools version
python -m agent.tools.dvc_tools compare data-v1 data-v2
python -m agent.tools.dvc_tools reproduce data-v1
~~~

The data-v1 and data-v2 tags point to two committed DVC dataset versions. The reproduction tool prints a plan for an isolated Git worktree; it does not overwrite the active checkout. The original data-v1 checkout was restored and its SHA-256 matched the baseline.

## MLflow

Keep this command running in its own WSL terminal:

~~~bash
cd ~/MLOPS/A3
source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
~~~

In another terminal:

~~~bash
cd ~/MLOPS/A3
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m agent.mlflow_cli experiments
python -m agent.mlflow_cli runs 1
python -m agent.mlflow_cli best 1 --metric f1
python -m agent.mlflow_cli details 3c6ecf539a474e9f9e628b1108d23a31
~~~

The three baseline MLflow runs use seeds 42, 43, and 44. Their metrics, model artifacts, dataset SHA-256, Git commit, and DVC hash were logged to the local tracking server. The best baseline F1 was 0.800 for seed 44; a metric ranking does not establish production readiness.

## Airflow

Start Airflow separately while MLflow remains available:

~~~bash
cd ~/MLOPS/A3
source .venv/bin/activate
export MLOPS_PROJECT_ROOT="$PWD"
export AIRFLOW_HOME="$PWD/airflow-home"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
airflow standalone
~~~

Airflow 3 creates an admin password in airflow-home/simple_auth_manager_passwords.json.generated. Do not commit that file or paste its contents into a report. In a third terminal, obtain a short-lived token and inspect the DAG:

~~~bash
cd ~/MLOPS/A3
source .venv/bin/activate
export AIRFLOW_HOME="$PWD/airflow-home"
export AIRFLOW_API_URL=http://127.0.0.1:8080
export AIRFLOW_API_TOKEN="$(python scripts/airflow_token.py)"
airflow dags unpause equipment_training -y
python -m agent.airflow_cli runs
~~~

The token helper prompts for username admin and the generated password. Re-run it in a new terminal or when the token expires. The DAG stages are validation, preprocessing, training, MLflow evaluation, and a quality gate. The task retry limit is two. To inspect a run, replace the value below with an actual dag_run_id:

~~~bash
RUN_ID='manual__2026-11-12T13:53:20.209069+00:00'
python -m agent.airflow_cli status "$RUN_ID"
python -m agent.airflow_cli tasks "$RUN_ID"
python -m agent.airflow_cli failed "$RUN_ID"
python -m agent.airflow_cli log "$RUN_ID" quality_gate
~~~

A real normal run completed successfully. Injected validation and quality gate runs failed at their intended tasks. The first normal run was delayed because MLflow had stopped; it completed after MLflow restarted. The injected quality gate run scored F1 0.789 but failed by design. These failures are tests, not model degradation evidence.

## Docker and Kubernetes

The image includes the DVC-restored model JSON. With Docker, kubectl, and a local Minikube cluster available:

~~~bash
cd ~/MLOPS/A3
dvc pull
docker build -t production-ml-agent:local .
minikube start --driver=docker
minikube image load production-ml-agent:local
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent rollout status deployment/equipment-api
kubectl -n mlops-agent get pods,deployments,services
python -m agent.kubernetes_cli version
python -m agent.kubernetes_cli diagnose
~~~

The Deployment requests two replicas, readiness and liveness probes, and CPU/memory requests and limits. The ClusterIP Service is reachable locally through a port forward:

~~~bash
kubectl -n mlops-agent port-forward svc/equipment-api 8000:80
~~~

In another terminal:

~~~bash
curl -fsS http://127.0.0.1:8000/health/ready
curl -fsS http://127.0.0.1:8000/version
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"temperature_c":80,"vibration_mm_s":4,"load_pct":65}'
~~~

The Kubernetes tools are restricted to namespace mlops-agent. Only the equipment-api Deployment can be restarted, and that action requires explicit approval. Kubernetes deployment output has not yet been supplied for this checkout, so live Pod health is not claimed in the report.

## Conversational agent

The natural-language interface uses an OpenAI-compatible Chat Completions endpoint and local structured tools. Set the provider's key in the shell; never save it in Git. The default endpoint is Groq and the default model is openai/gpt-oss-20b. Set OPENAI_BASE_URL and OPENAI_MODEL to use a different compatible provider.

~~~bash
read -rsp 'API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
export OPENAI_BASE_URL=https://api.groq.com/openai/v1
export OPENAI_MODEL=openai/gpt-oss-20b
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export AIRFLOW_API_URL=http://127.0.0.1:8080
export AIRFLOW_API_TOKEN="$(python scripts/airflow_token.py)"
python -m agent.main "Compare recent F1 runs and identify their dataset version"
python -m agent.main "Find the best recent model, identify its dataset version, check deployment health and Airflow status, and assess retraining"
python -m agent.main --audit --experiment 1
~~~

The --audit mode checks all four services without an LLM key and reports facts, unknowns, and the underlying responses. The conversational mode needs an API key. Training is never triggered solely by a request to assess whether retraining is needed. An actual training request requires --approve-training; a restart request requires --approve-restart. Tool metadata logs contain request IDs, tool names, status, and duration, not credentials. Optional --trace PATH writes the same metadata to a local JSONL file.

## Testing and limitations

~~~bash
python -m pytest -q
python -m ruff check --select E4,E7,E9,F agent api dags scripts tests training
~~~

Tests cover tool selection, multiple tool calls, invalid arguments, service failures, timeouts, approval gates, redaction, provenance, training, and API inference. Tests that need the model artifact skip until dvc pull restores it. External service tests use mocks in CI.

The dataset and DVC remote are local demonstrations. The deployment image is not yet linked to an MLflow run ID, so a healthy Deployment alone cannot prove that the best MLflow candidate is serving traffic. Airflow standalone and its simple authentication are for local development. A shared DVC remote, durable tracking backend, tagged model image, and production authentication are future improvements.
