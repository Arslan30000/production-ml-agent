# Production ML Agent

I am building an agent that checks real MLOps services before answering questions about data, experiments, training, or deployments. I am developing it in stages. Stage 1 contains a repeatable classification baseline and DVC data tools.

I will connect DVC dataset provenance, MLflow experiments, Airflow training workflows, and Kubernetes diagnostics through structured tools. I will use FastAPI for model inference.

## Connect this WSL folder to GitHub

I run this once in my own WSL terminal. The remote already has the Stage 1 source files. `reset --mixed` aligns Git's index with that remote without replacing files in my working folder.

```bash
cd ~/MLOPS/A3
git init -b main
git remote add origin https://github.com/Arslan30000/production-ml-agent.git
git fetch origin main
git reset --mixed origin/main
git branch --set-upstream-to=origin/main main
git status
```

## Stage 1 setup (Ubuntu/WSL)

I use Python 3.11 or newer. From this repository's root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m training.generate_data --output data/raw/equipment.csv
python -m training.train --input data/raw/equipment.csv --model artifacts/model.json --metrics reports/metrics.json
python -m unittest discover -s tests -v
pytest -q
```

The generator uses a fixed seed. The training script splits the data deterministically and writes a JSON model and evaluation metrics. The equipment readings are synthetic demonstration data, not readings from real machines.

## Versioning with DVC

I run these commands after connecting the local Git repository. The default remote is a folder next to this checkout. It supports a local demonstration, but another computer needs a copy of that remote before it can pull the data.

```bash
dvc init
python -m training.generate_data --output data/raw/equipment.csv
dvc add data/raw/equipment.csv
dvc remote add -d local ../production-ml-agent-dvc-storage
dvc repro
dvc push
git add .dvc .dvcignore data/raw/equipment.csv.dvc data/raw/.gitignore dvc.yaml dvc.lock reports/.gitignore README.md
git commit -m "Track baseline dataset with DVC"
git push
```

I check `git status` before committing and include any other DVC configuration files it created. To make a second version, I regenerate the data with a different `--seed`, run `dvc add` and `dvc push` again, and commit the updated `.dvc` file. The two commits can then be compared with `dvc diff`.

```bash
python -m agent.tools.dvc_tools list
python -m agent.tools.dvc_tools version
python -m agent.tools.dvc_tools status
python -m agent.tools.dvc_tools compare HEAD~1 HEAD
python -m agent.tools.dvc_tools reproduce HEAD~1
```

`reproduce` prints checkout instructions. I use a separate clone or worktree for an older dataset so my active work is not overwritten. On another machine, I would configure a shared DVC remote, run `dvc pull`, and then `dvc checkout` at the recorded Git revision. I do not put credentials in the repository.

## Verification and next stages

I committed `data-v1` and `data-v2`, compared their DVC hashes, and restored `data-v1` in an isolated worktree. The restored dataset's SHA-256 matched the original baseline. My DVC remote is local to this machine, so another computer needs the remote folder to fetch these objects.

## Stage 2: MLflow experiments

I install the updated project dependencies in `~/MLOPS/A3`, with the virtual environment active:

```bash
cd ~/MLOPS/A3
python -m pip install -e '.[dev]'
pytest -q
```

In a second WSL terminal I start the local tracking server, leaving it running:

```bash
cd ~/MLOPS/A3
source .venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000
```

Back in the first terminal I log multiple real runs using the current committed DVC dataset. Every run stores F1, accuracy, precision, recall, the train/test split, the model and metrics JSON artifacts, the dataset SHA-256, the DVC MD5, and the Git commit containing the dataset pointer.

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m training.log_experiment --seed 42
python -m training.log_experiment --seed 43
python -m training.log_experiment --seed 44
python -m agent.mlflow_cli experiments
```

I take the returned experiment ID and run IDs from the output, then inspect and compare them:

```bash
python -m agent.mlflow_cli runs EXPERIMENT_ID
python -m agent.mlflow_cli best EXPERIMENT_ID --metric f1
python -m agent.mlflow_cli details RUN_ID
python -m agent.mlflow_cli compare RUN_ID_1 RUN_ID_2 --metric f1
```

I can open `http://127.0.0.1:5000` to inspect the same runs in the MLflow UI. A high F1 score chooses a candidate, but it does not by itself mean the model is ready to deploy. The model registry tool returns registered versions if I later register a model; an empty list is valid while none are registered.

## Stage 3: Airflow and Kubernetes

I install the optional services in the active WSL virtual environment with `python -m pip install -e '.[airflow,api,dev]'`. For a repeatable Airflow installation, I can use the version-matched Airflow constraints if pip reports dependency conflicts. I keep Airflow's database outside Git.

In one terminal I keep MLflow running with `mlflow server --host 127.0.0.1 --port 5000`. Before triggering the DAG, I check that `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:5000/` returns a successful HTTP status. If MLflow is stopped, the evaluation task cannot log its run and the quality gate cannot execute. In a second terminal I start Airflow:

```bash
cd ~/MLOPS/A3
source .venv/bin/activate
export MLOPS_PROJECT_ROOT="$PWD"
export AIRFLOW_HOME="$PWD/airflow-home"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
airflow standalone
```

Airflow creates the username `admin` and saves its generated password in `airflow-home/simple_auth_manager_passwords.json.generated`. If I missed the password in the first startup output, I read it locally with `cat airflow-home/simple_auth_manager_passwords.json.generated`. I do not commit or paste that password into a report. In another terminal, with the same virtual environment active, I obtain a short-lived API token without placing my password in shell history:

```bash
cd ~/MLOPS/A3
source .venv/bin/activate
export AIRFLOW_HOME="$PWD/airflow-home"
export AIRFLOW_API_URL=http://127.0.0.1:8080
export AIRFLOW_API_TOKEN="$(python scripts/airflow_token.py)"
airflow dags unpause equipment_training -y
python -m agent.airflow_cli dags
python -m agent.airflow_cli trigger --confirm
python -m agent.airflow_cli runs
# Copy the actual dag_run_id from the trigger response, not the text RUN_ID.
RUN_ID='manual__2026-11-12T13:33:11.028062+00:00'
python -m agent.airflow_cli status "$RUN_ID"
python -m agent.airflow_cli tasks "$RUN_ID"
python -m agent.airflow_cli failed "$RUN_ID"
python -m agent.airflow_cli trigger --confirm --inject-failure validation
python -m agent.airflow_cli trigger --confirm --inject-failure quality_gate
```

I unpause the DAG before inspecting runs because Airflow accepts manual triggers while paused but does not schedule their tasks. The `RUN_ID` above is an example from one local run; I replace it with the exact `dag_run_id` I want to inspect. The DAG has validation, preprocessing, training, evaluation with MLflow logging, and a quality gate. It retries task failures twice. I inspect each failed run ID with `python -m agent.airflow_cli failed "$RUN_ID"` and read its log with `python -m agent.airflow_cli log "$RUN_ID" TASK_ID`. I deliberately inject a validation failure and a quality gate failure to check that Airflow stops downstream tasks. I return to a normal run by triggering without `--inject-failure`. A failed quality gate does not promote the model. The trigger tool requires `--confirm`; inspection commands are read-only.

For Kubernetes I need a local cluster with Docker, `kubectl`, and either Minikube or Kind. With Minikube's Docker driver, I run:

```bash
cd ~/MLOPS/A3
dvc pull
docker build -t production-ml-agent:local .
minikube start --driver=docker
minikube image load production-ml-agent:local
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent rollout status deployment/equipment-api
python -m agent.kubernetes_cli version
python -m agent.kubernetes_cli diagnose
python -m agent.kubernetes_cli services
```

The Deployment requests two replicas with readiness and liveness probes and bounded CPU and memory. The Service is ClusterIP, so I test it locally with `kubectl -n mlops-agent port-forward svc/equipment-api 8000:80` and call `/health/ready`, `/version`, and `/predict` on `http://127.0.0.1:8000`. I only inspect resources in `mlops-agent`. A deployment restart needs explicit confirmation and only accepts `equipment-api`.

In a separate terminal while the port forward is running, I check the actual model response:

```bash
curl -s http://127.0.0.1:8000/health/ready
curl -s http://127.0.0.1:8000/version
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"temperature_c":80,"vibration_mm_s":4,"load_pct":65}'
kubectl -n mlops-agent get pods,deployments,services
python -m unittest discover -s tests -v
```


