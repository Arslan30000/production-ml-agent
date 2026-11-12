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

Next I will add the Airflow DAG, FastAPI on Kubernetes, and then the unified agent, CI, and final report.


