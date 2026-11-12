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

The tests cover deterministic generation and training, command selection, invalid references, tool failures, and timeouts. Live DVC version and remote evidence still needs a Git checkout with DVC installed. I will record real hashes and screenshots when that runs.

Next I will add MLflow tracking, an Airflow DAG, FastAPI on Kubernetes, and then the unified agent, CI, and final report.


