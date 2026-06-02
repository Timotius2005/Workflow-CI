# Workflow-CI — MLflow Project + Docker CI/CD

> **Criterion 3 — Advanced** | SMSML Dicoding  
> **Student:** Timotius Kristafael Harjanto

---

## Repository Structure

```
Workflow-CI/
│
├── .github/
│   └── workflows/
│       └── training.yml      ← GitHub Actions CI/CD pipeline
│
├── MLProject/
│   ├── modelling.py          ← Training script (argparse entry point)
│   ├── conda.yaml            ← Conda environment spec
│   ├── MLproject             ← MLflow Project definition
│   ├── Dockerfile            ← Docker image for model serving
│   ├── requirements.txt      ← pip dependencies
│   └── dataset_preprocessed/ ← Preprocessed data (committed or CI-generated)
│
└── README.md
```

---

## Required GitHub Secrets

Configure these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | DagsHub personal access token |
| `DOCKER_USERNAME` | Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token |

---

## MLflow Project — Local Run

```bash
cd MLProject

# With conda (recommended for exact reproducibility)
mlflow run . \
  -P n_estimators=200 \
  -P max_depth=20 \
  -P dagshub_username=Timotius2005

# Without conda (faster, uses current env)
pip install -r requirements.txt
mlflow run . --env-manager=local \
  -P n_estimators=200 \
  -P max_depth=20
```

---

## GitHub Actions CI/CD Pipeline

`.github/workflows/training.yml` runs **three jobs**:

### Job 1: `train`
1. Checkout code
2. Set up Python 3.12.7
3. Install dependencies
4. Download / generate preprocessed dataset inline
5. Execute `mlflow run . --env-manager=local` with configurable hyperparameters
6. Upload trained model as GitHub Artifact (90-day retention)

### Job 2: `docker`
1. Download trained model artifact
2. Build Docker image (multi-arch via `docker/build-push-action`)
3. Push to Docker Hub with tags: `latest`, `v1.0.<run_number>`, `sha-<commit>`

### Job 3: `release`
1. Download model artifact
2. Compress to `.tar.gz`
3. Create GitHub Release with model archive attached

---

## Docker Image Usage

```bash
# Pull the image
docker pull <DOCKER_USERNAME>/adult-income-ml:latest

# Run model server
docker run -d -p 5001:5001 \
  --name adult-income-server \
  <DOCKER_USERNAME>/adult-income-ml:latest

# Test the endpoint
curl http://localhost:5001/ping

# Make a prediction
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["age","workclass","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","age_group","capital_net","has_capital_gain","has_capital_loss","higher_education","work_hours_category","is_married"], "data": [[0.42,4,9,0.7,2,3,0,4,1,0.3,0.0,0.1,39,2,0.3,1,0,1,1,1]]}}'
```

---

## Manual Trigger (workflow_dispatch)

Go to **Actions → ML Training Pipeline → Run workflow** and set:

| Parameter | Default | Description |
|---|---|---|
| `n_estimators` | 200 | Number of trees in the forest |
| `max_depth` | 20 | Maximum tree depth (0 = unlimited) |
| `min_samples_split` | 2 | Min samples to split a node |
| `min_samples_leaf` | 1 | Min samples per leaf |

---

## MLproject Entry Points

```yaml
# conda.yaml  — full reproducible environment
# MLproject   — defines parameters + command
entry_points:
  main:
    parameters:
      n_estimators:      {type: int, default: 200}
      max_depth:         {type: int, default: 20}
      min_samples_split: {type: int, default: 2}
      min_samples_leaf:  {type: int, default: 1}
      data_path:         {type: string}
      dagshub_username:  {type: string}
      dagshub_repo:      {type: string}
    command: "python modelling.py ..."
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Docker push fails | Check `DOCKER_USERNAME` / `DOCKER_PASSWORD` secrets |
| DagsHub connection refused | Verify `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` |
| MLflow run fails | Check conda is installed or use `--env-manager=local` |
| No `train.csv` found | The CI step generates it inline from UCI; check internet in runner |
