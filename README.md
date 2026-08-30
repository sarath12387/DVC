# MLOps Pipeline — Cats vs Dogs Classification

End-to-end MLOps pipeline for binary image classification (cats vs dogs) for a
pet adoption platform. Covers model building, experiment tracking, packaging,
containerization, CI/CD deployment, and monitoring using open-source tools.

**Stack:** Git + DVC | PyTorch | MLflow | FastAPI | Docker | GitHub Actions | Docker Hub | Docker Compose

## Project Structure

```
mlops-cats-dogs/
├── data/
│   ├── raw/PetImages/        # <- put the Kaggle dataset here (Cat/, Dog/)
│   └── processed/            # created by preprocess.py (224x224, 80/10/10)
├── src/
│   ├── preprocess.py         # M1: clean, resize, split
│   ├── train.py              # M1: CNN + MLflow tracking
│   ├── predict.py            # inference utility
│   └── app.py                # M2: FastAPI service | M5: logging + metrics
├── tests/
│   ├── test_preprocess.py    # M3: preprocessing unit tests
│   └── test_predict.py       # M3: inference unit tests
├── scripts/
│   ├── smoke_test.py         # M4: post-deploy smoke test
│   └── performance_tracking.py  # M5: post-deployment accuracy check
├── .github/workflows/ci.yml  # M3+M4: CI/CD pipeline
├── Dockerfile                # M2: containerization
├── docker-compose.yml        # M4: deployment target
├── requirements.txt          # M2: pinned dependencies
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the **Cats and Dogs Classification Dataset** from Kaggle and extract
so that you have `data/raw/PetImages/Cat/` and `data/raw/PetImages/Dog/`.

## M1 — Model Development & Experiment Tracking

```bash
# 1. Version code with Git
git init && git add . && git commit -m "Initial project structure"

# 2. Version data with DVC (tracks data/, keeps Git lean)
dvc init
dvc add data/raw
git add data/raw.dvc .dvc .dvcignore && git commit -m "Track raw dataset with DVC"

# 3. Preprocess: remove corrupted images, resize 224x224, split 80/10/10
python src/preprocess.py --raw-dir data/raw/PetImages --out-dir data/processed
dvc add data/processed
git add data/processed.dvc && git commit -m "Track processed data with DVC"

# 4. Train + track experiments (run 2-3 times with different params)
python src/train.py --epochs 5  --lr 0.001 --batch-size 32
python src/train.py --epochs 8  --lr 0.0005 --batch-size 64

# 5. Inspect runs, metrics, loss curves, confusion matrix
mlflow ui           # open http://localhost:5000
```

Model is saved to `models/model.pt` (state dict + class names).

> Tip: for a quick first run, add `--limit-per-class 2000` to preprocess.py.

## M2 — Packaging & Containerization

```bash
# Run the API locally (without Docker)
MODEL_PATH=models/model.pt uvicorn app:app --app-dir src --port 8000

# Containerize
docker build -t YOUR_DOCKERHUB_USERNAME/cats-dogs-api:latest .
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/cats-dogs-api:latest

# Verify
curl http://localhost:8000/health
curl -F "file=@some_cat.jpg" http://localhost:8000/predict
```

Endpoints: `GET /health`, `POST /predict` (multipart image), `GET /metrics`.

## M3 — CI Pipeline (GitHub Actions)

```bash
pytest tests/ -v        # run the unit tests locally
```

The workflow `.github/workflows/ci.yml` runs on every push / pull request:
**checkout → install deps → pytest → docker build → push to Docker Hub**.

One-time setup:
1. Create a GitHub repo and push this project.
2. Create a Docker Hub access token (Account Settings → Security).
3. Add repo secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`
   (GitHub repo → Settings → Secrets and variables → Actions).
4. Replace `YOUR_DOCKERHUB_USERNAME` in `docker-compose.yml`.
5. The trained `models/model.pt` must be available to the build —
   simplest: `git lfs install && git lfs track "*.pt"` then commit the model.

## M4 — CD & Deployment (Docker Compose + self-hosted runner)

The `deploy` job in `ci.yml` runs on a **self-hosted runner** (your laptop/VM):

1. GitHub repo → Settings → Actions → Runners → *New self-hosted runner* —
   follow the shown commands to install and start the runner on your machine.
2. On every push to `main` that passes tests, the pipeline pushes a new image
   and the deploy job runs `docker compose pull && docker compose up -d`,
   then executes the smoke test.

```bash
# Manual deploy + smoke test (same thing the pipeline does)
docker compose up -d
python scripts/smoke_test.py --url http://localhost:8000
```

The smoke test calls `/health` and makes one `/predict` call; a non-zero exit
fails the pipeline.

## M5 — Monitoring & Performance Tracking

- Every request is logged (method, path, status, latency, predicted label —
  never image content).
- `GET /metrics` exposes in-app counters: total requests, predictions by
  label, errors, average latency.

```bash
curl http://localhost:8000/metrics

# Post-deployment accuracy on a batch of labeled test images
python scripts/performance_tracking.py --n-per-class 15
# -> prints accuracy and writes performance_report.csv
```

## Deliverables Checklist

- [ ] Zip of the repo: source code, DVC files, CI/CD workflow, Dockerfile,
      docker-compose.yml, trained `models/model.pt`
- [ ] Screen recording (<5 min): code change → push → CI tests → image build
      & push → auto deploy → `/health` + `/predict` on the live service
