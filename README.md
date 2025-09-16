# Porch Scoring Pipeline

This repository contains the **Acxiom/Epsilon model scoring pipeline** for Porch.
The pipeline fetches raw files from **S3**, preprocesses them, scores them using XGBoost models, and writes results back to **S3 outputs**.

---

## 🚀 Features

* Read input data (Acxiom/Epsilon) directly from S3
* Preprocess features (handle nulls, categorical encoding, etc.)
* Score records using pre-trained XGBoost models from S3
* Write outputs to S3 with timestamped filenames
* Lightweight Docker image for SageMaker / Step Functions deployment

---

## 📂 Project Structure

```
.
├── app.py                 # Main entrypoint (runs the Acxiom pipeline)
├── Dockerfile             # Docker image for local & SageMaker runs
├── requirements.txt       # Python dependencies
├── config/
│   └── config.py          # Central config (S3 assets, cache dir, etc.)
├── helpers/
│   ├── preprocessing_acxiom.py  # Preprocessing logic for Acxiom data
│   ├── read_s3.py               # S3 download utility
│   ├── scoring.py               # Scoring logic (merge + model predictions)
│   └── write_s3.py              # S3 upload utility
├── .gitignore             # Ignore secrets, cache, temp files
├── .dockerignore          # Keep Docker image clean
└── data_cache/            # Local cache (inputs + outputs, gitignored)
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/jithunair-blend/porch-scoring.git
cd porch-scoring
```

### 2. Install dependencies (local run)

Use Python 3.12:

```bash
pip install -r requirements.txt
```

### 3. Configure AWS credentials

Create a `.env` file in the project root with:

```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

⚠️ **Never commit `.env`** — it is ignored by `.gitignore`.

---

## ▶️ Running the Pipeline

### Local Python

```bash
python app.py
```

This will:

1. Download input files from S3 → `data_cache/`
2. Preprocess + score Acxiom data
3. Upload results to `s3://porchoutput/acxiom/...`

---

### Docker

Build the image:

```bash
docker build -t porch-scoring .
```

Run with `.env` for AWS credentials:

```bash
docker run --rm --env-file .env porch-scoring
```

---

## ☁️ Deployment (SageMaker)

1. Build and tag the image:

   ```bash
   docker build -t porch-scoring .
   docker tag porch-scoring:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/porch-scoring:latest
   ```

2. Push to ECR:

   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com
   docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/porch-scoring:latest
   ```

3. Run in SageMaker as a **Processing Job** or inside **Step Functions** to fully automate scoring.

---

## 📝 Notes

* All local files (inputs/outputs) are cached under `data_cache/` (ignored in git + Docker).
* Outputs are automatically timestamped when written to S3.
* Currently only **Acxiom** pipeline is enabled; **Epsilon** code will be integrated next.