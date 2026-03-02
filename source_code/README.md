# AI Pattern Mining Pipeline (Step-by-Step)

This project mines AI design patterns from papers and repositories, generates synthetic and community code summaries, creates embeddings, trains classifiers, and predicts unlabeled items.

This guide is written for the current scripts and folder structure in this workspace.

## 1) Prerequisites

- Python 3.10+ (3.11 recommended)
- Google Gemini API key
- Windows PowerShell (examples below use PowerShell)

Install dependencies:

```powershell
cd source_code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set environment variables (session-level example):

```powershell
$env:GOOGLE_API_KEY="YOUR_KEY"
$env:PATTERN_EXTRACT_TAG="ai"
$env:PATTERN_EXTRACT_VERSION="v1"
```

Notes:
- If GOOGLE_API_KEY is not set, some scripts will prompt for it.
- PATTERN_EXTRACT_TAG and PATTERN_EXTRACT_VERSION control input/output folders under outputs/<tag>/<version>.

---

## 2) Put input data in the correct folders

### 2.1 Papers (PDFs)

Place PDFs in:

data/raw/papers/<tag>/

Example for tag ai:

```powershell
mkdir .\data\raw\papers\ai -Force
copy "D:\your_data\papers\RAG_survey.pdf" .\data\raw\papers\ai\
copy "D:\your_data\papers\agents_overview.pdf" .\data\raw\papers\ai\
```

### 2.2 Repositories (source code)

Place each repository as its own folder in:

data/raw/repos/

Example:

```powershell
mkdir .\data\raw\repos -Force
robocopy "D:\your_data\repos\jodie" .\data\raw\repos\jodie /E
robocopy "D:\your_data\repos\optillm" .\data\raw\repos\optillm /E
```

After this, you should have paths like:
- data/raw/papers/ai/*.pdf
- data/raw/repos/jodie/
- data/raw/repos/optillm/

---

## 3) Run pipeline step-by-step

Run all commands from source_code root with the venv activated.

### Step 1A: Extract patterns from papers

```powershell
python .\step-01\step1_extract_patterns.py
```

Main outputs:
- outputs/<tag>/<version>/extracted_patterns/l2_patterns.json
- outputs/<tag>/<version>/extracted_patterns/*_patterns.json

### Step 1B: Embed + cluster extracted patterns

```powershell
python .\step-01\step1_cluster_and_summarize_patterns.py
```

Main outputs:
- outputs/<tag>/<version>/extracted_patterns/pattern_embeddings.csv
- outputs/<tag>/<version>/extracted_patterns/umap_clustered_dataset.csv
- outputs/<tag>/<version>/extracted_patterns/clusters.json

Important:
- If your workflow requires curated_clusters for later synthetic generation, ensure outputs/<tag>/<version>/curated_patterns.json exists.

### Step 2A: Build callgraph communities from repos

```powershell
python .\step-02\step2_callgraph_and_communities.py
```

Main outputs:
- outputs/<tag>/<version>/callgraph_communities/<repo>/cluster_*.py

### Step 2B: Summarize each community code cluster

```powershell
python .\step-02\step2_communities_to_code_summary_communities.py
```

Main output:
- outputs/<tag>/<version>/code_samples_with_summary_communities.json

### Step 2C: Embed community code summaries

```powershell
python .\step-02\step2_code_summary_to_embeddings_communities.py
```

Main output:
- outputs/<tag>/<version>/code_summary_embeddings_communities.csv

### Step 3A: Generate synthetic code samples from curated patterns

```powershell
python .\step-03\step3_generate_bootstrap_samples.py
```

Main outputs:
- outputs/<tag>/<version>/code_samples.json
- outputs/<tag>/<version>/extracted_patterns/pattern_summaries.json

Tip (free tier):
- Keep sample count low and rerun; scripts are resumable via cached outputs.

### Step 3B: Summarize synthetic code samples

```powershell
python .\step-03\step3_boostrap_samples_to_code_summary.py
```

Main output:
- outputs/<tag>/<version>/code_samples_with_summary_synthetic.json

### Step 3C: Embed synthetic code summaries

```powershell
python .\step-03\step3_code_summary_to_embeddings_synthetic.py
```

Main output:
- outputs/<tag>/<version>/code_summary_embeddings_synthetic.csv

### Step 3D: Preprocess and split training files

By default, synthetic + community embeddings are combined.

To include synthetic:

```powershell
$env:USE_SYNTHETIC_DATA="true"
python .\step-03\step3_preprocess_data.py
```

To use only community embeddings:

```powershell
$env:USE_SYNTHETIC_DATA="false"
python .\step-03\step3_preprocess_data.py
```

Main outputs:
- outputs/<tag>/<version>/labeled_data.csv  (includes file, pattern, verified)
- outputs/<tag>/<version>/embeddings_data.csv

### Step 3E: Train models + save artifacts

```powershell
python .\step-03\step3_train_models_cv.py
```

Main outputs:
- outputs/<tag>/<version>/artifacts/models/
	- logreg_model.joblib
	- svc_model.joblib
	- knn_model.joblib
	- label_encoder.joblib
	- classes.json
	- feature_list.json
- outputs/<tag>/<version>/artifacts/outputs/
	- probs_*.csv
	- weighted_confusion.png
	- ensemble_classification_predictions.csv

### Step 4: Predict unlabeled items using saved models

```powershell
python .\step-04\step4_forecast_and_simulate.py
```

Main output:
- outputs/<tag>/<version>/artifacts/outputs/unverified_weighted_predictions.csv

Optional utility:

```powershell
python .\step-04\simulate_counts_confusion_matrix.py
```

---

## 4) Minimal run order (copy/paste)

```powershell
python .\step-01\step1_extract_patterns.py
python .\step-01\step1_cluster_and_summarize_patterns.py
python .\step-02\step2_callgraph_and_communities.py
python .\step-02\step2_communities_to_code_summary_communities.py
python .\step-02\step2_code_summary_to_embeddings_communities.py
python .\step-03\step3_generate_bootstrap_samples.py
python .\step-03\step3_boostrap_samples_to_code_summary.py
python .\step-03\step3_code_summary_to_embeddings_synthetic.py
$env:USE_SYNTHETIC_DATA="true"
python .\step-03\step3_preprocess_data.py
python .\step-03\step3_train_models_cv.py
python .\step-04\step4_forecast_and_simulate.py
```

---

## 5) Common issues

- 429 / RESOURCE_EXHAUSTED on Gemini:
	- wait and rerun (scripts save partial progress)
	- reduce synthetic generation volume
- Missing curated_patterns.json:
	- provide curated patterns before running synthetic generation
- No predictions in Step 4:
	- check labeled_data.csv verified column and whether any rows are unverified

