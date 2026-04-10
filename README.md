# HOW CAN AI FIND MY MODEL? A Model-Finding Experimental Study Considering Data Formats, Embeddings, and Retrieval Strategies

This repository benchmarks how well different retrieval pipelines can discover relevant system dynamics models from multiple machine-readable representations of the same corpus.

The project compares:

- Lexical retrieval methods such as BM25, TF-IDF, and query likelihood
- Dense retrieval with local and OpenAI embedding models
- Re-ranking pipelines using cross-encoders and an LLM
- Three document representations: raw XMILE, JSON-LD, and JSON-LD enriched with simulation model descriptions.

The repository is organized so that someone can reproduce the pipeline from preprocessing to evaluation, inspect the saved experiment outputs, and extend the benchmark with new retrieval strategies.

## Why this work is useful

- It frames simulation model discovery as an information retrieval (IR) problem.
- It keeps the original simulation-model representations alongside transformed formats.
- It evaluates retrieval quality with graded relevance labels in [`evaluation/ground_truth.json`](evaluation/ground_truth.json).
- It stores experiment summaries, per-query results, timing logs, and Excel exports for both lexical and dense pipelines.
- It provides a concrete example of how narrative augmentation changes retrieval performance.

## Repository at a glance

```text
.
├── Vensim_Models/
│   ├── XML-based-Files/          # XMILE source files
│   ├── JSONLD-Files/             # Structured JSON-LD converted from XMILE
│   ├── JSONLD-Narratives/        # JSON-LD enriched with model background information
│   └── Background_Information/   # Spreadsheet and reference material used for enrichment
├── evaluation/
│   ├── ground_truth.json         # Queries and graded relevance labels
│   ├── lexical/                  # Saved lexical retrieval results
│   └── dense/                    # Saved dense retrieval results
├── embeddings/                   # Persistent ChromaDB indexes for dense retrieval
├── preprocess_XML.py             # XMILE -> JSON-LD conversion
├── narratives_append.py          # Adds background information to JSON-LD files
├── evaluate_lexical.py           # Lexical retrieval benchmark
├── evaluate_dense_index.py       # Dense retrieval benchmark with Chroma indexes
├── cost.py                       # OpenAI cost estimation helper
├── requirements.txt
└── .env.example
```

## Included data and outputs

At the time this README was prepared, the repository already contains:

- 40 XMILE files in `Vensim_Models/XML-based-Files`
- 40 JSON-LD files in `Vensim_Models/JSONLD-Files`
- 40 narrative-enriched JSON-LD files in `Vensim_Models/JSONLD-Narratives`
- Saved lexical and dense evaluation outputs under `evaluation/`

## Method overview

The benchmark follows this workflow:

1. Start from a corpus of system dynamics models exported to XMILE.
2. Convert XMILE files into compact JSON-LD representations.
3. Enrich JSON-LD files with narrative background information from the spreadsheet in `Vensim_Models/Background_Information`.
4. Evaluate lexical retrieval over XMILE, JSON-LD, and narrative-enriched JSON-LD.
5. Evaluate dense retrieval over the same representations with multiple chunk sizes, overlap settings, embedding models, and reranking strategies.
6. Save JSON summaries, timing logs, and Excel workbooks for downstream analysis.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/jgbotello/Simulation-Model-Discovery-M-S.git
cd Simulation-Model-Discovery-M-S
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and add your own OpenAI API key:

```bash
cp .env.example .env
```

Then edit `.env` so it contains:

```env
OPENAI_API_KEY=your_openai_api_key_here
RECOMPUTE_EMBEDDINGS=true
```

- Set `RECOMPUTE_EMBEDDINGS=false` if you do not want to delete and rebuild existing Chroma indexes from scratch.

## Reproducing the full pipeline

### Step 1. Generate JSON-LD from XMILE

```bash
python preprocess_XML.py
```

This reads `.xmile` files from `Vensim_Models/XML-based-Files` and writes compact JSON-LD files to `Vensim_Models/JSONLD-Files`.

### Step 2. Append narrative background information

```bash
python narratives_append.py
```

This reads model background information from [`Vensim_Models/Background_Information/Models_Info.xlsx`](Vensim_Models/Background_Information/Models_Info.xlsx) and writes enriched files to [`Vensim_Models/JSONLD-Narratives`](Vensim_Models/JSONLD-Narratives).

### Step 3. Run lexical retrieval experiments

```bash
python evaluate_lexical.py
```

Outputs are saved under [`evaluation/lexical`](evaluation/lexical).

Key artifacts:

- `summary_overall.json`
- `summary_by_type.json`
- `results_per_query.json`
- `timing_summary.json`
- `lexical_experiment_results.xlsx`

### Step 4. Run dense retrieval experiments

```bash
python evaluate_dense_index.py
```

Outputs are saved under [`evaluation/dense`](evaluation/dense).

Persistent vector indexes are stored under `embeddings/`.

Key artifacts:

- `summary_overall.json`
- `summary_by_type.json`
- `results_per_query.json`
- `timing_summary.json`
- `index_build_log.json`
- `dense_experiment_results.xlsx`

### Step 5. Estimate OpenAI cost

```bash
python cost.py
```

This script estimates token usage and API cost for the OpenAI-based dense-retrieval components.

## Experiment Configurations

### Data Representations

- `xml`: raw XMILE files
- `json`: compact JSON-LD converted from XMILE
- `json_narratives`: JSON-LD enriched with model background information

### Lexical methods

- BM25
- TF-IDF
- Query likelihood

### Dense retrieval variations

- Chunk sizes: `800`, `2000`, `4000`
- Chunk overlaps: `120`, `200`, `300`
- Embedding models:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `BAAI/bge-base-en-v1.5`
  - `text-embedding-3-large`
- Retrieval pipelines:
  - `dense_only`
  - `dense_rerank`
  - `dense_llm_rerank`


## Reproducibility notes

- Run all commands from the repository root.
- Dense experiments can take substantial time because they iterate across many combinations and may build or load multiple indexes.
- The `embeddings/` directory stores ChromaDB state, so results can be rerun faster when `RECOMPUTE_EMBEDDINGS=false`.
- Excel outputs rely on `openpyxl`.
- The cost-estimation script relies on `tiktoken`.

## Citation and reuse

If you build on this repository in a paper, thesis, or benchmark extension, cite the repository and please document:

- Which representation was used
- Which retrieval pipeline was used
- Whether embeddings were rebuilt
- Which relevance file was used for evaluation
- Any changes to chunking, reranking, or prompt settings

