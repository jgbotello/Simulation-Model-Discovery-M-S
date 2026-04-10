import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import mean
from rank_bm25 import BM25Okapi

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# CONFIG
# =========================
BASE_DIR = Path("Vensim_Models")
GROUND_TRUTH_PATH = Path("evaluation") / "ground_truth.json"
OUTPUT_DIR = Path("evaluation") / "lexical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATION_PATHS = {
    "xml": {
        "dir": BASE_DIR / "XML-based-Files",
        "pattern": "*.xmile",
    },
    "json": {
        "dir": BASE_DIR / "JSONLD-Files",
        "pattern": "*.jsonld",
    },
    "json_narratives": {
        "dir": BASE_DIR / "JSONLD-Narratives",
        "pattern": "*.jsonld",
    },
}

REPRESENTATIONS = [
    "xml",
    "json",
    "json_narratives",
]

LEXICAL_RETRIEVALS = [
    "bm25",
    "tfidf",
    "query_likelihood",
]

TOP_K_DOCS = 5
QUERY_TYPE_ORDER = ["direct", "abstract", "ambiguous"]


# =========================
# EXPERIMENT MANIFEST
# =========================
def build_lexical_experiments():
    experiments = []

    for representation, retrieval in product(REPRESENTATIONS, LEXICAL_RETRIEVALS):
        exp_name = f"{representation}__{retrieval}"

        experiments.append({
            "name": exp_name,
            "experiment_group": "lexical",
            "representation": representation,
            "retrieval_type": "lexical",
            "lexical_method": retrieval,
            "index_level": "document",
            "chunk_size": None,
            "chunk_overlap": None,
            "embedding_backend": None,
            "embedding_model": None,
            "top_k_docs": TOP_K_DOCS,
        })

    return experiments


LEXICAL_EXPERIMENTS = build_lexical_experiments()


# =========================
# TEXT / FILE HELPERS
# =========================
def normalize_doc_name(name: str) -> str:
    base = os.path.basename(str(name)).strip().lower()
    for ext in [".xmile", ".jsonld", ".mdl", ".xml", ".json", ".txt"]:
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base


def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_representation_document(path: Path, representation: str) -> str:
    """
    Read one document as plain text for lexical retrieval.
    For JSON/JSON narratives, canonicalize to compact JSON text.
    For XML, keep raw text.
    """
    if representation == "xml":
        return read_text_file(path)

    if representation in {"json", "json_narratives"}:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    raise ValueError(f"Unsupported representation: {representation}")


def load_corpus(representation: str) -> list[dict]:
    cfg = REPRESENTATION_PATHS[representation]
    doc_dir = cfg["dir"]
    pattern = cfg["pattern"]

    if not doc_dir.exists():
        raise FileNotFoundError(f"Representation folder not found: {doc_dir}")

    paths = sorted(doc_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"No files found for representation={representation} in {doc_dir}")

    corpus = []
    for path in paths:
        text = read_representation_document(path, representation)
        corpus.append({
            "doc_name": path.name,
            "doc_id": normalize_doc_name(path.name),
            "path": str(path),
            "text": text,
        })

    return corpus


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


# =========================
# GROUND TRUTH
# =========================
def load_ground_truth(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for item in data:
        graded = item.get("graded_relevance", {})
        graded_norm = {normalize_doc_name(k): int(v) for k, v in graded.items()}

        # if "relevant" exists but graded_relevance does not, fallback to binary relevance
        if not graded_norm and "relevant" in item:
            graded_norm = {normalize_doc_name(x): 1 for x in item["relevant"]}

        rows.append({
            "query_type": item["query_type"],
            "query": item["query"],
            "graded_relevance": graded_norm,
            "relevant": sorted(list(graded_norm.keys())),
        })

    return rows


def validate_query_type_counts(rows: list[dict]) -> dict:
    counts = defaultdict(int)
    for row in rows:
        counts[row["query_type"]] += 1
    return dict(counts)


# =========================
# METRICS
# =========================
def precision_at_k(ranked_docs: list[str], relevant_docs: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = ranked_docs[:k]
    hits = sum(1 for d in top_k if d in relevant_docs)
    return hits / k


def recall_at_k(ranked_docs: list[str], relevant_docs: set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    top_k = ranked_docs[:k]
    hits = sum(1 for d in top_k if d in relevant_docs)
    return hits / len(relevant_docs)


def dcg_at_k(ranked_docs: list[str], graded_relevance: dict[str, int], k: int) -> float:
    score = 0.0
    for i, doc in enumerate(ranked_docs[:k], start=1):
        rel = graded_relevance.get(doc, 0)
        gain = (2 ** rel - 1)
        score += gain / math.log2(i + 1)
    return score


def ndcg_at_k(ranked_docs: list[str], graded_relevance: dict[str, int], k: int) -> float:
    dcg = dcg_at_k(ranked_docs, graded_relevance, k)

    ideal_rels = sorted(graded_relevance.values(), reverse=True)[:k]
    if not ideal_rels:
        return 0.0

    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        gain = (2 ** rel - 1)
        idcg += gain / math.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def precision_at_r(ranked_docs: list[str], relevant_docs: set[str]) -> float:
    r = len(relevant_docs)
    if r == 0:
        return 0.0
    top_r = ranked_docs[:r]
    hits = sum(1 for d in top_r if d in relevant_docs)
    return hits / r


# =========================
# BM25
# =========================
class BM25Retriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.doc_tokens = [tokenize(doc["text"]) for doc in corpus]
        self.bm25 = BM25Okapi(self.doc_tokens)

    def score_query(self, query: str) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scores = self.bm25.get_scores(q_terms)

        ranked = [
            (doc["doc_id"], float(score))
            for doc, score in zip(self.corpus, scores)
        ]
        return sorted(ranked, key=lambda x: x[1], reverse=True)


# =========================
# TF-IDF
# =========================
class TFIDFRetriever:
    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
        self.doc_matrix = self.vectorizer.fit_transform([doc["text"] for doc in corpus])

    def score_query(self, query: str) -> list[tuple[str, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().ravel()

        ranked = [
            (doc["doc_id"], float(score))
            for doc, score in zip(self.corpus, scores)
        ]
        return sorted(ranked, key=lambda x: x[1], reverse=True)


# =========================
# QUERY LIKELIHOOD (Dirichlet smoothing)
# =========================
class QueryLikelihoodRetriever:
    def __init__(self, corpus: list[dict], mu: float = 1000.0):
        self.corpus = corpus
        self.mu = mu

        self.doc_tokens = [tokenize(doc["text"]) for doc in corpus]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]

        self.collection_tf = Counter()
        self.collection_length = 0
        for tokens in self.doc_tokens:
            self.collection_tf.update(tokens)
            self.collection_length += len(tokens)

    def _collection_prob(self, term: str) -> float:
        if self.collection_length == 0:
            return 0.0
        return self.collection_tf.get(term, 0) / self.collection_length

    def score_query(self, query: str) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scores = []

        for doc, tf, dl in zip(self.corpus, self.term_freqs, self.doc_lengths):
            log_prob = 0.0
            for term in q_terms:
                p_wc = self._collection_prob(term)
                numerator = tf.get(term, 0) + self.mu * p_wc
                denominator = dl + self.mu

                if numerator <= 0 or denominator <= 0:
                    # term unseen everywhere -> effectively impossible
                    log_prob += -1e9
                else:
                    log_prob += math.log(numerator / denominator)

            scores.append((doc["doc_id"], log_prob))

        return sorted(scores, key=lambda x: x[1], reverse=True)


# =========================
# RETRIEVAL FACTORY
# =========================
def build_lexical_retriever(corpus: list[dict], lexical_method: str):
    if lexical_method == "bm25":
        return BM25Retriever(corpus)
    if lexical_method == "tfidf":
        return TFIDFRetriever(corpus)
    if lexical_method == "query_likelihood":
        return QueryLikelihoodRetriever(corpus)
    raise ValueError(f"Unsupported lexical method: {lexical_method}")


# =========================
# EVALUATION
# =========================
def evaluate_one_query(
    query: str,
    query_type: str,
    graded_relevance: dict[str, int],
    retriever,
    experiment: dict,
) -> dict:
    ranked_pairs = retriever.score_query(query)
    ranked_docs = [doc_id for doc_id, _score in ranked_pairs]

    relevant_set = {doc for doc, rel in graded_relevance.items() if rel > 0}
    k = experiment["top_k_docs"]

    return {
        "experiment_name": experiment["name"],
        "experiment_group": experiment["experiment_group"],
        "representation": experiment["representation"],
        "retrieval_type": experiment["retrieval_type"],
        "lexical_method": experiment["lexical_method"],
        "query_type": query_type,
        "query": query,
        "relevant_docs": sorted(list(relevant_set)),
        "graded_relevance": graded_relevance,
        "ranked_docs": ranked_docs[:k],
        "full_ranked_docs": ranked_docs,
        "precision@5": precision_at_k(ranked_docs, relevant_set, k=k),
        "recall@5": recall_at_k(ranked_docs, relevant_set, k=k),
        "ndcg@5": ndcg_at_k(ranked_docs, graded_relevance, k=k),
        "precision@R": precision_at_r(ranked_docs, relevant_set),
    }


def summarize(rows: list[dict]) -> dict:
    return {
        "n_queries": len(rows),
        "precision@5": mean(r["precision@5"] for r in rows) if rows else 0.0,
        "recall@5": mean(r["recall@5"] for r in rows) if rows else 0.0,
        "ndcg@5": mean(r["ndcg@5"] for r in rows) if rows else 0.0,
        "precision@R": mean(r["precision@R"] for r in rows) if rows else 0.0,
    }


def summarize_by_type(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["query_type"]].append(r)

    out = {}
    for query_type in QUERY_TYPE_ORDER:
        if grouped.get(query_type):
            out[query_type] = summarize(grouped[query_type])

    for query_type, q_rows in grouped.items():
        if query_type not in out:
            out[query_type] = summarize(q_rows)

    return out


def print_summary_block(title: str, stats: dict):
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


def print_experiment_table(summary_rows: list[dict]):
    print("\n=== TOP 5 LEXICAL EXPERIMENTS (by nDCG@5) ===")
    header = (
        f"{'experiment':<38}"
        f"{'n':>4}"
        f"{'p@5':>10}"
        f"{'r@5':>10}"
        f"{'ndcg@5':>12}"
        f"{'p@R':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['experiment_name']:<38}"
            f"{row['n_queries']:>4}"
            f"{row['precision@5']:>10.4f}"
            f"{row['recall@5']:>10.4f}"
            f"{row['ndcg@5']:>12.4f}"
            f"{row['precision@R']:>10.4f}"
        )


def print_by_type_table(by_type_rows: list[dict]):
    print("\n=== TOP 5 LEXICAL EXPERIMENTS BY QUERY TYPE (by nDCG@5) ===")

    header = (
        f"{'query_type':<12}"
        f"{'experiment':<38}"
        f"{'n':>4}"
        f"{'p@5':>10}"
        f"{'r@5':>10}"
        f"{'ndcg@5':>12}"
        f"{'p@R':>10}"
    )
    print(header)
    print("-" * len(header))

    grouped = defaultdict(list)
    for row in by_type_rows:
        grouped[row["query_type"]].append(row)

    for query_type in QUERY_TYPE_ORDER:
        rows = grouped.get(query_type, [])
        if not rows:
            continue

        rows = sorted(rows, key=lambda x: x["ndcg@5"], reverse=True)

        for i, row in enumerate(rows):
            query_type_label = query_type if i == 0 else ""
            print(
                f"{query_type_label:<12}"
                f"{row['experiment_name']:<38}"
                f"{row['n_queries']:>4}"
                f"{row['precision@5']:>10.4f}"
                f"{row['recall@5']:>10.4f}"
                f"{row['ndcg@5']:>12.4f}"
                f"{row['precision@R']:>10.4f}"
            )

        print("-" * len(header))


def print_timing_table(timing_rows: list[dict]):
    print("\n=== COMPUTATIONAL TIME BY EXPERIMENT ===")
    header = (
        f"{'experiment':<38}"
        f"{'representation':<18}"
        f"{'method':<18}"
        f"{'queries':>8}"
        f"{'seconds':>12}"
        f"{'sec/query':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in timing_rows:
        print(
            f"{row['experiment_name']:<38}"
            f"{row['representation']:<18}"
            f"{row['lexical_method']:<18}"
            f"{row['n_queries']:>8}"
            f"{row['elapsed_seconds']:>12.4f}"
            f"{row['seconds_per_query']:>12.4f}"
        )


# =========================
# MAIN
# =========================
def main():
    total_start_time = time.perf_counter()

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    queries = load_ground_truth(GROUND_TRUTH_PATH)
    query_type_counts = validate_query_type_counts(queries)

    all_per_query_rows = []
    all_summary_rows = []
    all_by_type_rows = []
    all_timing_rows = []

    # cache corpora/retrievers by representation + lexical method
    corpus_cache = {}
    retriever_cache = {}

    for experiment in LEXICAL_EXPERIMENTS:
        experiment_start_time = time.perf_counter()

        representation = experiment["representation"]
        lexical_method = experiment["lexical_method"]

        corpus_key = representation
        retriever_key = (representation, lexical_method)

        if corpus_key not in corpus_cache:
            corpus_cache[corpus_key] = load_corpus(representation)

        if retriever_key not in retriever_cache:
            retriever_cache[retriever_key] = build_lexical_retriever(
                corpus=corpus_cache[corpus_key],
                lexical_method=lexical_method
            )

        retriever = retriever_cache[retriever_key]

        experiment_rows = []
        for item in queries:
            row = evaluate_one_query(
                query=item["query"],
                query_type=item["query_type"],
                graded_relevance=item["graded_relevance"],
                retriever=retriever,
                experiment=experiment,
            )
            experiment_rows.append(row)
            all_per_query_rows.append(row)

        overall = summarize(experiment_rows)
        overall_row = {
            "experiment_name": experiment["name"],
            "experiment_group": experiment["experiment_group"],
            "representation": experiment["representation"],
            "lexical_method": experiment["lexical_method"],
            **overall,
        }
        all_summary_rows.append(overall_row)

        by_type = summarize_by_type(experiment_rows)
        for query_type, stats in by_type.items():
            all_by_type_rows.append({
                "experiment_name": experiment["name"],
                "experiment_group": experiment["experiment_group"],
                "representation": experiment["representation"],
                "lexical_method": experiment["lexical_method"],
                "query_type": query_type,
                **stats,
            })

        experiment_elapsed = time.perf_counter() - experiment_start_time
        all_timing_rows.append({
            "experiment_name": experiment["name"],
            "experiment_group": experiment["experiment_group"],
            "representation": experiment["representation"],
            "lexical_method": experiment["lexical_method"],
            "n_queries": len(queries),
            "elapsed_seconds": experiment_elapsed,
            "seconds_per_query": experiment_elapsed / len(queries) if queries else 0.0,
        })

    total_elapsed = time.perf_counter() - total_start_time

    timing_summary = {
        "total_experiments": len(LEXICAL_EXPERIMENTS),
        "total_queries": len(queries),
        "total_elapsed_seconds": total_elapsed,
        "average_seconds_per_experiment": (
            total_elapsed / len(LEXICAL_EXPERIMENTS) if LEXICAL_EXPERIMENTS else 0.0
        ),
        "average_seconds_per_query_overall": (
            total_elapsed / (len(LEXICAL_EXPERIMENTS) * len(queries))
            if LEXICAL_EXPERIMENTS and queries else 0.0
        ),
    }

    # -------------------------
    # Save JSON
    # -------------------------
    with open(OUTPUT_DIR / "results_per_query.json", "w", encoding="utf-8") as f:
        json.dump(all_per_query_rows, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "summary_overall.json", "w", encoding="utf-8") as f:
        json.dump(all_summary_rows, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "summary_by_type.json", "w", encoding="utf-8") as f:
        json.dump(all_by_type_rows, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "query_type_counts.json", "w", encoding="utf-8") as f:
        json.dump(query_type_counts, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "timing_per_experiment.json", "w", encoding="utf-8") as f:
        json.dump(all_timing_rows, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "timing_summary.json", "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Save Excel
    # -------------------------
    df_per_query = pd.DataFrame(all_per_query_rows)
    df_summary = pd.DataFrame(all_summary_rows)
    df_by_type = pd.DataFrame(all_by_type_rows)
    df_query_type_counts = pd.DataFrame(
        [{"query_type": k, "count": v} for k, v in query_type_counts.items()]
    )
    df_timing = pd.DataFrame(all_timing_rows)
    df_timing_summary = pd.DataFrame([timing_summary])

    df_by_type_pivot = df_by_type.pivot_table(
        index=["experiment_name", "representation", "lexical_method"],
        columns="query_type",
        values=["precision@5", "recall@5", "ndcg@5", "precision@R"]
    ).reset_index()

    # Flatten MultiIndex columns so Excel export works with index=False
    flat_columns = []
    for col in df_by_type_pivot.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if x not in ("", None)]
            flat_columns.append("_".join(parts))
        else:
            flat_columns.append(str(col))

    df_by_type_pivot.columns = flat_columns

    excel_path = OUTPUT_DIR / "lexical_experiment_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary_overall", index=False)
        df_by_type.to_excel(writer, sheet_name="summary_by_type", index=False)
        df_by_type_pivot.to_excel(writer, sheet_name="summary_by_type_pivot", index=False)
        df_per_query.to_excel(writer, sheet_name="per_query", index=False)
        df_query_type_counts.to_excel(writer, sheet_name="query_type_counts", index=False)
        df_timing.to_excel(writer, sheet_name="timing_per_experiment", index=False)
        df_timing_summary.to_excel(writer, sheet_name="timing_summary", index=False)

    
    # -------------------------
    # Console output
    # -------------------------
    top_summary = sorted(
        all_summary_rows,
        key=lambda x: x["ndcg@5"],
        reverse=True
    )[:5]

    top_by_type_rows = []

    grouped = defaultdict(list)
    for row in all_by_type_rows:
        grouped[row["query_type"]].append(row)

    for qtype, rows in grouped.items():
        top_rows = sorted(
            rows,
            key=lambda x: x["ndcg@5"],
            reverse=True
        )[:5]
        top_by_type_rows.extend(top_rows)

    print_experiment_table(top_summary)
    print_by_type_table(top_by_type_rows)

    print("\n=== TOTAL COMPUTATIONAL TIME ===")
    print(f"Total experiments: {timing_summary['total_experiments']}")
    print(f"Total queries: {timing_summary['total_queries']}")
    print(f"Total elapsed seconds: {timing_summary['total_elapsed_seconds']:.4f}")
    print(f"Average seconds per experiment: {timing_summary['average_seconds_per_experiment']:.4f}")
    print(f"Average seconds per query overall: {timing_summary['average_seconds_per_query_overall']:.4f}")

    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"Excel file: {excel_path}")


if __name__ == "__main__":
    main()