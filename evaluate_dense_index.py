import json
import math
import os
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean
import random

import chromadb
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

# =========================
# CONFIG
# =========================
BASE_DIR = Path("Vensim_Models")
GROUND_TRUTH_PATH = Path("evaluation") / "ground_truth.json"
OUTPUT_DIR = Path("evaluation") / "dense"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Persistent directory for Chroma indexes / embeddings
EMBEDDINGS_DIR = Path("embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# If True, existing Chroma collections are deleted and rebuilt
RECOMPUTE_EMBEDDINGS = os.getenv("RECOMPUTE_EMBEDDINGS", "false").strip().lower() in {"1", "true", "yes", "y"}

CHROMA_BATCH_SIZE = 256

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

CHUNK_SIZES = [800, 2000, 4000]
CHUNK_OVERLAPS = [120, 200, 300]

EMBEDDING_MODELS = [
    {
        "backend": "local",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "label": "minilm",
    },
    {
        "backend": "local",
        "model": "BAAI/bge-base-en-v1.5",
        "label": "bge_base",
    },
    {
        "backend": "openai",
        "model": "text-embedding-3-large",
        "label": "openai_large",
    },
]

RETRIEVAL_PIPELINES = [
    "dense_only",
    "dense_rerank",
    "dense_llm_rerank",
]

TOP_K_CHUNKS = 20
TOP_K_DOCS = 5
QUERY_TYPE_ORDER = ["direct", "abstract", "ambiguous"]

DOC_SCORE_MODE = "weighted_sum"
DOC_SCORE_ALPHA = 0.7

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

LLM_RERANK_MODEL = os.getenv(
    "LLM_RERANK_MODEL",
    "gpt-4o-mini"
)

RERANK_TOP_N = 20
LLM_RERANK_TOP_N = 10


# =========================
# EXPERIMENT MANIFEST
# =========================
def build_dense_experiments():
    experiments = []

    for representation, chunk_size, chunk_overlap, emb, retrieval in product(
        REPRESENTATIONS,
        CHUNK_SIZES,
        CHUNK_OVERLAPS,
        EMBEDDING_MODELS,
        RETRIEVAL_PIPELINES,
    ):
        exp_name = (
            f"{representation}"
            f"__cs{chunk_size}"
            f"__ov{chunk_overlap}"
            f"__{emb['label']}"
            f"__{retrieval}"
        )

        experiments.append({
            "name": exp_name,
            "experiment_group": "dense",
            "representation": representation,
            "retrieval_type": retrieval,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_backend": emb["backend"],
            "embedding_model": emb["model"],
            "embedding_label": emb["label"],
            "top_k_chunks": TOP_K_CHUNKS,
            "top_k_docs": TOP_K_DOCS,
            "doc_score_mode": DOC_SCORE_MODE,
        })

    return experiments


DENSE_EXPERIMENTS = build_dense_experiments()


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
    if representation == "xml":
        return read_text_file(path)

    if representation in {"json", "json_narratives"}:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    raise ValueError(f"Unsupported representation: {representation}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap

    return chunks


def load_chunked_corpus(representation: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
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
        raw = read_representation_document(path, representation)
        chunks = chunk_text(raw, chunk_size, chunk_overlap)

        doc_id = normalize_doc_name(path.name)

        for i, chunk in enumerate(chunks):
            corpus.append({
                "doc_name": path.name,
                "doc_id": doc_id,
                "chunk_index": i,
                "chunk_id": f"{doc_id}::chunk::{i}",
                "path": str(path),
                "text": chunk,
            })

    return corpus


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
# EMBEDDING HELPERS
# =========================
class LocalEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return emb.astype(np.float32)


class OpenAIEmbedder:
    def __init__(self, model_name: str):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            vectors = [item.embedding for item in response.data]
            all_vectors.extend(vectors)

        arr = np.array(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[query],
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm


def build_embedder(backend: str, model_name: str):
    if backend == "local":
        return LocalEmbedder(model_name)
    if backend == "openai":
        return OpenAIEmbedder(model_name)
    raise ValueError(f"Unsupported embedding backend: {backend}")


# =========================
# RERANKERS
# =========================
class DenseCrossEncoderReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return hits

        pairs = [(query, h["text"]) for h in hits]
        scores = self.model.predict(pairs)

        reranked = []
        for h, score in zip(hits, scores):
            item = dict(h)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked


class DenseLLMReranker:
    def __init__(self, model_name: str = LLM_RERANK_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for dense_llm_rerank.")
        self.model_name = str(model_name)
        self.client = OpenAI(api_key=api_key)

    def _score_one(self, query: str, text: str) -> float:
        query = str(query) if query is not None else ""
        text = str(text) if text is not None else ""

        # Evita payloads demasiado grandes o caracteres raros inesperados
        text = text[:12000]

        system_msg = (
            "You are an expert evaluator in simulation model discovery. "
            "Score the relevance of the chunk for the query from 0 to 100. "
            "Return only one integer."
        )

        user_msg = f"Query:\n{query}\n\nChunk:\n{text}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=5,
            )
        except Exception as e:
            print("\n[LLM RERANK ERROR]")
            print("model:", repr(self.model_name))
            print("query type:", type(query), "len:", len(query))
            print("text type:", type(text), "len:", len(text))
            print("query preview:", repr(query[:300]))
            print("text preview:", repr(text[:500]))
            raise e

        raw = (response.choices[0].message.content or "").strip()

        try:
            return float(raw)
        except Exception:
            cleaned = "".join(ch for ch in raw if ch.isdigit() or ch in ".-")
            return float(cleaned) if cleaned else 0.0

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        if not hits:
            return hits

        reranked = []
        for h in hits:
            score = self._score_one( query=query, text=h.get("text", ""))
            time.sleep(random.uniform(0.1, 0.3))  # To avoid hitting rate limits
            item = dict(h)
            item["llm_rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["llm_rerank_score"], reverse=True)
        return reranked


# =========================
# CHROMA INDEX HELPERS
# =========================
def get_chroma_client():
    return chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))


def make_collection_name(representation: str, chunk_size: int, chunk_overlap: int, embedding_label: str) -> str:
    return f"{representation}__cs{chunk_size}__ov{chunk_overlap}__{embedding_label}"


def _batch_iter(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def build_or_load_chroma_index(
    corpus: list[dict],
    embedder,
    representation: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_label: str,
    recompute_embeddings: bool = RECOMPUTE_EMBEDDINGS,
):
    client = get_chroma_client()
    collection_name = make_collection_name(
        representation=representation,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_label=embedding_label,
    )

    existing_collections = {c.name for c in client.list_collections()}

    if recompute_embeddings and collection_name in existing_collections:
        client.delete_collection(collection_name)
        existing_collections.remove(collection_name)

    reused_existing_index = False
    embedding_build_seconds = 0.0

    if collection_name in existing_collections:
        collection = client.get_collection(collection_name)
        if collection.count() == len(corpus):
            reused_existing_index = True
            return {
                "collection": collection,
                "collection_name": collection_name,
                "embedding_build_seconds": embedding_build_seconds,
                "reused_existing_index": reused_existing_index,
            }
        else:
            client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    build_start = time.perf_counter()

    ids = [c["chunk_id"] for c in corpus]
    documents = [c["text"] for c in corpus]
    metadatas = [
        {
            "doc_name": c["doc_name"],
            "doc_id": c["doc_id"],
            "chunk_index": int(c["chunk_index"]),
            "chunk_id": c["chunk_id"],
            "path": c["path"],
        }
        for c in corpus
    ]

    for batch_ids, batch_docs, batch_meta in zip(
        _batch_iter(ids, CHROMA_BATCH_SIZE),
        _batch_iter(documents, CHROMA_BATCH_SIZE),
        _batch_iter(metadatas, CHROMA_BATCH_SIZE),
    ):
        batch_embeddings = embedder.embed_texts(batch_docs)
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=batch_embeddings.tolist(),
        )

    embedding_build_seconds = time.perf_counter() - build_start

    return {
        "collection": collection,
        "collection_name": collection_name,
        "embedding_build_seconds": embedding_build_seconds,
        "reused_existing_index": reused_existing_index,
    }


# =========================
# INDEX / RETRIEVAL
# =========================
def aggregate_hits_to_docs(
    hits: list[dict],
    score_field: str,
    top_k_docs: int,
    mode: str = "weighted_sum",
    alpha: float = DOC_SCORE_ALPHA,
) -> list[tuple[str, float]]:
    grouped = defaultdict(list)
    for h in hits:
        grouped[h["doc_id"]].append(h)

    doc_scores = []
    for doc_id, doc_hits in grouped.items():
        doc_hits = sorted(doc_hits, key=lambda x: x[score_field], reverse=True)

        if mode == "weighted_sum":
            score = 0.0
            for i, h in enumerate(doc_hits):
                score += (alpha ** i) * float(h[score_field])
        elif mode == "best":
            score = float(doc_hits[0][score_field])
        else:
            raise ValueError(f"Unsupported doc score mode: {mode}")

        doc_scores.append((doc_id, score))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k_docs]


def retrieve_dense_only(query: str, collection, embedder, experiment: dict) -> dict:
    query_emb = embedder.embed_query(query)

    result = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=experiment["top_k_chunks"],
        include=["documents", "metadatas", "distances"],
    )

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    hits = []
    for doc_text, meta, dist in zip(documents, metadatas, distances):
        item = {
            "doc_name": meta["doc_name"],
            "doc_id": meta["doc_id"],
            "chunk_index": meta["chunk_index"],
            "chunk_id": meta["chunk_id"],
            "path": meta["path"],
            "text": doc_text,
            "dense_score": float(1.0 - dist),
            "distance": float(dist),
        }
        hits.append(item)

    doc_ranking = aggregate_hits_to_docs(
        hits=hits,
        score_field="dense_score",
        top_k_docs=experiment["top_k_docs"],
        mode=experiment["doc_score_mode"],
    )

    return {
        "hits": hits,
        "doc_ranking": doc_ranking,
    }


def retrieve_dense_rerank(query: str, collection, embedder, reranker, experiment: dict) -> dict:
    initial = retrieve_dense_only(query, collection, embedder, experiment)
    hits = initial["hits"][:RERANK_TOP_N]

    reranked_hits = reranker.rerank(query, hits)

    doc_ranking = aggregate_hits_to_docs(
        hits=reranked_hits,
        score_field="rerank_score",
        top_k_docs=experiment["top_k_docs"],
        mode=experiment["doc_score_mode"],
    )

    return {
        "hits": reranked_hits,
        "doc_ranking": doc_ranking,
    }


def retrieve_dense_llm_rerank(query: str, collection, embedder, reranker, experiment: dict) -> dict:
    initial = retrieve_dense_only(query, collection, embedder, experiment)
    hits = initial["hits"][:LLM_RERANK_TOP_N]

    reranked_hits = reranker.rerank(query, hits)

    doc_ranking = aggregate_hits_to_docs(
        hits=reranked_hits,
        score_field="llm_rerank_score",
        top_k_docs=experiment["top_k_docs"],
        mode=experiment["doc_score_mode"],
    )

    return {
        "hits": reranked_hits,
        "doc_ranking": doc_ranking,
    }


# =========================
# EVALUATION
# =========================
def evaluate_one_query(
    query: str,
    query_type: str,
    graded_relevance: dict[str, int],
    retrieval_output: dict,
    experiment: dict,
) -> dict:
    ranked_docs = [doc_id for doc_id, _score in retrieval_output["doc_ranking"]]

    relevant_set = {doc for doc, rel in graded_relevance.items() if rel > 0}
    k = experiment["top_k_docs"]

    return {
        "experiment_name": experiment["name"],
        "experiment_group": experiment["experiment_group"],
        "representation": experiment["representation"],
        "retrieval_type": experiment["retrieval_type"],
        "chunk_size": experiment["chunk_size"],
        "chunk_overlap": experiment["chunk_overlap"],
        "embedding_backend": experiment["embedding_backend"],
        "embedding_model": experiment["embedding_model"],
        "embedding_label": experiment["embedding_label"],
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


# =========================
# CONSOLE OUTPUT
# =========================
def print_experiment_table(summary_rows: list[dict]):
    print("\n=== TOP 5 DENSE EXPERIMENTS (by nDCG@5) ===")
    header = (
        f"{'experiment':<70}"
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
            f"{row['experiment_name']:<70}"
            f"{row['n_queries']:>4}"
            f"{row['precision@5']:>10.4f}"
            f"{row['recall@5']:>10.4f}"
            f"{row['ndcg@5']:>12.4f}"
            f"{row['precision@R']:>10.4f}"
        )


def print_by_type_table(by_type_rows: list[dict]):
    print("\n=== TOP 5 DENSE EXPERIMENTS BY QUERY TYPE (by nDCG@5) ===")

    header = (
        f"{'query_type':<12}"
        f"{'experiment':<70}"
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
                f"{row['experiment_name']:<70}"
                f"{row['n_queries']:>4}"
                f"{row['precision@5']:>10.4f}"
                f"{row['recall@5']:>10.4f}"
                f"{row['ndcg@5']:>12.4f}"
                f"{row['precision@R']:>10.4f}"
            )

        print("-" * len(header))


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
    all_index_rows = []

    corpus_cache = {}
    embedder_cache = {}
    index_cache = {}
    cross_encoder_cache = {}
    llm_reranker_cache = {}

    for experiment in DENSE_EXPERIMENTS:
        experiment_start_time = time.perf_counter()

        representation = experiment["representation"]
        chunk_size = experiment["chunk_size"]
        chunk_overlap = experiment["chunk_overlap"]
        embedding_backend = experiment["embedding_backend"]
        embedding_model = experiment["embedding_model"]
        embedding_label = experiment["embedding_label"]
        retrieval_type = experiment["retrieval_type"]

        corpus_key = (representation, chunk_size, chunk_overlap)
        embedder_key = (embedding_backend, embedding_model)
        index_key = (representation, chunk_size, chunk_overlap, embedding_backend, embedding_label)

        if corpus_key not in corpus_cache:
            corpus_cache[corpus_key] = load_chunked_corpus(
                representation=representation,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        if embedder_key not in embedder_cache:
            embedder_cache[embedder_key] = build_embedder(
                backend=embedding_backend,
                model_name=embedding_model,
            )

        if index_key not in index_cache:
            index_info = build_or_load_chroma_index(
                corpus=corpus_cache[corpus_key],
                embedder=embedder_cache[embedder_key],
                representation=representation,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_label=embedding_label,
                recompute_embeddings=RECOMPUTE_EMBEDDINGS,
            )
            index_cache[index_key] = index_info

            all_index_rows.append({
                "collection_name": index_info["collection_name"],
                "representation": representation,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_backend": embedding_backend,
                "embedding_model": embedding_model,
                "embedding_label": embedding_label,
                "n_chunks": len(corpus_cache[corpus_key]),
                "embedding_build_seconds": index_info["embedding_build_seconds"],
                "reused_existing_index": index_info["reused_existing_index"],
                "recompute_embeddings": RECOMPUTE_EMBEDDINGS,
            })

        embedder = embedder_cache[embedder_key]
        collection = index_cache[index_key]["collection"]

        reranker = None
        if retrieval_type == "dense_rerank":
            if "cross_encoder" not in cross_encoder_cache:
                cross_encoder_cache["cross_encoder"] = DenseCrossEncoderReranker()
            reranker = cross_encoder_cache["cross_encoder"]

        if retrieval_type == "dense_llm_rerank":
            if "llm_reranker" not in llm_reranker_cache:
                llm_reranker_cache["llm_reranker"] = DenseLLMReranker()
            reranker = llm_reranker_cache["llm_reranker"]

        experiment_rows = []
        for item in queries:
            query = item["query"]

            if retrieval_type == "dense_only":
                retrieval_output = retrieve_dense_only(
                    query=query,
                    collection=collection,
                    embedder=embedder,
                    experiment=experiment,
                )
            elif retrieval_type == "dense_rerank":
                retrieval_output = retrieve_dense_rerank(
                    query=query,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    experiment=experiment,
                )
            elif retrieval_type == "dense_llm_rerank":
                retrieval_output = retrieve_dense_llm_rerank(
                    query=query,
                    collection=collection,
                    embedder=embedder,
                    reranker=reranker,
                    experiment=experiment,
                )
            else:
                raise ValueError(f"Unsupported retrieval_type: {retrieval_type}")

            row = evaluate_one_query(
                query=query,
                query_type=item["query_type"],
                graded_relevance=item["graded_relevance"],
                retrieval_output=retrieval_output,
                experiment=experiment,
            )
            experiment_rows.append(row)
            all_per_query_rows.append(row)

        overall = summarize(experiment_rows)
        overall_row = {
            "experiment_name": experiment["name"],
            "experiment_group": experiment["experiment_group"],
            "representation": experiment["representation"],
            "chunk_size": experiment["chunk_size"],
            "chunk_overlap": experiment["chunk_overlap"],
            "embedding_backend": experiment["embedding_backend"],
            "embedding_model": experiment["embedding_model"],
            "embedding_label": experiment["embedding_label"],
            "retrieval_type": experiment["retrieval_type"],
            **overall,
        }
        all_summary_rows.append(overall_row)

        by_type = summarize_by_type(experiment_rows)
        for query_type, stats in by_type.items():
            all_by_type_rows.append({
                "experiment_name": experiment["name"],
                "experiment_group": experiment["experiment_group"],
                "representation": experiment["representation"],
                "chunk_size": experiment["chunk_size"],
                "chunk_overlap": experiment["chunk_overlap"],
                "embedding_backend": experiment["embedding_backend"],
                "embedding_model": experiment["embedding_model"],
                "embedding_label": experiment["embedding_label"],
                "retrieval_type": experiment["retrieval_type"],
                "query_type": query_type,
                **stats,
            })

        experiment_elapsed = time.perf_counter() - experiment_start_time
        all_timing_rows.append({
            "experiment_name": experiment["name"],
            "experiment_group": experiment["experiment_group"],
            "representation": experiment["representation"],
            "chunk_size": experiment["chunk_size"],
            "chunk_overlap": experiment["chunk_overlap"],
            "embedding_backend": experiment["embedding_backend"],
            "embedding_model": experiment["embedding_model"],
            "embedding_label": experiment["embedding_label"],
            "retrieval_type": experiment["retrieval_type"],
            "n_queries": len(queries),
            "elapsed_seconds": experiment_elapsed,
            "seconds_per_query": experiment_elapsed / len(queries) if queries else 0.0,
            "collection_name": index_cache[index_key]["collection_name"],
            "embedding_build_seconds": index_cache[index_key]["embedding_build_seconds"],
            "reused_existing_index": index_cache[index_key]["reused_existing_index"],
            "recompute_embeddings": RECOMPUTE_EMBEDDINGS,
        })

    total_elapsed = time.perf_counter() - total_start_time

    timing_summary = {
        "total_experiments": len(DENSE_EXPERIMENTS),
        "total_queries": len(queries),
        "total_elapsed_seconds": total_elapsed,
        "average_seconds_per_experiment": (
            total_elapsed / len(DENSE_EXPERIMENTS) if DENSE_EXPERIMENTS else 0.0
        ),
        "average_seconds_per_query_overall": (
            total_elapsed / (len(DENSE_EXPERIMENTS) * len(queries))
            if DENSE_EXPERIMENTS and queries else 0.0
        ),
        "total_index_build_seconds": sum(r["embedding_build_seconds"] for r in all_index_rows),
        "total_indexes_built_or_loaded": len(all_index_rows),
        "recompute_embeddings": RECOMPUTE_EMBEDDINGS,
        "embeddings_dir": str(EMBEDDINGS_DIR),
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

    with open(OUTPUT_DIR / "index_build_log.json", "w", encoding="utf-8") as f:
        json.dump(all_index_rows, f, indent=2, ensure_ascii=False)

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
    df_index_log = pd.DataFrame(all_index_rows)

    df_by_type_pivot = df_by_type.pivot_table(
        index=[
            "experiment_name",
            "representation",
            "chunk_size",
            "chunk_overlap",
            "embedding_label",
            "retrieval_type",
        ],
        columns="query_type",
        values=["precision@5", "recall@5", "ndcg@5", "precision@R"]
    ).reset_index()

    flat_columns = []
    for col in df_by_type_pivot.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if x not in ("", None)]
            flat_columns.append("_".join(parts))
        else:
            flat_columns.append(str(col))
    df_by_type_pivot.columns = flat_columns

    excel_path = OUTPUT_DIR / "dense_experiment_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary_overall", index=False)
        df_by_type.to_excel(writer, sheet_name="summary_by_type", index=False)
        df_by_type_pivot.to_excel(writer, sheet_name="summary_by_type_pivot", index=False)
        df_per_query.to_excel(writer, sheet_name="per_query", index=False)
        df_query_type_counts.to_excel(writer, sheet_name="query_type_counts", index=False)
        df_timing.to_excel(writer, sheet_name="timing_per_experiment", index=False)
        df_timing_summary.to_excel(writer, sheet_name="timing_summary", index=False)
        df_index_log.to_excel(writer, sheet_name="index_build_log", index=False)

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


if __name__ == "__main__":
    main()