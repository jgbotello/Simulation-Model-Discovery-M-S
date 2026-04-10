import json
from pathlib import Path
from statistics import mean

try:
    import tiktoken
except ImportError:
    raise ImportError("Install tiktoken first: pip install tiktoken")

# =========================
# CONFIG
# =========================
BASE_DIR = Path("Vensim_Models")
GROUND_TRUTH_PATH = Path("evaluation") / "ground_truth.json"

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

REPRESENTATIONS = ["xml", "json", "json_narratives"]
CHUNK_SIZES = [800, 2000, 4000]
CHUNK_OVERLAPS = [120, 200, 300]

# Same as your reranker
LLM_RERANK_TOP_N = 10
LLM_RERANK_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"

# =========================
# PRICING
# Adjust these if prices change
# =========================
GPT4O_MINI_INPUT_COST_PER_1M = 0.15
GPT4O_MINI_OUTPUT_COST_PER_1M = 0.60
TEXT_EMBEDDING_3_LARGE_COST_PER_1M = 0.13

# Your reranker returns only one integer with max_tokens=5
ESTIMATED_OUTPUT_TOKENS_PER_CALL = 5

# Use real queries to estimate rerank prompt size
USE_ALL_QUERIES_FOR_PROMPT_AVG = True


# =========================
# HELPERS
# =========================
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

        for i, chunk in enumerate(chunks):
            corpus.append({
                "doc_name": path.name,
                "chunk_index": i,
                "text": chunk,
            })

    return corpus


def load_queries(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["query"] for item in data]


# =========================
# TOKEN COUNTING
# =========================
def get_token_encoder(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


ENCODER = get_token_encoder(LLM_RERANK_MODEL)


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def build_llm_messages(query: str, text: str) -> tuple[str, str]:
    query = str(query) if query is not None else ""
    text = str(text) if text is not None else ""
    text = text[:12000]

    system_msg = (
        "You are an expert evaluator in simulation model discovery. "
        "Score the relevance of the chunk for the query from 0 to 100. "
        "Return only one integer."
    )
    user_msg = f"Query:\n{query}\n\nChunk:\n{text}"
    return system_msg, user_msg


def estimate_chat_tokens(query: str, text: str) -> int:
    # Approximation for chat formatting overhead
    system_msg, user_msg = build_llm_messages(query, text)
    return count_tokens(system_msg) + count_tokens(user_msg) + 10


def estimate_embedding_tokens(text: str) -> int:
    # Same tokenizer used as approximation for embeddings cost analysis
    return count_tokens(text)


# =========================
# COST ANALYSIS
# =========================
def format_usd(value: float) -> str:
    return f"${value:,.6f}"


def main():
    queries = load_queries(GROUND_TRUTH_PATH)
    n_queries = len(queries)

    print("\n=== OPENAI COST ANALYSIS ===")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"LLM reranker model: {LLM_RERANK_MODEL}")
    print(f"Queries loaded: {n_queries}")
    print(f"LLM_RERANK_TOP_N: {LLM_RERANK_TOP_N}")

    grand_total_embedding_tokens = 0
    grand_total_llm_input_tokens = 0
    grand_total_llm_output_tokens = 0
    grand_total_llm_calls = 0

    for representation in REPRESENTATIONS:
        print(f"\n{'=' * 90}")
        print(f"REPRESENTATION: {representation}")
        print(f"{'=' * 90}")

        for chunk_size in CHUNK_SIZES:
            for overlap in CHUNK_OVERLAPS:
                corpus = load_chunked_corpus(
                    representation=representation,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                )

                n_chunks = len(corpus)

                # ---------- Embedding tokens ----------
                embedding_token_counts = [estimate_embedding_tokens(item["text"]) for item in corpus]
                avg_embedding_tokens = mean(embedding_token_counts) if embedding_token_counts else 0
                total_embedding_tokens = sum(embedding_token_counts)
                embedding_cost = (
                    total_embedding_tokens / 1_000_000
                ) * TEXT_EMBEDDING_3_LARGE_COST_PER_1M

                # ---------- LLM rerank tokens ----------
                llm_prompt_token_counts = []

                if USE_ALL_QUERIES_FOR_PROMPT_AVG:
                    # More faithful estimate: average over all queries x all chunks
                    for q in queries:
                        for item in corpus:
                            llm_prompt_token_counts.append(
                                estimate_chat_tokens(q, item["text"])
                            )
                else:
                    # Faster estimate: first query only
                    q = queries[0]
                    for item in corpus:
                        llm_prompt_token_counts.append(
                            estimate_chat_tokens(q, item["text"])
                        )

                avg_llm_input_tokens = mean(llm_prompt_token_counts) if llm_prompt_token_counts else 0

                # Per experiment with this config:
                # each query reranks top 10 chunks
                llm_calls = n_queries * LLM_RERANK_TOP_N
                total_llm_input_tokens = int(avg_llm_input_tokens * llm_calls)
                total_llm_output_tokens = ESTIMATED_OUTPUT_TOKENS_PER_CALL * llm_calls

                llm_input_cost = (
                    total_llm_input_tokens / 1_000_000
                ) * GPT4O_MINI_INPUT_COST_PER_1M
                llm_output_cost = (
                    total_llm_output_tokens / 1_000_000
                ) * GPT4O_MINI_OUTPUT_COST_PER_1M
                llm_total_cost = llm_input_cost + llm_output_cost

                total_cost = embedding_cost + llm_total_cost

                grand_total_embedding_tokens += total_embedding_tokens
                grand_total_llm_input_tokens += total_llm_input_tokens
                grand_total_llm_output_tokens += total_llm_output_tokens
                grand_total_llm_calls += llm_calls

                print(
                    f"\n[{representation} | cs{chunk_size} | ov{overlap}]"
                )
                print(f"  Files/chunks indexed: {n_chunks}")
                print(f"  Avg embedding tokens/chunk: {avg_embedding_tokens:.2f}")
                print(f"  Total embedding tokens: {total_embedding_tokens:,}")
                print(f"  Embedding cost ({EMBEDDING_MODEL}): {format_usd(embedding_cost)}")

                print(f"  Avg LLM input tokens/call: {avg_llm_input_tokens:.2f}")
                print(f"  LLM calls (queries * topN): {llm_calls:,}")
                print(f"  Total LLM input tokens: {total_llm_input_tokens:,}")
                print(f"  Total LLM output tokens: {total_llm_output_tokens:,}")
                print(f"  LLM input cost ({LLM_RERANK_MODEL}): {format_usd(llm_input_cost)}")
                print(f"  LLM output cost ({LLM_RERANK_MODEL}): {format_usd(llm_output_cost)}")
                print(f"  Total LLM rerank cost: {format_usd(llm_total_cost)}")

                print(f"  TOTAL OPENAI COST (embeddings + rerank): {format_usd(total_cost)}")

    grand_embedding_cost = (
        grand_total_embedding_tokens / 1_000_000
    ) * TEXT_EMBEDDING_3_LARGE_COST_PER_1M
    grand_llm_input_cost = (
        grand_total_llm_input_tokens / 1_000_000
    ) * GPT4O_MINI_INPUT_COST_PER_1M
    grand_llm_output_cost = (
        grand_total_llm_output_tokens / 1_000_000
    ) * GPT4O_MINI_OUTPUT_COST_PER_1M
    grand_llm_cost = grand_llm_input_cost + grand_llm_output_cost
    grand_total_cost = grand_embedding_cost + grand_llm_cost

    print(f"\n{'#' * 90}")
    print("GRAND TOTAL ACROSS ALL REPRESENTATIONS / CHUNK SIZES / OVERLAPS")
    print(f"{'#' * 90}")
    print(f"Total embedding tokens: {grand_total_embedding_tokens:,}")
    print(f"Total embedding cost: {format_usd(grand_embedding_cost)}")
    print(f"Total LLM calls: {grand_total_llm_calls:,}")
    print(f"Total LLM input tokens: {grand_total_llm_input_tokens:,}")
    print(f"Total LLM output tokens: {grand_total_llm_output_tokens:,}")
    print(f"Total LLM input cost: {format_usd(grand_llm_input_cost)}")
    print(f"Total LLM output cost: {format_usd(grand_llm_output_cost)}")
    print(f"Total LLM rerank cost: {format_usd(grand_llm_cost)}")
    print(f"GRAND TOTAL OPENAI COST: {format_usd(grand_total_cost)}")


if __name__ == "__main__":
    main()