from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.prompts import base
from mcp.types import TextContent
from mcp import types
from PIL import Image as PILImage
import math
import sys
import os
import json
import faiss
import numpy as np
from pathlib import Path
import requests
from markitdown import MarkItDown
import time
from models import AddInput, AddOutput, SqrtInput, SqrtOutput, StringsToIntsInput, StringsToIntsOutput, ExpSumInput, ExpSumOutput, TemperatureInput, TemperatureOutput
from PIL import Image as PILImage
from tqdm import tqdm
import hashlib
from dotenv import load_dotenv


mcp = FastMCP("Calculator")

# Simple in-process cache for FAISS index and metadata
_INDEX_OBJ = None
_META_OBJ = None
_INDEX_MTIME = None
_META_MTIME = None

def _load_index_and_metadata_cached():
    global _INDEX_OBJ, _META_OBJ, _INDEX_MTIME, _META_MTIME
    index_path = ROOT / "faiss_index" / "index.bin"
    meta_path = ROOT / "faiss_index" / "metadata.json"
    try:
        idx_mtime = index_path.stat().st_mtime if index_path.exists() else None
        meta_mtime = meta_path.stat().st_mtime if meta_path.exists() else None
        reload_index = _INDEX_OBJ is None or _INDEX_MTIME != idx_mtime
        reload_meta = _META_OBJ is None or _META_MTIME != meta_mtime
        if reload_index and index_path.exists():
            _INDEX_OBJ = faiss.read_index(str(index_path))
            _INDEX_MTIME = idx_mtime
        if reload_meta and meta_path.exists():
            raw = json.loads(meta_path.read_text(encoding='utf-8'))
            _META_OBJ = raw['chunks'] if isinstance(raw, dict) and 'chunks' in raw else raw
            _META_MTIME = meta_mtime
    except Exception as e:
        mcp_log("ERROR", f"Cache load failed: {e}")
    return _INDEX_OBJ, _META_OBJ

# Load env and allow configurable embedding model
load_dotenv()
from openai import OpenAI as _OpenAI
_OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
_openai_client = _OpenAI()
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40#can be set up to 50
ROOT = Path(__file__).parent.resolve()
from rank_bm25 import BM25Okapi


def _load_synonyms() -> dict:
    try:
        return json.loads((ROOT / 'babycare_synonyms.json').read_text(encoding='utf-8'))
    except Exception:
        return {}


def _expand_query_with_synonyms(text: str) -> str:
    """Expand query by words (not characters) using a simple space split.
    Falls back gracefully if synonyms are missing.
    """
    syn = _load_synonyms()
    tokens = text.split()
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in syn:
            expanded.extend(syn[t])
    return " ".join(expanded)


# BM25 cache (rebuild if metadata mtime changes)
_BM25_OBJ = None
_BM25_MTIME = None
_BM25_SIZE = 0

def _get_bm25(metadata: list[dict]) -> BM25Okapi:
    global _BM25_OBJ, _BM25_MTIME, _BM25_SIZE
    meta_path = ROOT / 'faiss_index' / 'metadata.json'
    mtime = meta_path.stat().st_mtime if meta_path.exists() else None
    need_rebuild = (_BM25_OBJ is None) or (_BM25_MTIME != mtime) or (_BM25_SIZE != len(metadata))
    if need_rebuild:
        corpus = []
        for m in metadata:
            if 'text' in m:
                corpus.append(m['text'])
            elif 'chunk' in m:
                corpus.append(m['chunk'])
            else:
                corpus.append(str(m))
        tokenized_corpus = [doc.split() for doc in corpus]
        _BM25_OBJ = BM25Okapi(tokenized_corpus)
        _BM25_MTIME = mtime
        _BM25_SIZE = len(metadata)
    return _BM25_OBJ


def _bm25_search(expanded_query: str, metadata: list[dict], top_k: int = 20) -> dict[int, float]:
    bm25 = _get_bm25(metadata)
    scores = bm25.get_scores(expanded_query.split())
    idxs = np.argsort(scores)[::-1]
    # dedupe is implicit; select top_k positives
    out = {}
    for i in idxs:
        i = int(i)
        sc = float(scores[i])
        if sc <= 0:
            break
        out[i] = sc
        if len(out) >= top_k:
            break
    return out


def _rrf_fusion(bm25_indices: iter, vec_ranking: list[int], k: int = 60) -> list[int]:
    # bm25_indices is a set/dict keys of indices
    bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_indices)}
    vec_ranks = {idx: rank + 1 for rank, idx in enumerate(vec_ranking)}
    all_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())
    def score(doc_id: int) -> float:
        s = 0.0
        if doc_id in bm25_ranks:
            s += 1.0 / (bm25_ranks[doc_id] + k)
        if doc_id in vec_ranks:
            s += 1.0 / (vec_ranks[doc_id] + k)
        return s
    ranked = sorted(all_ids, key=lambda d: score(d), reverse=True)
    return ranked


def get_embedding(text: str) -> np.ndarray:
    """Get embedding with timeout and retry logic."""
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = _openai_client.embeddings.create(
                model=_OPENAI_EMBED_MODEL,
                input=text,
                timeout=30
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            if attempt == max_retries - 1:
                mcp_log("ERROR", f"Embedding failed after {max_retries} attempts: {e}")
                raise e
            mcp_log("WARN", f"Embedding attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(1)

from temperature_rules import extract_temperature

def _format_temp_range_as_both_units(min_v: float, max_v: float, unit: str) -> str:
    if unit.upper() == 'F':
        cmin = (min_v - 32) * 5/9
        cmax = (max_v - 32) * 5/9
        return f"{int(min_v)}–{int(max_v)}°F ({int(round(cmin))}–{int(round(cmax))}°C)"
    else:
        fmin = (min_v * 9/5) + 32
        fmax = (max_v * 9/5) + 32
        return f"{int(round(fmin))}–{int(round(fmax))}°F ({int(min_v)}–{int(max_v)}°C)"


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

def mcp_log(level: str, message: str) -> None:
    """Log a message to stderr to avoid interfering with JSON communication"""
    sys.stderr.write(f"{level}: {message}\n")
    sys.stderr.flush()

@mcp.tool()
def search_documents(query: str) -> list[str]:
    """Hybrid search (BM25+Vector) with query expansion and RRF fusion. Returns top snippets with source filenames."""
    ensure_faiss_ready()
    mcp_log("SEARCH", f"Query: {query}")
    try:
        # Load metadata and FAISS index (cached)
        index, metadata = _load_index_and_metadata_cached()
        if index is None or metadata is None:
            # Fallback to direct load once
            index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
            metadata_raw = json.loads((ROOT / "faiss_index" / "metadata.json").read_text(encoding='utf-8'))
            metadata = metadata_raw['chunks'] if isinstance(metadata_raw, dict) and 'chunks' in metadata_raw else metadata_raw

        # 1) Query expansion via local synonyms
        expanded = _expand_query_with_synonyms(query)

        # 2) BM25 over chunk texts
        bm25_scores = _bm25_search(expanded, metadata, top_k=20)

        # 3) Vector search over original query
        query_vec = get_embedding(query).reshape(1, -1)
        _D, I = index.search(query_vec, k=20)
        vec_ranking = [int(i) for i in I[0] if i < len(metadata)]

        # 4) RRF fusion
        fused = _rrf_fusion(bm25_scores.keys(), vec_ranking, k=60)

        # 4.5) Light reranker with context tagging
        def _context_tag(text: str) -> str:
            tl = text.lower()
            if any(w in tl for w in ["bath", "bathtub", "tub", "water temperature", "shower"]):
                return "bath"
            if any(w in tl for w in ["fever", "temperature of baby", "high temperature", "doctor", "sick"]):
                return "fever"
            if any(w in tl for w in ["room", "nursery", "sleep", "crib", "bedroom"]):
                return "room"
            return "general"

        ql = query.lower()
        query_pref = None
        if any(w in ql for w in ["bath", "bathtub", "tub", "water"]):
            query_pref = "bath"
        elif any(w in ql for w in ["fever", "sick", "ill", "temperature of baby"]):
            query_pref = "fever"
        elif any(w in ql for w in ["room", "sleep", "nursery", "bedroom"]):
            query_pref = "room"

        reranked = []
        for rank_pos, idx in enumerate(fused):
            if idx >= len(metadata):
                continue
            data = metadata[idx]
            text = data.get('text', data.get('chunk', '')) or ''
            tag = _context_tag(text)
            base = 1.0 / (1.0 + rank_pos)  # higher for earlier
            boost = 0.3 if (query_pref and tag == query_pref) else 0.0
            reranked.append((idx, base + boost, tag))
        reranked.sort(key=lambda x: x[1], reverse=True)

        # 5) Compose structured results with file name and chunk id, with optional temperature formatting
        top = reranked[:5]
        # Debug print ranked order for troubleshooting with chunk content
        try:
            dbg = []
            for (i, s, t) in top:
                chunk_text = metadata[i].get('text', metadata[i].get('chunk', ''))
                preview = (chunk_text[:100] + '...') if len(chunk_text) > 100 else chunk_text
                dbg.append({
                    'idx': i,
                    'score': round(s, 4),
                    'tag': t,
                    'source': metadata[i].get('doc_id', metadata[i].get('doc', '')),
                    'content_preview': preview
                })
            mcp_log("RANKED", f"Top reranked: {dbg}")
        except Exception:
            pass

        results: list[dict] = []
        sources = []
        for (idx, score, tag) in top:
            data = metadata[idx]
            chunk_text = data.get('text', data.get('chunk', ''))
            doc_name = data.get('doc_id', data.get('doc', ''))
            chunk_id = data.get('chunk_id', data.get('id', idx))

            entry = {
                'text': chunk_text,
                'context_tag': tag,
                'source': doc_name,
                'chunk_id': chunk_id
            }

            # If a clear temperature range is found, add a formatted_short field to help downstream LLM
            temps = extract_temperature(chunk_text)
            if temps:
                t = temps[0]
                entry['formatted_temp'] = _format_temp_range_as_both_units(t['min'], t['max'], t['unit'])

            results.append(entry)
            if doc_name:
                sources.append(doc_name)

        # Append de-duplicated sources separately for convenience
        unique_sources = []
        seen = set()
        for s in sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)

        payload = {
            'results': results,
            'sources': unique_sources
        }
        # Return as a single JSON string element (stable for MCP clients)
        return [json.dumps(payload, ensure_ascii=False)]
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        mcp_log("ERROR", f"Search failed: {str(e)}")
        mcp_log("ERROR", f"Traceback: {error_details}")
        return [f"ERROR: Failed to search: {str(e)} | Details: {error_details[:200]}"]

@mcp.tool()
def convert_temperature(input: TemperatureInput) -> TemperatureOutput:
    """
    Convert temperature between Celsius and Fahrenheit.

    Parameters:
    - input.value: the numeric temperature to convert
    - input.to_scale: target scale, either 'C' for Celsius or 'F' for Fahrenheit

    Returns:
    - Converted temperature
    """
    if input.to_scale.upper() == 'F':
        result = (input.value * 9/5) + 32
    elif input.to_scale.upper() == 'C':
        result = (input.value - 32) * 5/9
    else:
        raise ValueError("Invalid target scale. Use 'C' for Celsius or 'F' for Fahrenheit.")

    return TemperatureOutput(result=result)

@mcp.tool()
def add(input: AddInput) -> AddOutput:
    print("CALLED: add(AddInput) -> AddOutput")
    return AddOutput(result=input.a + input.b)

@mcp.tool()
def sqrt(input: SqrtInput) -> SqrtOutput:
    """Square root of a number"""
    print("CALLED: sqrt(SqrtInput) -> SqrtOutput")
    return SqrtOutput(result=input.a ** 0.5)

# subtraction tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    print("CALLED: subtract(a: int, b: int) -> int:")
    return int(a - b)

# multiplication tool
@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print("CALLED: multiply(a: int, b: int) -> int:")
    return int(a * b)

#  division tool
@mcp.tool()
def divide(a: int, b: int) -> float:
    """Divide two numbers"""
    print("CALLED: divide(a: int, b: int) -> float:")
    return float(a / b)

# power tool
@mcp.tool()
def power(a: int, b: int) -> int:
    """Power of two numbers"""
    print("CALLED: power(a: int, b: int) -> int:")
    return int(a ** b)


# cube root tool
@mcp.tool()
def cbrt(a: int) -> float:
    """Cube root of a number"""
    print("CALLED: cbrt(a: int) -> float:")
    return float(a ** (1/3))

# factorial tool
@mcp.tool()
def factorial(a: int) -> int:
    """factorial of a number"""
    print("CALLED: factorial(a: int) -> int:")
    return int(math.factorial(a))

# log tool
@mcp.tool()
def log(a: int) -> float:
    """log of a number"""
    print("CALLED: log(a: int) -> float:")
    return float(math.log(a))

# remainder tool
@mcp.tool()
def remainder(a: int, b: int) -> int:
    """remainder of two numbers divison"""
    print("CALLED: remainder(a: int, b: int) -> int:")
    return int(a % b)

# sin tool
@mcp.tool()
def sin(a: int) -> float:
    """sin of a number"""
    print("CALLED: sin(a: int) -> float:")
    return float(math.sin(a))

# cos tool
@mcp.tool()
def cos(a: int) -> float:
    """cos of a number"""
    print("CALLED: cos(a: int) -> float:")
    return float(math.cos(a))

# tan tool
@mcp.tool()
def tan(a: int) -> float:
    """tan of a number"""
    print("CALLED: tan(a: int) -> float:")
    return float(math.tan(a))

# mine tool
@mcp.tool()
def mine(a: int, b: int) -> int:
    """special mining tool"""
    print("CALLED: mine(a: int, b: int) -> int:")
    return int(a - b - b)

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """Create a thumbnail from an image"""
    print("CALLED: create_thumbnail(image_path: str) -> Image:")
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")

@mcp.tool()
def strings_to_chars_to_int(input: StringsToIntsInput) -> StringsToIntsOutput:
    """Return the ASCII values of the characters in a word"""
    print("CALLED: strings_to_chars_to_int(StringsToIntsInput) -> StringsToIntsOutput")
    ascii_values = [ord(char) for char in input.string]
    return StringsToIntsOutput(ascii_values=ascii_values)

@mcp.tool()
def int_list_to_exponential_sum(input: ExpSumInput) -> ExpSumOutput:
    """Return sum of exponentials of numbers in a list"""
    print("CALLED: int_list_to_exponential_sum(ExpSumInput) -> ExpSumOutput")
    result = sum(math.exp(i) for i in input.int_list)
    return ExpSumOutput(result=result)

@mcp.tool()
def fibonacci_numbers(n: int) -> list:
    """Return the first n Fibonacci Numbers"""
    print("CALLED: fibonacci_numbers(n: int) -> list:")
    if n <= 0:
        return []
    fib_sequence = [0, 1]
    for _ in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]

# DEFINE RESOURCES

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    print("CALLED: get_greeting(name: str) -> str:")
    return f"Hello, {name}!"


# DEFINE AVAILABLE PROMPTS
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
    print("CALLED: review_code(code: str) -> str:")


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]

def process_documents():
    """Process documents and create FAISS index"""
    mcp_log("INFO", "Indexing documents with MarkItDown...")
    ROOT = Path(__file__).parent.resolve()
    DOC_PATH = ROOT / "documents"
    INDEX_CACHE = ROOT / "faiss_index"
    INDEX_CACHE.mkdir(exist_ok=True)
    INDEX_FILE = INDEX_CACHE / "index.bin"
    METADATA_FILE = INDEX_CACHE / "metadata.json"
    CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"

    def file_hash(path):
        return hashlib.md5(Path(path).read_bytes()).hexdigest()

    CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
    index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None
    all_embeddings = []
    converter = MarkItDown()

    for file in DOC_PATH.glob("*.*"):
        fhash = file_hash(file)
        if file.name in CACHE_META and CACHE_META[file.name] == fhash:
            mcp_log("SKIP", f"Skipping unchanged file: {file.name}")
            continue

        mcp_log("PROC", f"Processing: {file.name}")
        try:
            result = converter.convert(str(file))
            markdown = result.text_content
            chunks = list(chunk_text(markdown))
            embeddings_for_file = []
            new_metadata = []
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file.name}")):
                embedding = get_embedding(chunk)
                embeddings_for_file.append(embedding)
                new_metadata.append({"doc": file.name, "chunk": chunk, "chunk_id": f"{file.stem}_{i}"})
            if embeddings_for_file:
                if index is None:
                    dim = len(embeddings_for_file[0])
                    index = faiss.IndexFlatL2(dim)
                index.add(np.stack(embeddings_for_file))
                metadata.extend(new_metadata)
            CACHE_META[file.name] = fhash
        except Exception as e:
            mcp_log("ERROR", f"Failed to process {file.name}: {e}")

    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    if index and index.ntotal > 0:
        faiss.write_index(index, str(INDEX_FILE))
        mcp_log("SUCCESS", "Saved FAISS index and metadata")
    else:
        mcp_log("WARN", "No new documents or updates to process.")

def ensure_faiss_ready():
    from pathlib import Path
    index_path = ROOT / "faiss_index" / "index.bin"
    meta_path = ROOT / "faiss_index" / "metadata.json"
    if not (index_path.exists() and meta_path.exists()):
        mcp_log("INFO", "Index not found — running process_documents()...")
        process_documents()
    else:
        mcp_log("INFO", "Index already exists. Skipping regeneration.")


if __name__ == "__main__":
    print("STARTING THE SERVER AT AMAZING LOCATION")



    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        mcp.run() # Run without transport for dev server
    else:
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(target=lambda: mcp.run(transport="stdio"))
        server_thread.daemon = True
        server_thread.start()

        # Wait a moment for the server to start
        time.sleep(2)

        # Process documents after server is running
        process_documents()

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
