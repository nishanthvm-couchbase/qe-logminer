#!/usr/bin/python3
"""
Text embeddings for near-duplicate failure dedup.

Two backends, chosen by EMBED_PROVIDER:

  local  (default) — sentence-transformers on the slave. No API cost, but needs the
                     lib (and a GPU for big models). Default model: MiniLM (384-dim).
  openai           — OpenAI embeddings API. No slave lib/GPU; tiny cost (~$0.02/1M
                     tokens for text-embedding-3-small). Recommended for accuracy at
                     negligible cost. Needs OPENAI_API_KEY.

Everything is best-effort and safe-fail: if the backend can't produce a vector
(lib missing, no key, network error), `available()`/`embed()` return False/None and
the caller falls back to the normal droid path — the feature can never break the
pipeline. Vectors are L2-normalized, so cosine similarity == dot product.

Enable in the pipeline with VECTOR_DEDUP=1 (see analyze_build.py).
  local :  pip install sentence-transformers
  openai:  export EMBED_PROVIDER=openai OPENAI_API_KEY=sk-...   (requests already a dep)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local").lower()

# Default model depends on provider.
_DEFAULT_MODEL = ("text-embedding-3-small" if EMBED_PROVIDER == "openai"
                  else "sentence-transformers/all-MiniLM-L6-v2")
EMBED_MODEL = os.environ.get("EMBED_MODEL", _DEFAULT_MODEL)

# Vector dimension. For openai/text-embedding-3-* the API can shrink to EMBED_DIM
# (512 is a good compact choice, 1536 = full 3-small). For local it must match the
# model. Keep the vector index `dimension` (create_vector_index.py) in sync.
_dim_env  = os.environ.get("EMBED_DIM")
EMBED_DIM = int(_dim_env) if _dim_env else (1536 if EMBED_PROVIDER == "openai" else 384)

EMBED_TRUST  = os.environ.get("EMBED_TRUST_REMOTE_CODE", "1") != "0"   # gte-* need this
EMBED_PROMPT = os.environ.get("EMBED_PROMPT", "")                      # optional instruct prefix

# Stored on each doc + used to filter candidates. Includes the dimension so vectors
# of a different model OR a different dimension are never compared (they're incomparable).
EMBED_TAG = f"{EMBED_MODEL}@{EMBED_DIM}"

_model = None
_state = None   # None = not tried, "ok" = usable, "off" = unavailable


def _l2norm(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v] if n > 0 else v


# --- local (sentence-transformers) ---

def _load_local():
    global _model, _state
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=EMBED_TRUST)
        _state = "ok"
        try:
            dim = _model.get_sentence_embedding_dimension()
        except Exception:
            dim = None
        if dim and dim != EMBED_DIM:
            logger.warning("embeddings: %s is %d-dim but EMBED_DIM=%d — set EMBED_DIM=%d "
                           "and the vector index `dimension` to match", EMBED_MODEL, dim, EMBED_DIM, dim)
        logger.info("embeddings: local %s (dim=%s)", EMBED_MODEL, dim or EMBED_DIM)
    except Exception as exc:
        _state = "off"
        logger.info("embeddings: disabled (%s) — install sentence-transformers / check model", exc)


def _embed_local(text):
    v = _model.encode(EMBED_PROMPT + text if EMBED_PROMPT else text, normalize_embeddings=True)
    return [float(x) for x in v]


# --- openai (REST via requests) ---

def _load_openai():
    global _state
    if not os.environ.get("OPENAI_API_KEY"):
        _state = "off"
        logger.info("embeddings: disabled — EMBED_PROVIDER=openai but OPENAI_API_KEY is unset")
        return
    try:
        import requests  # noqa: F401 (already a pipeline dep)
        _state = "ok"
        logger.info("embeddings: openai %s (dim=%d)", EMBED_MODEL, EMBED_DIM)
    except Exception as exc:
        _state = "off"
        logger.info("embeddings: disabled (%s)", exc)


def _embed_openai(text):
    import requests
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {"model": EMBED_MODEL, "input": (EMBED_PROMPT + text) if EMBED_PROMPT else text}
    if EMBED_DIM:                                   # 3-* models support shrinking via `dimensions`
        body["dimensions"] = EMBED_DIM
    r = requests.post(base + "/embeddings", json=body,
                      headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
                      timeout=int(os.environ.get("EMBED_TIMEOUT", "30")))
    r.raise_for_status()
    return _l2norm([float(x) for x in r.json()["data"][0]["embedding"]])


def _load():
    if _state is not None:
        return
    (_load_openai if EMBED_PROVIDER == "openai" else _load_local)()


def available() -> bool:
    _load()
    return _state == "ok"


def embed(text: str):
    """Return an L2-normalized float vector for `text`, or None if unavailable/empty."""
    _load()
    if _state != "ok" or not text:
        return None
    try:
        return _embed_openai(text) if EMBED_PROVIDER == "openai" else _embed_local(text)
    except Exception as exc:
        logger.warning("embeddings: encode failed (%s)", exc)
        return None


def cosine(a, b) -> float:
    """Dot product of two normalized vectors == cosine similarity (sqrt guards
    any non-unit vectors)."""
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na  += x * x
        nb  += y * y
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def best_match(vec, candidates, field: str = "embedding"):
    """Return (candidate, similarity) for the closest candidate, or (None, -1)."""
    best, best_sim = None, -1.0
    if not vec:
        return best, best_sim
    for c in candidates:
        sim = cosine(vec, c.get(field))
        if sim > best_sim:
            best, best_sim = c, sim
    return best, best_sim
