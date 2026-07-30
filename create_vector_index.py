#!/usr/bin/python3
"""
Create the index that backs near-duplicate (vector) dedup on `test_analysis`.

Two tiers:

  1. GSI (always) — covers `AnalysisStore.candidate_embeddings`: filter by
     type/component/embedding_model, order by build_id. This alone makes the
     brute-force cosine path (in analyze_build) efficient at current scale.

  2. Vector ANN (optional, for large corpora) — a Couchbase vector index so the
     nearest-neighbour search can run in the cluster instead of pulling candidates.
     Syntax differs by server version/service; the exact DDL is PRINTED for you to
     run/adjust rather than executed blindly.

Usage:
  python3 create_vector_index.py                 # creates the GSI index
  python3 create_vector_index.py --password X    # non-default creds
"""
from __future__ import annotations

import argparse
import os
from datetime import timedelta

# Vector dimension of the embedding model (MiniLM=384, gte-large=1024,
# gte-Qwen2-1.5B=1536, gte-Qwen2-7B=3584). Keep in sync with embeddings.EMBED_DIM.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))

from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from couchbase.auth import PasswordAuthenticator

BUCKET = "test_analysis"

GSI_DDL = (
    f"CREATE INDEX idx_tfa_embed ON `{BUCKET}`(component, embedding_model, build_id) "
    f"WHERE type = 'test_failure_analysis' AND embedding IS NOT MISSING"
)

# Optional ANN index — adjust `dimension` to your model (MiniLM/bge-small = 384) and
# the training params to your server version. Requires the Index/Search vector feature.
ANN_DDL_HINT = (
    f"CREATE INDEX idx_tfa_ann ON `{BUCKET}`(embedding VECTOR) "
    f"WHERE type = 'test_failure_analysis' "
    f"WITH {{ 'dimension': {EMBED_DIM}, 'similarity': 'cosine', 'description': 'IVF,SQ8' }}"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="172.23.105.219")
    ap.add_argument("--user", default="Administrator")
    ap.add_argument("--password", default="esabhcuoc")
    args = ap.parse_args()

    to = ClusterTimeoutOptions(connect_timeout=timedelta(seconds=20),
                               query_timeout=timedelta(seconds=120))
    cl = Cluster(f"couchbase://{args.host}",
                 ClusterOptions(PasswordAuthenticator(args.user, args.password), timeout_options=to))
    cl.wait_until_ready(timedelta(seconds=20))

    print("Creating GSI index for candidate_embeddings…")
    try:
        list(cl.query(GSI_DDL))
        print("  created: idx_tfa_embed")
    except Exception as exc:
        msg = str(exc)
        if "already exist" in msg.lower() or "duplicate index" in msg.lower():
            print("  already exists: idx_tfa_embed")
        else:
            print("  FAILED:", msg)

    print("\nOptional — for large corpora, create a vector ANN index (adjust to your")
    print("server version, then run in the Query workbench):\n")
    print("  " + ANN_DDL_HINT)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
