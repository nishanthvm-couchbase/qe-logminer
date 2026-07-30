#!/usr/bin/python3
"""
Per-call droid token-usage recorder for the Test Analysis pipeline.

`droid exec --output-format json` returns an exact `usage` block per call
(input/output/cache tokens) + duration_ms + session_id. run_droid() passes that
here with the job context (phase, job, component, os, build, test). We record ONE
event per droid call to:

  * a local JSONL file  (TOKEN_USAGE_LOG, default ./token_usage.jsonl on the droid
    machine) — resilient, append-only, no concurrency risk; and
  * a Couchbase doc (best-effort) in a DEDICATED collection
    `test_analysis`._default.`token_usage`, keyed `tokusage_<session_id>`, so the
    dashboard can N1QL-aggregate tokens by job / component / time later.

Both are best-effort: token logging must NEVER break the analysis pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_cfg = {
    "jsonl":         os.environ.get("TOKEN_USAGE_LOG", "token_usage.jsonl"),
    "cb_enabled":    os.environ.get("TOKEN_USAGE_CB", "1") != "0",
    "cb_host":       os.environ.get("CB_HOST", ""),
    "cb_bucket":     os.environ.get("TOKEN_USAGE_BUCKET", "test_analysis"),
    "cb_scope":      os.environ.get("TOKEN_USAGE_SCOPE", "_default"),
    "cb_collection": os.environ.get("TOKEN_USAGE_COLLECTION", "token_usage"),
    "metrics_collection": os.environ.get("ANALYSIS_METRICS_COLLECTION", "analysis_metrics"),
    "metrics_jsonl":      os.environ.get("ANALYSIS_METRICS_LOG", "analysis_metrics.jsonl"),
    "cb_user":       os.environ.get("CB_USER", "Administrator"),
    "cb_pass":       os.environ.get("CB_PASS", "esabhcuoc"),
}
_cluster = None
_cluster_tried = False
_cols: Dict[str, Any] = {}
_totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0}   # cumulative this process (≈ this run)
_lock = threading.Lock()


def configure(**kw) -> None:
    """Override config (host, creds, bucket/scope/collection, jsonl, cb_enabled). None = keep."""
    for k, v in kw.items():
        if v is not None and k in _cfg:
            _cfg[k] = v


def _get_cluster():
    global _cluster, _cluster_tried
    if _cluster is not None or _cluster_tried:
        return _cluster
    _cluster_tried = True
    if not (_cfg["cb_enabled"] and _cfg["cb_host"]):
        return None
    try:
        from couchbase.cluster import Cluster
        from couchbase.options import ClusterOptions
        from couchbase.auth import PasswordAuthenticator
        _cluster = Cluster("couchbase://%s" % _cfg["cb_host"],
                           ClusterOptions(PasswordAuthenticator(_cfg["cb_user"], _cfg["cb_pass"])))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  token_usage: CB connect failed (%s) — JSONL only\n" % e)
        _cluster = None
    return _cluster


def _get_collection(coll_name):
    """Cached handle to bucket/scope/<coll_name>; creates it + a primary index
    idempotently. Returns None if CB is unavailable (callers degrade to JSONL)."""
    if coll_name in _cols:
        return _cols[coll_name]
    cl = _get_cluster()
    if cl is None:
        _cols[coll_name] = None
        return None
    b, s = _cfg["cb_bucket"], _cfg["cb_scope"]
    try:
        for stmt in (f"CREATE COLLECTION `{b}`.`{s}`.`{coll_name}` IF NOT EXISTS",
                     f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{b}`.`{s}`.`{coll_name}`"):
            try:
                for _ in cl.query(stmt):
                    pass
            except Exception:
                pass
        _cols[coll_name] = cl.bucket(b).scope(s).collection(coll_name)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  token_usage: collection %s failed (%s)\n" % (coll_name, e))
        _cols[coll_name] = None
    return _cols[coll_name]


def _collection():
    return _get_collection(_cfg["cb_collection"])


def totals() -> Dict[str, int]:
    """Cumulative droid token spend recorded this process (≈ this analyze_build run)."""
    with _lock:
        return dict(_totals)


def record_metrics(key: str, doc: Dict[str, Any]) -> None:
    """Persist one run-level metrics doc (dedup/complexity/token counters) to a
    JSONL file AND the `analysis_metrics` collection. Best-effort, never raises."""
    doc = {**doc, "ts": datetime.now(timezone.utc).isoformat()}
    try:
        with _lock, open(_cfg["metrics_jsonl"], "a", encoding="utf-8") as f:
            f.write(json.dumps(doc) + "\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  metrics: jsonl write failed (%s)\n" % e)
    col = _get_collection(_cfg["metrics_collection"])
    if col is not None:
        try:
            col.upsert(key, doc)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("  metrics: CB upsert failed (%s)\n" % e)


def record(meta: Optional[Dict[str, Any]], usage: Optional[Dict[str, Any]],
           duration_ms: Optional[int], session_id: Optional[str], ok: bool) -> None:
    meta = meta or {}
    usage = usage or {}
    it = usage.get("input_tokens") or 0
    ot = usage.get("output_tokens") or 0
    with _lock:
        _totals["calls"] += 1
        _totals["input_tokens"] += it
        _totals["output_tokens"] += ot
    # Only the details needed to visualise token spend by job / component over time.
    ev = {
        "ts":            datetime.now(timezone.utc).isoformat(),
        "phase":         meta.get("phase"),                       # summarize | analysis
        "job_name":      meta.get("name") or meta.get("job_name"),
        "component":     meta.get("component"),
        "build":         meta.get("build"),
        "model":         meta.get("model"),
        "input_tokens":  it,
        "output_tokens": ot,
        "total_tokens":  it + ot,
        "duration_ms":   duration_ms,
    }

    # 1) local JSONL (always)
    try:
        with _lock, open(_cfg["jsonl"], "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  token_usage: jsonl write failed (%s)\n" % e)

    # 2) Couchbase event doc in the dedicated collection (best-effort)
    col = _collection()
    if col is not None:
        try:
            uniq = session_id or hashlib.md5(
                (ev["ts"] + str(ev["job_name"]) + str(ev["phase"])).encode()
            ).hexdigest()
            col.upsert("tokusage_" + uniq, ev)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("  token_usage: CB upsert failed (%s)\n" % e)
