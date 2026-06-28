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
    "cb_user":       os.environ.get("CB_USER", "Administrator"),
    "cb_pass":       os.environ.get("CB_PASS", "esabhcuoc"),
}
_col = None
_col_tried = False
_lock = threading.Lock()


def configure(**kw) -> None:
    """Override config (host, creds, bucket/scope/collection, jsonl, cb_enabled). None = keep."""
    for k, v in kw.items():
        if v is not None and k in _cfg:
            _cfg[k] = v


def _collection():
    global _col, _col_tried
    if _col is not None or _col_tried:
        return _col
    _col_tried = True
    if not (_cfg["cb_enabled"] and _cfg["cb_host"]):
        return None
    b, s, c = _cfg["cb_bucket"], _cfg["cb_scope"], _cfg["cb_collection"]
    try:
        from couchbase.cluster import Cluster
        from couchbase.options import ClusterOptions
        from couchbase.auth import PasswordAuthenticator
        cl = Cluster("couchbase://%s" % _cfg["cb_host"],
                     ClusterOptions(PasswordAuthenticator(_cfg["cb_user"], _cfg["cb_pass"])))
        # Best-effort: make sure the dedicated collection (+ a primary index for the
        # dashboard's aggregate queries) exists. Idempotent; ignore if it already does.
        for stmt in (f"CREATE COLLECTION `{b}`.`{s}`.`{c}` IF NOT EXISTS",
                     f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{b}`.`{s}`.`{c}`"):
            try:
                for _ in cl.query(stmt):
                    pass
            except Exception:
                pass
        _col = cl.bucket(b).scope(s).collection(c)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  token_usage: CB connect failed (%s) — JSONL only\n" % e)
        _col = None
    return _col


def record(meta: Optional[Dict[str, Any]], usage: Optional[Dict[str, Any]],
           duration_ms: Optional[int], session_id: Optional[str], ok: bool) -> None:
    meta = meta or {}
    usage = usage or {}
    it = usage.get("input_tokens") or 0
    ot = usage.get("output_tokens") or 0
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
