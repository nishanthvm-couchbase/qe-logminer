#!/usr/bin/python3
"""
Per-call droid token-usage recorder for the Test Analysis pipeline.

`droid exec --output-format json` returns an exact `usage` block per call
(input/output/cache tokens) + duration_ms + session_id. run_droid() passes that
here with the job context (phase, job, component, os, build, test). We record ONE
event per droid call to:

  * a local JSONL file  (TOKEN_USAGE_LOG, default ./token_usage.jsonl on the droid
    machine) — resilient, append-only, no concurrency risk; and
  * a Couchbase `token_usage` doc (best-effort) in the test_analysis bucket, keyed
    `tokusage_<session_id>`, so the dashboard can N1QL-aggregate tokens by
    job / component / time later.

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
    "jsonl":      os.environ.get("TOKEN_USAGE_LOG", "token_usage.jsonl"),
    "cb_enabled": os.environ.get("TOKEN_USAGE_CB", "1") != "0",
    "cb_host":    os.environ.get("CB_HOST", ""),
    "cb_bucket":  os.environ.get("TOKEN_USAGE_BUCKET", "test_analysis"),
    "cb_user":    os.environ.get("CB_USER", "Administrator"),
    "cb_pass":    os.environ.get("CB_PASS", "esabhcuoc"),
}
_col = None
_col_tried = False
_lock = threading.Lock()


def configure(**kw) -> None:
    """Override config (host, creds, bucket, jsonl path, cb_enabled). None = keep."""
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
    try:
        from couchbase.cluster import Cluster
        from couchbase.options import ClusterOptions
        from couchbase.auth import PasswordAuthenticator
        cl = Cluster("couchbase://%s" % _cfg["cb_host"],
                     ClusterOptions(PasswordAuthenticator(_cfg["cb_user"], _cfg["cb_pass"])))
        _col = cl.bucket(_cfg["cb_bucket"]).default_collection()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  token_usage: CB connect failed (%s) — JSONL only\n" % e)
        _col = None
    return _col


def record(meta: Optional[Dict[str, Any]], usage: Optional[Dict[str, Any]],
           duration_ms: Optional[int], session_id: Optional[str], ok: bool) -> None:
    meta = meta or {}
    usage = usage or {}
    it = usage.get("input_tokens")
    ot = usage.get("output_tokens")
    ev = {
        "type":      "token_usage",
        "ts":        datetime.now(timezone.utc).isoformat(),
        "phase":     meta.get("phase"),                       # summarize | analysis
        "job_name":  meta.get("name") or meta.get("job_name"),
        "os":        meta.get("os"),
        "component": meta.get("component"),
        "build":     meta.get("build"),
        "build_id":  meta.get("build_id"),
        "test_name": meta.get("test_name"),
        "model":     meta.get("model"),
        "input_tokens":                it,
        "output_tokens":               ot,
        "cache_read_input_tokens":     usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "total_tokens":                (it or 0) + (ot or 0),
        "duration_ms": duration_ms,
        "session_id":  session_id,
        "ok":          ok,
    }

    # 1) local JSONL (always)
    try:
        with _lock, open(_cfg["jsonl"], "a", encoding="utf-8") as f:
            f.write(json.dumps(ev) + "\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("  token_usage: jsonl write failed (%s)\n" % e)

    # 2) Couchbase event doc (best-effort)
    col = _collection()
    if col is not None:
        try:
            uniq = session_id or hashlib.md5(
                (ev["ts"] + str(ev["job_name"]) + str(ev["test_name"]) + str(ev["phase"])).encode()
            ).hexdigest()
            col.upsert("tokusage_" + uniq, ev)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("  token_usage: CB upsert failed (%s)\n" % e)
