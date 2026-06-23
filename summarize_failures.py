#!/usr/bin/python3
"""
Stage 2 of the logminer pipeline — droid-generated failure summaries.

Stage 1 (parse_console_log.py) turns a Jenkins consoleText into a parse_result
JSON whose `failed_tests[]` each carry: test_name, params, traceback, error_lines.

This stage takes that JSON and, for EACH failed test, asks `droid exec` to read
the error_lines (+ traceback + params) and return a structured summary. Each
summary becomes its OWN analysis doc (NOT a field on the run doc), keyed
deterministically, so a later step can N1QL-query them across builds and fold
them into a per-job "analysis doc".

Doc shape (one per failed test):
    {
      "type": "test_failure_analysis",        # N1QL: WHERE type = ...
      "job_name": "...", "build_id": 44388, "build": "8.1.0-2300",
      "os": "...", "component": "...",
      "test_name": "...",
      "summary": "<droid free-text, the primary field>",
      "category": "product_bug|test_bug|infra|environment|timeout|unknown",
      "root_cause": "...", "suggested_fix": "...", "confidence": "high|medium|low",
      "source": "<consoleText url or file>",
      "error_lines": "...",                    # kept for traceability / re-analysis
      "schema_version": 1
    }

Run standalone (local, no Couchbase needed):
    python3 summarize_failures.py parse_result_consoleText.json

Push straight to Couchbase:
    python3 summarize_failures.py parse_result_consoleText.json \
        --push --cb-host 172.23.105.219 --bucket test_analysis

Override identity (the run doc's identity lives in the collector, not the log):
    ... --job test_suite_executor --build-id 44388 \
        --build 8.1.0-2300 --os debian --component 2i
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

# droid exec defaults. "deepseek-v4-pro" is the org-permitted "Droid Core" model
# (droid's own default claude-opus-* is blocked by org policy here). Override with
# --model / DROID_MODEL if your org allows a different one.
DROID_BIN = os.environ.get("DROID_BIN", "droid")
DEFAULT_MODEL = os.environ.get("DROID_MODEL", "deepseek-v4-pro")
DROID_TIMEOUT = int(os.environ.get("DROID_TIMEOUT", "180"))  # seconds per failure

VALID_CATEGORIES = {
    "product_bug", "test_bug", "infra", "environment", "timeout", "unknown",
}

PROMPT_TEMPLATE = """\
You are a Couchbase QE test-failure analyst. Below is ONE failed test from a \
Jenkins test run: its name, input params, the Python traceback, and the \
filtered ERROR/WARNING log lines (with a little surrounding context).

Analyse it and return ONLY a single JSON object — no prose, no markdown fences, \
nothing before or after it — with EXACTLY these keys:

  "summary":        one or two plain-English sentences: what went wrong.
  "category":       one of {categories}.
  "root_cause":     the most likely underlying cause, concise.
  "suggested_fix":  a concrete next step or fix.
  "confidence":     "high", "medium", or "low".

Guidance on category:
  product_bug  - the server/product misbehaved (crash, wrong result, rebalance failure)
  test_bug     - the test code/assertion/setup is wrong or flaky
  infra        - CI/infra problem (ssh, build download, disk, node provisioning)
  environment  - config/version/platform mismatch
  timeout      - the test or build hit a time limit
  unknown      - genuinely cannot tell from the given data

TEST NAME:
{test_name}

PARAMS:
{params}

TRACEBACK:
{traceback}

ERROR LINES:
{error_lines}
"""


def make_key(job_name, build_id, test_name):
    """Deterministic key so re-runs upsert the same doc rather than duplicate."""
    raw = "%s-%s-%s" % (job_name, build_id, test_name)
    return "tfa_" + hashlib.md5(raw.encode()).hexdigest()


def derive_meta_from_source(source):
    """Best-effort job_name + build_id from a Jenkins consoleText URL.

    e.g. http://.../job/test_suite_executor/44388/consoleText
         -> ("test_suite_executor", "44388")
    Returns (job_name, build_id) with None for parts that can't be found.
    """
    job_name = build_id = None
    m = re.search(r"/job/([^/]+)/(\d+)/", source or "")
    if m:
        job_name, build_id = m.group(1), m.group(2)
    return job_name, build_id


def extract_json_object(text):
    """Pull the first balanced {...} JSON object out of droid's stdout.

    droid is instructed to emit only JSON, but we stay defensive: find the first
    '{' and walk braces (respecting strings) to its match, then json.loads it.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def normalize_summary(obj):
    """Coerce droid's object into our fixed schema, with safe fallbacks."""
    if not isinstance(obj, dict):
        obj = {}
    category = str(obj.get("category", "unknown")).strip().lower()
    if category not in VALID_CATEGORIES:
        category = "unknown"
    confidence = str(obj.get("confidence", "low")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return {
        "summary": str(obj.get("summary", "")).strip() or "(no summary produced)",
        "category": category,
        "root_cause": str(obj.get("root_cause", "")).strip(),
        "suggested_fix": str(obj.get("suggested_fix", "")).strip(),
        "confidence": confidence,
    }


def run_droid(prompt, model, auto="low"):
    """Invoke `droid exec` headless on a prompt file; return (ok, parsed_or_None)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(prompt)
            tmp_path = tf.name
        cmd = [DROID_BIN, "exec", "-f", tmp_path, "--model", model, "--auto", auto]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=DROID_TIMEOUT
        )
        if proc.returncode != 0:
            sys.stderr.write(
                "  droid exited %d: %s\n" % (proc.returncode, proc.stderr.strip()[:400])
            )
            return False, None
        return True, extract_json_object(proc.stdout)
    except subprocess.TimeoutExpired:
        sys.stderr.write("  droid timed out after %ds\n" % DROID_TIMEOUT)
        return False, None
    except FileNotFoundError:
        sys.stderr.write(
            "  droid binary not found (set DROID_BIN or add to PATH)\n"
        )
        return False, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def summarize_failure(failed_test, model):
    """Run droid for one failed test and return the normalized summary dict."""
    prompt = PROMPT_TEMPLATE.format(
        categories=sorted(VALID_CATEGORIES),
        test_name=failed_test.get("test_name", "unknown_test"),
        params=failed_test.get("params", ""),
        traceback=failed_test.get("traceback", "") or "(no traceback captured)",
        error_lines=failed_test.get("error_lines", "") or "(no error lines captured)",
    )
    ok, obj = run_droid(prompt, model)
    if not ok or obj is None:
        # Never crash the pipeline on a single bad analysis — emit a placeholder
        # doc so the failure is still recorded and can be re-analysed later.
        return {
            "summary": "(analysis unavailable — droid call failed)",
            "category": "unknown",
            "root_cause": "",
            "suggested_fix": "",
            "confidence": "low",
        }
    return normalize_summary(obj)


def build_analysis_docs(parse_result, meta, model):
    """Yield (key, doc) for every failed test in a parse_result JSON."""
    source = parse_result.get("source", "")
    src_job, src_build = derive_meta_from_source(source)
    job_name = meta.get("job") or src_job or "unknown_job"
    build_id = meta.get("build_id") or src_build or "0"

    docs = []
    failed = parse_result.get("failed_tests", [])
    for idx, ft in enumerate(failed):
        test_name = ft.get("test_name", "unknown_test")
        sys.stderr.write(
            "  [%d/%d] summarizing: %s\n" % (idx + 1, len(failed), test_name)
        )
        summary = summarize_failure(ft, model)
        doc = {
            "type": "test_failure_analysis",
            "schema_version": 1,
            "job_name": job_name,
            "build_id": int(build_id) if str(build_id).isdigit() else build_id,
            "build": meta.get("build"),
            "os": meta.get("os"),
            "component": meta.get("component"),
            "test_name": test_name,
            "params": ft.get("params", ""),
            "source": source,
            "error_lines": ft.get("error_lines", ""),
        }
        doc.update(summary)
        docs.append((make_key(job_name, build_id, test_name), doc))
    return docs


def push_to_couchbase(docs, host, bucket, user, password):
    """Upsert analysis docs into Couchbase. Imported lazily so the local path
    (no --push) has zero Couchbase dependency."""
    from couchbase.cluster import Cluster
    from couchbase.options import ClusterOptions
    from couchbase.auth import PasswordAuthenticator

    cluster = Cluster(
        "couchbase://%s" % host,
        ClusterOptions(PasswordAuthenticator(user, password)),
    )
    col = cluster.bucket(bucket).default_collection()
    n = 0
    for key, doc in docs:
        try:
            col.upsert(key, doc)
            n += 1
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("  upsert failed %s: %s\n" % (key, exc))
    return n


def main():
    ap = argparse.ArgumentParser(
        description="Generate droid failure summaries from a logminer parse_result JSON."
    )
    ap.add_argument("parse_result", help="path to parse_result_<x>.json from parse_console_log.py")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="droid model id")
    ap.add_argument("--out", help="output JSON path (default: analysis_<base>.json)")
    # identity overrides (the run doc's identity lives in the collector, not the log)
    ap.add_argument("--job", help="job/pseudo name override")
    ap.add_argument("--build-id", dest="build_id", help="Jenkins build id override")
    ap.add_argument("--build", help="product build, e.g. 8.1.0-2300")
    ap.add_argument("--os", dest="os_", help="os, e.g. debian")
    ap.add_argument("--component", help="component, e.g. 2i")
    # couchbase push
    ap.add_argument("--push", action="store_true", help="upsert docs to Couchbase")
    ap.add_argument("--cb-host", default=os.environ.get("CB_HOST", "172.23.105.219"))
    ap.add_argument("--bucket", default=os.environ.get("CB_ANALYSIS_BUCKET", "test_analysis"))
    ap.add_argument("--cb-user", default=os.environ.get("CB_USER", "Administrator"))
    ap.add_argument("--cb-pass", default=os.environ.get("CB_PASS", "esabhcuoc"))
    args = ap.parse_args()

    with open(args.parse_result) as f:
        parse_result = json.load(f)

    meta = {
        "job": args.job,
        "build_id": args.build_id,
        "build": args.build,
        "os": args.os_,
        "component": args.component,
    }

    n_failed = len(parse_result.get("failed_tests", []))
    sys.stderr.write("Summarizing %d failed test(s) with model %s...\n"
                     % (n_failed, args.model))
    docs = build_analysis_docs(parse_result, meta, args.model)

    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(args.parse_result))[0]
        base = base.replace("parse_result_", "")
        out_path = "analysis_%s.json" % base
    with open(out_path, "w") as f:
        json.dump([d for _, d in docs], f, indent=2)
    sys.stderr.write("Wrote %d analysis doc(s) to %s\n" % (len(docs), out_path))

    if args.push:
        pushed = push_to_couchbase(
            docs, args.cb_host, args.bucket, args.cb_user, args.cb_pass
        )
        sys.stderr.write("Pushed %d/%d doc(s) to %s/%s\n"
                         % (pushed, len(docs), args.cb_host, args.bucket))


if __name__ == "__main__":
    main()
