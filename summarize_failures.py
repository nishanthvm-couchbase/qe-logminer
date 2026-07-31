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
import time

# droid exec defaults. "deepseek-v4-pro" is the org-permitted "Droid Core" model
# (droid's own default claude-opus-* is blocked by org policy here). Override with
# --model / DROID_MODEL if your org allows a different one.
DROID_BIN = os.environ.get("DROID_BIN", "droid")
DEFAULT_MODEL = os.environ.get("DROID_MODEL", "deepseek-v4-pro")
DROID_TIMEOUT = int(os.environ.get("DROID_TIMEOUT", "180"))  # seconds per failure
# A single droid call can fail transiently under concurrent load (backend
# rate-limit / overload / brief 5xx). Retry those with exponential backoff
# instead of aborting the whole job. Genuinely fatal errors (auth, bad model,
# quota) are detected and NOT retried.
DROID_RETRIES = int(os.environ.get("DROID_RETRIES", "3"))            # extra attempts after the first
DROID_RETRY_BACKOFF = float(os.environ.get("DROID_RETRY_BACKOFF", "8"))  # base secs, doubles each retry
_FATAL_DROID = re.compile(
    r"not logged in|unauthor|forbidden|invalid api key|invalid.*token|"
    r"quota exceeded|no such model|unknown model|model .*not (found|available)|"
    r"insufficient (funds|credit|quota)", re.I)

# Volatile tokens that differ between otherwise-identical failure instances
# (a retried test logs new timestamps/IPs/pids each run). Stripped before hashing
# so true repeats collapse to one signature.
_VOLATILE = [
    re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,\.]?\d*"),  # ISO timestamps
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),                      # IPv4
    re.compile(r"0x[0-9a-fA-F]+"),                                   # hex addresses
    re.compile(r":\d{2,5}\b"),                                       # ports
    re.compile(r"\b\d{6,}\b"),                                       # long ints (epoch ms, pids)
]


def _normalize(s):
    """Strip volatile tokens (timestamps, IPs, ports, pids, big counts) and
    collapse whitespace, so true repeats of a failure normalize to the same text."""
    s = s or ""
    for rx in _VOLATILE:
        s = rx.sub("·", s)
    return re.sub(r"\s+", " ", s).strip()


def failure_signature(failure):
    """Stable hash identifying a *distinct* failure.

    Basis: test_name + params + the normalized TRACEBACK (exception type, the
    file:line frames, and the message). The traceback is what's actually stable
    across retries of the same failure — the surrounding `error_lines` log window
    is not (it carries per-run durations, counts, vbucket maps, etc.), so it must
    NOT be part of the key. Consequences:
      • same test, same params, same exception/trace  -> same sig -> ONE droid call
      • different params                               -> different sig -> own call
      • a genuinely different error/exception          -> different trace -> own call
    Falls back to the normalized error window only when no traceback was captured.
    """
    return hashlib.md5(signature_basis(failure).encode("utf-8", "ignore")).hexdigest()


def signature_basis(failure):
    """The normalized text the signature hashes — test_name + params + traceback,
    volatile tokens stripped. Also the basis for the near-dup embedding, so an exact
    repeat embeds identically and a near-identical failure lands close to it."""
    tn = (failure.get("test_name") or "").strip()
    params = _normalize(failure.get("params") or "")
    tb = _normalize(failure.get("traceback") or "")
    if not tb:                                   # no traceback parsed — best-effort fallback
        tb = _normalize(failure.get("error_lines") or "")[:4000]
    return tn + "||" + params + "||" + tb


def estimate_prompt_tokens(failed_test, chars_per_token=3.5):
    """Rough INPUT-token estimate for the summarize prompt we WOULD have sent for this
    failure — the full template + evidence. Used to report tokens SAVED when a droid
    call is avoided (dedup/vector reuse). Deliberately lenient (chars/3.5) so savings
    are a worst-case-ish figure, never an over-count of real spend."""
    prompt = PROMPT_TEMPLATE.format(
        categories=sorted(VALID_CATEGORIES),
        test_name=failed_test.get("test_name", ""),
        params=failed_test.get("params", ""),
        traceback=failed_test.get("traceback", "") or "(no traceback captured)",
        error_lines=failed_test.get("error_lines", "") or "(no error lines captured)",
    )
    return int(len(prompt) / max(chars_per_token, 1.0)) + 1

VALID_CATEGORIES = {
    "product_bug", "test_bug", "infra", "environment", "timeout", "unknown",
}


class DroidError(RuntimeError):
    """Raised when a droid call fails (e.g. out of tokens, model blocked, timeout)
    and we are NOT ignoring droid failures. Lets the orchestrator stop instead of
    writing placeholder docs."""

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
  "complexity":     "high" or "low". "high" ONLY if diagnosing this failure genuinely \
required non-obvious, multi-factor reasoning (interacting causes, subtle state, \
cross-log correlation). "low" for straightforward/mechanical failures (plain \
assertion mismatch, timeout, missing resource, obvious ssh/infra/build error).

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
    complexity = str(obj.get("complexity", "low")).strip().lower()
    if complexity not in ("high", "low"):
        complexity = "low"
    return {
        "summary": str(obj.get("summary", "")).strip() or "(no summary produced)",
        "category": category,
        "root_cause": str(obj.get("root_cause", "")).strip(),
        "suggested_fix": str(obj.get("suggested_fix", "")).strip(),
        "confidence": confidence,
        "complexity": complexity,
    }


def _run_droid_once(prompt, model, auto, meta):
    """One `droid exec` invocation; return (ok, parsed_or_None, err_text).

    Uses --output-format json so we get a structured envelope:
      { type, is_error, duration_ms, session_id, result, usage:{input_tokens,...} }
    The model's answer is in `result`; exact token usage is in `usage`. We record
    one token-usage event per attempt (best-effort, never raises).
    """
    tmp_path = None
    usage = session_id = duration_ms = None
    ok = False
    obj = None
    err = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(prompt)
            tmp_path = tf.name
        cmd = [DROID_BIN, "exec", "-f", tmp_path, "--model", model,
               "--auto", auto, "--output-format", "json"]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=DROID_TIMEOUT
        )
        envelope = None
        try:
            envelope = json.loads(proc.stdout)
        except Exception:
            envelope = None

        if isinstance(envelope, dict):
            usage       = envelope.get("usage")
            session_id  = envelope.get("session_id")
            duration_ms = envelope.get("duration_ms")
            is_error    = bool(envelope.get("is_error", proc.returncode != 0))
            ok          = (proc.returncode == 0) and not is_error
            obj         = extract_json_object(envelope.get("result") or "") if ok else None
            if not ok:
                err = str(envelope.get("result", "")) or proc.stderr.strip()
                sys.stderr.write("  droid error: %s\n" % err[:300])
        else:
            # Fallback: not JSON (older droid / unexpected) — treat stdout as the answer.
            ok  = proc.returncode == 0
            obj = extract_json_object(proc.stdout) if ok else None
            if not ok:
                err = proc.stderr.strip()
                sys.stderr.write("  droid exited %d: %s\n" % (proc.returncode, err[:400]))
        return ok, obj, err
    except subprocess.TimeoutExpired:
        sys.stderr.write("  droid timed out after %ds\n" % DROID_TIMEOUT)
        return False, None, "timeout"
    except FileNotFoundError:
        sys.stderr.write("  droid binary not found (set DROID_BIN or add to PATH)\n")
        return False, None, "binary not found"
    finally:
        try:
            import token_usage
            token_usage.record({**(meta or {}), "model": model}, usage, duration_ms, session_id, ok)
        except Exception:
            pass
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_droid(prompt, model, auto="low", meta=None):
    """Invoke droid with bounded retry/backoff; return (ok, parsed_or_None).

    Transient failures (rate-limit / overload / 5xx / timeout) are retried up to
    DROID_RETRIES times with exponential backoff so one bad call doesn't abort the
    whole job. Clearly fatal errors (auth, bad model, quota) are NOT retried —
    they'll still surface fast so the fail-fast path can stop the backfill.
    """
    for attempt in range(DROID_RETRIES + 1):
        ok, obj, err = _run_droid_once(prompt, model, auto, meta)
        if ok:
            return True, obj
        if err and _FATAL_DROID.search(err):
            sys.stderr.write("  droid error looks fatal — not retrying\n")
            break
        if attempt < DROID_RETRIES:
            wait = DROID_RETRY_BACKOFF * (2 ** attempt)
            sys.stderr.write("  droid attempt %d/%d failed (transient) — retrying in %.0fs\n"
                             % (attempt + 1, DROID_RETRIES + 1, wait))
            time.sleep(wait)
    return False, None


def summarize_failure(failed_test, model, ignore_failure=False, meta=None, context_summary=None):
    """Run droid for one failed test and return the normalized summary dict.

    On droid failure: raise DroidError (default) so the caller can STOP — this
    prevents writing placeholder docs and burning through a partial backfill once
    droid is out of tokens. With ignore_failure=True, emit a placeholder instead.

    `meta` (job/os/component/build context) is forwarded for token-usage tracking.
    `context_summary`: a prior summary of a near-identical failure (from vector
    dedup). When given, it's injected as a strong hint so droid can confirm/refine
    it instead of reasoning from scratch — used for HIGH-complexity near-dups.
    """
    prompt = PROMPT_TEMPLATE.format(
        categories=sorted(VALID_CATEGORIES),
        test_name=failed_test.get("test_name", "unknown_test"),
        params=failed_test.get("params", ""),
        traceback=failed_test.get("traceback", "") or "(no traceback captured)",
        error_lines=failed_test.get("error_lines", "") or "(no error lines captured)",
    )
    if context_summary:
        prompt += (
            "\n\nNOTE — a very similar failure was previously analyzed:\n"
            f"  summary:    {context_summary.get('summary', '')}\n"
            f"  root_cause: {context_summary.get('root_cause', '')}\n"
            f"  category:   {context_summary.get('category', '')}\n"
            "Treat it as a strong hint: confirm it if it fits this failure, or correct it "
            "if this one genuinely differs. Still return the full JSON object."
        )
    call_meta = {**(meta or {}), "phase": "summarize",
                 "test_name": failed_test.get("test_name", "unknown_test")}
    ok, obj = run_droid(prompt, model, meta=call_meta)
    if not ok or obj is None:
        if ignore_failure:
            return {
                "summary": "(analysis unavailable — droid call failed)",
                "category": "unknown", "root_cause": "",
                "suggested_fix": "", "confidence": "low",
            }
        raise DroidError(
            "droid summary failed for %s" % failed_test.get("test_name", "?"))
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
        summary = summarize_failure(ft, model, meta={
            "name": job_name, "build_id": build_id, "build": meta.get("build"),
            "os": meta.get("os"), "component": meta.get("component"),
        })
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
