#!/usr/bin/python3
"""
Downstream analysis job — entry point.

Triggered after a test_suite_executor build finishes. Runs ON a droid machine.
Pipeline:

  build-url ─▶ fetch params + consoleText
            ─▶ reconstruct greenboard identity (job_identity)
            ─▶ parse_console_log → failed tests
            ─▶ droid summary per failure        → Tier-1 docs  (test_analysis bucket)
            ─▶ gather history + existing doc
            ─▶ droid synthesis (code owns numbers) → Tier-2 doc (one per job+build)
            ─▶ direct Couchbase upsert (temp cluster 172.23.105.219)

Production use:
  analyze_build.py --build-url http://qe-jenkins1.../job/test_suite_executor/44388/

Local/offline testing (no Jenkins, no cluster, no droid):
  analyze_build.py --console-file consoleText \
      --name "debian-2i_fooGSI..." --display-name debian-2i_foo \
      --os DEBIAN --component 2I --build 8.1.0-2299 --build-id 44388 \
      --dry-run --no-droid
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import requests

from parse_console_log import fetch_content, build_result
from job_identity import identity_from_actions
from summarize_failures import summarize_failure, failure_signature, signature_basis, DroidError
try:
    import embeddings
except Exception:
    embeddings = None
from build_analysis import build_analysis_doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("analyze_build")

PLACEHOLDER_SUMMARY = {
    "summary": "(droid skipped)", "category": "unknown",
    "root_cause": "", "suggested_fix": "", "confidence": "low",
}
CAPPED_SUMMARY = {
    "summary": "(not analyzed — per-job failure cap reached)", "category": "unknown",
    "root_cause": "", "suggested_fix": "", "confidence": "low", "capped": True,
}
# Token guardrail: summarize at most this many UNIQUE failures per job with droid.
# Beyond it, failures are still recorded (so counts stay correct) but not sent to
# droid. Protects a shared token budget from a single heavily-failing job (e.g. a
# job with 40 failures would otherwise be ~40 droid calls). 0 = no cap.
MAX_FAILURES_PER_JOB = int(os.environ.get("DROID_MAX_FAILURES_PER_JOB", "5"))

# Near-duplicate dedup via local embeddings (zero-token). OFF by default; enable with
# VECTOR_DEDUP=1 after `pip install sentence-transformers` on the slave and creating the
# index (see create_vector_index.py). A new failure whose embedding is within
# VECTOR_SIM_THRESHOLD of an already-summarized one REUSES that summary — no droid call.
VECTOR_DEDUP         = os.environ.get("VECTOR_DEDUP", "0") == "1"
VECTOR_SIM_THRESHOLD = float(os.environ.get("VECTOR_SIM_THRESHOLD", "0.93"))


def _jenkins_auth():
    u, p = os.environ.get("UBER_USER"), os.environ.get("UBER_PASS")
    return (u, p) if u and p else None


def fetch_build(build_url: str):
    """Return (actions, build_id, console_text) for an executor build URL."""
    api = build_url.rstrip("/") + "/api/json"
    resp = requests.get(api, params={"depth": 0}, auth=_jenkins_auth(), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    console = fetch_content(build_url.rstrip("/") + "/consoleText")
    return data.get("actions"), data.get("number"), console


def make_summary_doc(identity, build_id, failure, summary, sig=None, embedding=None):
    doc = {
        "type": "test_failure_analysis",
        "schema_version": 1,
        "job_name": identity["name"],            # name WITH variants (greenboard key)
        "display_name": identity["display_name"],
        "os": identity["os"],
        "component": identity["component"],
        "build": identity["build"],
        "build_id": build_id,
        "test_name": failure.get("test_name", "unknown_test"),
        # signature distinguishing this failure from same-named ones with different
        # params/error — so distinct parametrized failures are kept separate.
        "sig": sig,
        "params": failure.get("params", ""),
        # error_lines is intentionally NOT stored: the summary is already distilled
        # from it, Tier-2 reads only summaries, and full logs persist in S3 (linked
        # from greenboard). We keep the compact traceback for human spot-checks.
        "traceback": failure.get("traceback", ""),
        **summary,
    }
    # near-dup embedding (present only when VECTOR_DEDUP produced a vector) — seeds
    # future nearest-neighbour reuse. Stored per-model since vectors aren't cross-comparable.
    if embedding:
        doc["embedding"] = embedding
        doc["embedding_model"] = embeddings.EMBED_TAG if embeddings else "unknown"
    return doc


def main():
    ap = argparse.ArgumentParser(description="Generate Test Analysis docs for one executor build.")
    src = ap.add_argument_group("source (one of)")
    src.add_argument("--build-url", help="executor build URL (production path)")
    src.add_argument("--console-file", help="local consoleText file (offline testing)")
    src.add_argument("--actions-file", help="JSON file of the build's `actions` array (offline)")

    idg = ap.add_argument_group("identity overrides (offline testing)")
    for f in ("os", "component", "name", "display-name", "build"):
        idg.add_argument(f"--{f}")
    idg.add_argument("--build-id", type=int)

    ap.add_argument("--cb-host", default=os.environ.get("CB_HOST", "172.23.105.219"))
    ap.add_argument("--cb-user", default=os.environ.get("CB_USER", "Administrator"))
    ap.add_argument("--cb-pass", default=os.environ.get("CB_PASS", "esabhcuoc"))
    ap.add_argument("--model", default=os.environ.get("DROID_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--dry-run", action="store_true", help="no Couchbase writes/reads; print docs")
    ap.add_argument("--no-droid", action="store_true", help="skip droid (placeholder summaries)")
    ap.add_argument("--ignore-droid-failure", action="store_true",
                    help="on a droid failure, write a placeholder and continue instead of "
                         "stopping (default: stop with a non-zero exit, write nothing)")
    args = ap.parse_args()

    # ---- token-usage tracking (per droid call → JSONL + CB token_usage docs) ----
    try:
        import token_usage
        token_usage.configure(cb_host=(None if args.dry_run else args.cb_host),
                              cb_user=args.cb_user, cb_pass=args.cb_pass)
    except Exception:
        pass

    # ---- acquire actions + console ----
    actions, build_id, console = None, args.build_id, None
    try:
        if args.build_url:
            logger.info("Fetching executor build %s", args.build_url)
            actions, build_id, console = fetch_build(args.build_url)
            source = args.build_url
        else:
            if not args.console_file:
                ap.error("provide --build-url, or --console-file for offline mode")
            console = fetch_content(args.console_file)
            source = args.console_file
            if args.actions_file:
                actions = json.load(open(args.actions_file))
    except Exception as exc:
        # Console/log unavailable (e.g. Jenkins purged it AND the S3 copy is missing).
        # Exit cleanly rather than crash — backfill treats this run as "nothing to do".
        logger.warning("Could not fetch build/console (%s): %s — skipping.",
                       args.build_url or args.console_file, exc)
        return 0

    # ---- identity ----
    if args.name and args.build and args.os and args.component:
        identity = {
            "os": args.os, "component": args.component, "name": args.name,
            "display_name": getattr(args, "display_name") or args.name,
            "variants": {}, "build": args.build, "build_id": build_id,
        }
    else:
        identity = identity_from_actions(actions, build_id)
    if not identity:
        logger.info("Not an analyzable executor run (no component/build params) — exiting.")
        return 0
    build_id = identity.get("build_id") or build_id
    logger.info("Identity: %s | %s | %s | %s (build_id=%s)",
                identity["os"], identity["component"], identity["name"],
                identity["build"], build_id)

    # ---- parse ----
    parse_result = build_result(console, source)
    failures = parse_result["failed_tests"]
    logger.info("Parsed: %d tests, %d passed, %d failed, %d aborted",
                parse_result["total_tests"], parse_result["passed"],
                parse_result["failed"], parse_result["aborted"])
    if not failures:
        logger.info("Clean run (no failures) — no analysis doc per design. Exiting.")
        return 0

    # ---- store ----
    store = None
    if not args.dry_run:
        from analysis_store import AnalysisStore
        store = AnalysisStore(args.cb_host, args.cb_user, args.cb_pass)
        logger.info("Connected to test_analysis @ %s", args.cb_host)

    from analysis_store import key_summary, key_analysis
    try:
        # ---- Tier 1: per-failure summary docs ----
        # Dedup droid CALLS by failure signature: identical failures (same test +
        # params + error shape, timestamps/IPs/pids normalized out) are summarized
        # ONCE and the result reused. Genuinely different failures (different params
        # or a different error) have a different signature and get their own call.
        # Every failure still produces a doc, so stored docs / related / Tier-2 stats
        # are byte-identical to summarizing each one separately — only the droid
        # spend drops (e.g. 16 calls -> 3 when a test was retried 7x and 8x).
        summary_docs = []
        sig_cache = {}
        n_calls = 0        # tokened droid calls (fresh + context-augmented)
        n_capped = 0
        n_exact = 0        # identical-signature reuses (0 tokens)
        n_vector = 0       # low-complexity near-dups reused directly (0 tokens)
        n_vector_aug = 0   # high-complexity near-dups → droid call WITH prior context

        # Near-dup dedup: load recent same-component summaries carrying an embedding of
        # our model; a new failure within VECTOR_SIM_THRESHOLD of one reuses its summary
        # (no droid call) UNLESS that summary was rated high-complexity — then we re-call
        # droid but PRIME it with the prior summary (better analysis at ~same cost).
        # Fresh summaries are appended so later failures in THIS job also match them.
        vec_on = (VECTOR_DEDUP and not args.no_droid and store is not None
                  and embeddings is not None and embeddings.available())
        vec_candidates = []
        cand_fetched = cand_high = cand_low = 0
        if vec_on:
            vec_candidates = store.candidate_embeddings(identity["component"], embeddings.EMBED_TAG) or []
            cand_fetched = len(vec_candidates)
            cand_high = sum(1 for c in vec_candidates if str(c.get("complexity", "low")).lower() == "high")
            cand_low  = cand_fetched - cand_high
            logger.info("Vector dedup ON (threshold %.2f) — %d candidates for %s (%d high / %d low complexity)",
                        VECTOR_SIM_THRESHOLD, cand_fetched, identity["component"], cand_high, cand_low)

        REUSE_FIELDS = ("summary", "category", "root_cause", "suggested_fix", "confidence", "complexity")
        for i, failure in enumerate(failures, 1):
            tname = failure.get("test_name", "unknown_test")
            sig = failure_signature(failure)
            emb = embeddings.embed(signature_basis(failure)) if vec_on else None
            meta_ctx = {"name": identity["name"], "os": identity["os"],
                        "component": identity["component"], "build": identity["build"],
                        "build_id": build_id}

            if args.no_droid:
                summ = PLACEHOLDER_SUMMARY
            elif sig in sig_cache:
                summ = sig_cache[sig]
                n_exact += 1
                logger.info("  [summary %d/%d] %s — reused (identical failure)", i, len(failures), tname)
            else:
                vec_cand, vec_sim = (embeddings.best_match(emb, vec_candidates) if (vec_on and emb) else (None, -1.0))
                near      = vec_cand is not None and vec_sim >= VECTOR_SIM_THRESHOLD
                near_high = near and str(vec_cand.get("complexity", "low")).lower() == "high"
                under_cap = not (MAX_FAILURES_PER_JOB > 0 and n_calls >= MAX_FAILURES_PER_JOB)

                if near and not near_high:
                    # low-complexity near-dup → reuse directly, spend 0 tokens
                    summ = {k: vec_cand.get(k, "") for k in REUSE_FIELDS}
                    summ["reused_via"] = "vector"; summ["reuse_sim"] = round(vec_sim, 4)
                    sig_cache[sig] = summ; n_vector += 1
                    logger.info("  [summary %d/%d] %s — reused via vector (sim=%.3f, low-complexity, ~%s)",
                                i, len(failures), tname, vec_sim, vec_cand.get("test_name"))
                elif near_high and under_cap:
                    # high-complexity near-dup → re-analyze with droid, primed by the prior summary
                    logger.info("  [summary %d/%d] %s — near-dup HIGH complexity (sim=%.3f) → droid + context",
                                i, len(failures), tname, vec_sim)
                    summ = summarize_failure(failure, args.model, args.ignore_droid_failure,
                                             meta=meta_ctx, context_summary=vec_cand)
                    summ["augmented_from_vector"] = True; summ["reuse_sim"] = round(vec_sim, 4)
                    sig_cache[sig] = summ; n_calls += 1; n_vector_aug += 1
                elif under_cap:
                    # fresh droid call
                    logger.info("  [summary %d/%d] %s", i, len(failures), tname)
                    summ = summarize_failure(failure, args.model, args.ignore_droid_failure, meta=meta_ctx)
                    sig_cache[sig] = summ; n_calls += 1
                elif near:
                    # cap reached but a near-dup exists → reuse it (better than a blank capped doc)
                    summ = {k: vec_cand.get(k, "") for k in REUSE_FIELDS}
                    summ["reused_via"] = "vector_capped"; summ["reuse_sim"] = round(vec_sim, 4)
                    sig_cache[sig] = summ; n_vector += 1
                    logger.info("  [summary %d/%d] %s — CAP reached, reused near-dup (sim=%.3f)", i, len(failures), tname, vec_sim)
                else:
                    # cap reached, no near-dup → record but spend nothing
                    summ = CAPPED_SUMMARY; sig_cache[sig] = summ; n_capped += 1
                    logger.info("  [summary %d/%d] %s — CAPPED (>%d unique failures)", i, len(failures), tname, MAX_FAILURES_PER_JOB)

            doc = make_summary_doc(identity, build_id, failure, summ, sig, embedding=emb)
            summary_docs.append(doc)
            # seed the in-memory candidate set with real summaries (not capped/placeholder)
            if vec_on and emb and summ.get("summary") and not summ.get("capped"):
                cand = {k: summ.get(k, "") for k in REUSE_FIELDS}
                cand["embedding"] = emb; cand["test_name"] = tname
                vec_candidates.append(cand)
            if store:
                store.upsert_summary(key_summary(identity["name"], build_id, tname, sig), doc)

        pushed_high = sum(1 for d in summary_docs if str(d.get("complexity", "low")).lower() == "high")
        pushed_low  = len(summary_docs) - pushed_high
        if not args.no_droid:
            logger.info("Tier-1: %d droid call(s) [%d fresh, %d context-augmented] for %d failure(s) "
                        "(%d exact-dedup, %d vector-reuse, %d capped)",
                        n_calls, n_calls - n_vector_aug, n_vector_aug, len(failures),
                        n_exact, n_vector, n_capped)

        # ---- gather context for Tier 2 ----
        related, history, trend, existing = list(summary_docs), [], [], None
        if store:
            # prior reruns of this product-build + cross-build history + trend + existing doc
            related = store.related_summaries(identity["name"], identity["build"]) or summary_docs
            history = store.test_failure_history(identity["name"])
            trend   = store.job_trend(identity["name"])
            existing = store.get_analysis(
                key_analysis(identity["name"], identity["build"]))
        # ensure the just-computed summaries are represented even before they're queryable.
        # Key on (test_name, build_id, sig) so distinct parametrized failures of the same
        # method are all kept, not collapsed to one.
        seen = {(d.get("test_name"), d.get("build_id"), d.get("sig")) for d in related}
        related += [d for d in summary_docs
                    if (d.get("test_name"), d.get("build_id"), d.get("sig")) not in seen]

        # ---- Tier 2: analysis doc ----
        now_iso = datetime.now(timezone.utc).isoformat()
        analysis = build_analysis_doc(
            identity, parse_result, related, history, trend, existing,
            ("__none__" if args.no_droid else args.model), now_iso,
            ignore_failure=args.ignore_droid_failure,
        )
    except DroidError as exc:
        logger.error("Droid failure (likely out of tokens): %s. STOPPING — no analysis doc "
                     "written for this run. Restore tokens, then re-run with --skip-existing "
                     "(this job will be redone; completed jobs are skipped).", exc)
        return 3
    akey = key_analysis(identity["name"], identity["build"])
    if store:
        store.upsert_analysis(akey, analysis)
        logger.info("Upserted analysis doc %s (verdict=%s, %d failures)",
                    akey, analysis["verdict"], len(analysis["failures"]))
    else:
        logger.info("DRY RUN — analysis doc %s:", akey)
        print(json.dumps(analysis, indent=2, default=str))

    # ---- run metrics: dedup + complexity + token counters (logged + persisted) ----
    try:
        import token_usage as _tu
        tok = _tu.totals()
        run_metrics = {
            "kind": "analysis_run_metrics",
            "job_name": identity["name"], "display_name": identity["display_name"],
            "os": identity["os"], "component": identity["component"],
            "build": identity["build"], "build_id": build_id, "model": args.model,
            "failures_total":        len(failures),
            "droid_calls":           n_calls,               # total tokened calls
            "droid_calls_fresh":     n_calls - n_vector_aug,
            "droid_calls_augmented": n_vector_aug,          # high-complexity near-dups re-run w/ context
            "exact_dedup_saved":     n_exact,
            "vector_reuse":          n_vector,              # low-complexity near-dups reused (0 tokens)
            "capped":                n_capped,
            "saved_calls":           n_exact + n_vector,    # droid calls avoided by dedup
            "vector_dedup_enabled":  vec_on,
            "sim_threshold":         VECTOR_SIM_THRESHOLD,
            "candidates_fetched":    cand_fetched,
            "candidates_high":       cand_high,
            "candidates_low":        cand_low,
            "pushed_docs":           len(summary_docs),
            "pushed_high":           pushed_high,
            "pushed_low":            pushed_low,
            "tokens_input":          tok.get("input_tokens", 0),
            "tokens_output":         tok.get("output_tokens", 0),
            "tokens_total":          tok.get("input_tokens", 0) + tok.get("output_tokens", 0),
        }
        if store:
            _tu.record_metrics("arun_" + akey.split("_", 1)[1] + f"_{build_id}", run_metrics)
        logger.info("Run metrics: droid=%d (fresh=%d, aug=%d) | exact=%d vector-reuse=%d capped=%d | "
                    "pushed high/low=%d/%d | candidates high/low=%d/%d | tokens=%d",
                    n_calls, n_calls - n_vector_aug, n_vector_aug, n_exact, n_vector, n_capped,
                    pushed_high, pushed_low, cand_high, cand_low, run_metrics["tokens_total"])
    except Exception as _mexc:  # metrics must never fail the run
        logger.warning("metrics record failed (non-fatal): %s", _mexc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
