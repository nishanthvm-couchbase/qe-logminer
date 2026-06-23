#!/usr/bin/python3
"""
Tier-2 builder: one analysis doc per (job, product-build).

Division of labour (deliberate):
  • CODE owns the NUMBERS  — counts, category breakdown, per-test history signals,
    job pass/fail trend. Exact, reproducible. Never trust an LLM to count.
  • DROID owns the PROSE   — headline, verdict (regression/flaky/infra/...), grouping
    failures into themes, recommended action. Synthesis & judgement only.

The analysis doc is rebuilt on every rerun (it's keyed by product-build), so droid is
shown the EXISTING doc and asked to update it while keeping the schema.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from summarize_failures import run_droid, extract_json_object

logger = logging.getLogger(__name__)

VALID_VERDICTS    = {"regression", "flaky", "infra", "mixed", "clean", "unknown"}
VALID_CONFIDENCE  = {"high", "medium", "low"}
HISTORY_BUILDS    = 10   # how many recent builds of trend to surface


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def build_sort_key(build: str):
    """'8.1.0-2299' -> (8,1,0,2299) so builds order correctly."""
    try:
        rel, bno = build.split("-")
        parts = [int(x) for x in rel.split(".")]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts) + (int(bno),)
    except Exception:
        return (0, 0, 0, 0)


def _dedupe_failures(summaries: List[Dict]) -> List[Dict]:
    """One row per test_name (keep the newest build_id's summary)."""
    by_test: Dict[str, Dict] = {}
    for s in summaries:
        t = s.get("test_name", "?")
        cur = by_test.get(t)
        if cur is None or str(s.get("build_id", "")) > str(cur.get("build_id", "")):
            by_test[t] = s
    return list(by_test.values())


def compute_stats(parse_result: Dict, failures: List[Dict]) -> Dict[str, Any]:
    by_cat: Dict[str, int] = {}
    for f in failures:
        c = f.get("category", "unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
    return {
        "total_tests": parse_result.get("total_tests", 0),
        "passed":      parse_result.get("passed", 0),
        "failed":      parse_result.get("failed", 0),
        "aborted":     parse_result.get("aborted", 0),
        "by_category": by_cat,
    }


def compute_history(failing_test_names: List[str], history_records: List[Dict],
                    current_build: str) -> Dict[str, Dict]:
    """For each currently-failing test, derive first-seen / recurrence from the
    failure history (Tier-1 docs across builds). Failure-only data, so we report
    what we can prove: it FAILED in these builds. (Pass history isn't tracked per
    testcase — droid decides regression-vs-flaky from this + the error nature.)"""
    builds_by_test: Dict[str, set] = {}
    for r in history_records:
        t = r.get("test_name")
        b = r.get("build")
        if t and b:
            builds_by_test.setdefault(t, set()).add(b)

    out: Dict[str, Dict] = {}
    for t in failing_test_names:
        builds = sorted(builds_by_test.get(t, {current_build}), key=build_sort_key)
        out[t] = {
            "is_new":             builds == [current_build],
            "times_failed":       len(builds),
            "first_failed_build": builds[0] if builds else current_build,
        }
    return out


def compute_trend(trend_records: List[Dict], limit: int = HISTORY_BUILDS) -> List[Dict]:
    """Best run per build → pass rate, newest first, capped to `limit`."""
    best: Dict[str, Dict] = {}
    for r in trend_records:
        b = r.get("build")
        if not b:
            continue
        total = r.get("totalCount") or 0
        fail  = r.get("failCount") or 0
        passed = max(total - fail, 0)
        prev = best.get(b)
        # keep the run with the most passes (greenboard's "best run")
        if prev is None or passed > prev["_passed"]:
            best[b] = {"build": b, "result": r.get("result"),
                       "pass_rate": round(passed / total, 4) if total else 0.0,
                       "total": total, "fail": fail, "_passed": passed}
    rows = sorted(best.values(), key=lambda x: build_sort_key(x["build"]), reverse=True)[:limit]
    for x in rows:
        x.pop("_passed", None)
    return rows


# ---------------------------------------------------------------------------
# Droid synthesis (prose only)
# ---------------------------------------------------------------------------

def _synthesis_prompt(identity: Dict, stats: Dict, failures: List[Dict],
                      trend: List[Dict], existing: Optional[Dict]) -> str:
    compact_failures = [{
        "test": f["test_name"], "category": f.get("category"),
        "summary": f.get("summary"), "root_cause": f.get("root_cause"),
        "is_new": f.get("is_new"), "times_failed": f.get("times_failed"),
        "first_failed_build": f.get("first_failed_build"),
    } for f in failures]

    existing_block = (json.dumps(existing.get("_ai", existing), indent=2)
                      if existing else "(none — first analysis for this build)")

    return f"""\
You are a Couchbase QE release analyst. Produce a concise ANALYSIS of one test job's
run for one product build. You are given exact numbers and per-test history (already
computed — do NOT recount), and an existing analysis to update if present.

Return ONLY a single JSON object, no prose/markdown, with EXACTLY these keys:
  "headline":           one sentence — the single most important takeaway.
  "verdict":            one of {sorted(VALID_VERDICTS)}.
  "themes":             array of {{ "title": str, "tests": [str], "explanation": str }}
                        grouping the failures by shared root cause (may be empty).
  "recommended_action": one concrete next step for the on-call/dev.
  "confidence":         "high" | "medium" | "low".

Guidance:
  - "regression" if failures look like real product breakage new to this build;
    "flaky" if intermittent/infra-ish/recurring-without-pattern; "infra" if env/CI;
    "mixed" if several kinds; "clean" only if nothing meaningful failed.
  - Use is_new / times_failed / first_failed_build to judge regression vs recurring.

JOB: {identity['name']}   OS: {identity['os']}   COMPONENT: {identity['component']}
BUILD: {identity['build']}

STATS (authoritative, do not recompute):
{json.dumps(stats, indent=2)}

RECENT PASS-RATE TREND (newest first):
{json.dumps(trend, indent=2)}

FAILURES (with per-test history):
{json.dumps(compact_failures, indent=2)}

EXISTING ANALYSIS TO UPDATE:
{existing_block}
"""


def _normalize_synthesis(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    verdict = str(obj.get("verdict", "unknown")).strip().lower()
    if verdict not in VALID_VERDICTS:
        verdict = "unknown"
    conf = str(obj.get("confidence", "low")).strip().lower()
    if conf not in VALID_CONFIDENCE:
        conf = "low"
    themes = obj.get("themes")
    if not isinstance(themes, list):
        themes = []
    clean_themes = []
    for t in themes:
        if isinstance(t, dict):
            clean_themes.append({
                "title": str(t.get("title", "")).strip(),
                "tests": [str(x) for x in t.get("tests", []) if x],
                "explanation": str(t.get("explanation", "")).strip(),
            })
    return {
        "headline": str(obj.get("headline", "")).strip() or "(no headline)",
        "verdict": verdict,
        "themes": clean_themes,
        "recommended_action": str(obj.get("recommended_action", "")).strip(),
        "confidence": conf,
    }


def synthesize(identity: Dict, stats: Dict, failures: List[Dict],
               trend: List[Dict], existing: Optional[Dict], model: str) -> Dict[str, Any]:
    if model and model != "__none__":
        prompt = _synthesis_prompt(identity, stats, failures, trend, existing)
        ok, obj = run_droid(prompt, model)
    else:
        ok, obj = False, None
    if not ok or obj is None:
        n_new = sum(1 for f in failures if f.get("is_new"))
        return {
            "headline": f"{stats['failed']} failed, {stats['aborted']} aborted "
                        f"({n_new} new) — automated synthesis unavailable",
            "verdict": "mixed" if stats["failed"] else "unknown",
            "themes": [], "recommended_action": "",
            "confidence": "low",
        }
    return _normalize_synthesis(obj)


# ---------------------------------------------------------------------------
# Public: assemble the Tier-2 doc
# ---------------------------------------------------------------------------

def build_analysis_doc(
    identity: Dict, parse_result: Dict, summaries: List[Dict],
    history_records: List[Dict], trend_records: List[Dict],
    existing: Optional[Dict], model: str, now_iso: str,
) -> Dict[str, Any]:
    failures = _dedupe_failures(summaries)
    hist = compute_history([f.get("test_name") for f in failures],
                           history_records, identity["build"])
    # merge tier-1 summary + history signals (code-owned) into each failure row
    merged = []
    for f in failures:
        h = hist.get(f.get("test_name"), {})
        merged.append({
            "test_name": f.get("test_name"),
            "category": f.get("category", "unknown"),
            "summary": f.get("summary", ""),
            "root_cause": f.get("root_cause", ""),
            "suggested_fix": f.get("suggested_fix", ""),
            "confidence": f.get("confidence", "low"),
            "build_id": f.get("build_id"),
            "is_new": h.get("is_new", True),
            "times_failed": h.get("times_failed", 1),
            "first_failed_build": h.get("first_failed_build", identity["build"]),
        })

    stats = compute_stats(parse_result, merged)
    trend = compute_trend(trend_records)
    ai = synthesize(identity, stats, merged, trend, existing, model)

    return {
        "type": "test_build_analysis",
        "schema_version": 1,
        # identity (code-owned; matches greenboard row)
        "os": identity["os"],
        "component": identity["component"],
        "name": identity["name"],
        "display_name": identity["display_name"],
        "build": identity["build"],
        "build_ids": sorted({f.get("build_id") for f in summaries if f.get("build_id")}),
        "generated_at": now_iso,
        # numbers (code-owned)
        "stats": stats,
        "trend": trend,
        "failures": merged,
        # narrative (droid-owned)
        "headline": ai["headline"],
        "verdict": ai["verdict"],
        "themes": ai["themes"],
        "recommended_action": ai["recommended_action"],
        "confidence": ai["confidence"],
        # keep the raw AI block so the next rerun can show droid its prior output
        "_ai": ai,
    }
