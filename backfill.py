#!/usr/bin/python3
"""
Backfill / reconciliation for the Test Analysis feature.

Drives the analysis pipeline over a window of already-collected builds so the
`test_analysis` bucket ends up looking like we'd been running live all along.
Doubles as a catch-up tool: if the downstream job (or Jenkins) was down, re-run
this for the gap with --skip-existing.

HOW IT WORKS
────────────
It needs NO Jenkins API for the analysis itself — greenboard already holds the
identity (os/component/name/build/build_id) and the logs persist in S3:
    http://cb-logs-qe.s3-website-us-west-2.amazonaws.com/<build>/jenkins_logs/<project>/<build_id>/consoleText.txt
(Jenkins purges consoles after ~5 days, so S3 is the durable source.)

  1. greenboard: last N product builds of <version>  → their `{build}_server` docs
  2. each doc → failing runs (os, component, name-with-variants, build_id, executor url)
  3. per run → S3 console URL + identity → trigger the `test-analysis-runner` Jenkins
     job in backfill mode (fans out across every droid machine)
  4. processed BUILD-BY-BUILD ASCENDING, waiting for each build's batch to finish
     before the next — so per-job history (is_new / first_failed_build / regression)
     is computed against the builds below it.

  --skip-existing : skip (job, build) that already has an analysis doc  (catch-up)
  --dry-run       : print the plan; trigger nothing
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from urllib.parse import urlparse

import requests

from analysis_store import AnalysisStore, key_analysis
from build_analysis import build_sort_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("backfill")

S3_BASE = "http://cb-logs-qe.s3-website-us-west-2.amazonaws.com"
NON_RUN_RESULTS = {"SUCCESS", "PENDING", None}


# ---------------------------------------------------------------------------
# Enumeration helpers
# ---------------------------------------------------------------------------

def project_name_from_url(url: str):
    """'http://qe-jenkins1…/job/test_suite_executor-TAF/' -> 'test_suite_executor-TAF'."""
    if url and "/job/" in url:
        return url.split("/job/", 1)[1].strip("/").split("/")[0]
    return None


def s3_console_url(build: str, project: str, build_id) -> str:
    return f"{S3_BASE}/{build}/jenkins_logs/{project}/{build_id}/consoleText.txt"


def enumerate_failing_runs(build_doc: dict, build: str):
    """Every non-success, non-deleted run in a greenboard build doc."""
    out = []
    for osn, comps in (build_doc.get("os") or {}).items():
        for comp, jobs in (comps or {}).items():
            for jobname, runlist in (jobs or {}).items():
                if not isinstance(runlist, list):
                    continue
                for r in runlist:
                    if r.get("deleted"):
                        continue
                    if r.get("result") in NON_RUN_RESULTS:
                        continue
                    bid, url = r.get("build_id"), r.get("url")
                    if not bid or not url:
                        continue
                    proj = project_name_from_url(url)
                    if not proj:
                        continue
                    out.append({
                        "os": osn, "component": comp, "name": jobname,
                        "display_name": r.get("displayName") or jobname,
                        "build": build, "build_id": bid, "result": r.get("result"),
                        "console_url": s3_console_url(build, proj, bid),
                    })
    return out


# ---------------------------------------------------------------------------
# Jenkins dispatch
# ---------------------------------------------------------------------------

class Jenkins:
    def __init__(self, base, user, token, runner):
        self.base = base.rstrip("/")
        self.runner = runner
        self.s = requests.Session()
        if user and token:
            self.s.auth = (user, token)
        self._crumb = self._get_crumb()

    def _get_crumb(self):
        try:
            r = self.s.get(f"{self.base}/crumbIssuer/api/json", timeout=15)
            if r.status_code == 200:
                d = r.json()
                return {d["crumbRequestField"]: d["crumb"]}
        except Exception:
            pass
        return {}

    def trigger(self, params: dict):
        """Trigger a runner build; return the queue-item URL so we can track its result."""
        r = self.s.post(f"{self.base}/job/{self.runner}/buildWithParameters",
                        params=params, headers=self._crumb, timeout=30)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"trigger HTTP {r.status_code}: {r.text[:200]}")
        return r.headers.get("Location")     # .../queue/item/<id>/

    def wait_batch(self, queue_urls, poll: int):
        """Wait for every triggered build to finish; return the list of results
        ('SUCCESS' / 'FAILURE' / 'CANCELLED' / 'UNKNOWN'). FAILURE = analyze_build
        exited non-zero (droid failure under fail-fast, or a crash)."""
        results, build_urls = [], {}
        pending = {q: 0 for q in queue_urls if q}
        MAX_RESOLVE = 90                      # ~MAX_RESOLVE*poll secs to get a build number
        # Phase 1: queue item -> build URL
        while pending:
            nxt = {}
            for qu, tries in pending.items():
                try:
                    d = self.s.get(qu.rstrip('/') + "/api/json", timeout=15).json()
                except Exception:
                    d = None
                if d and d.get("cancelled"):
                    results.append("CANCELLED")
                elif d and d.get("executable"):
                    build_urls[qu] = d["executable"]["url"]
                elif tries + 1 >= MAX_RESOLVE:
                    logger.warning("  gave up resolving %s", qu); results.append("UNKNOWN")
                else:
                    nxt[qu] = tries + 1
            pending = nxt
            if pending:
                time.sleep(poll)
        # Phase 2: wait for builds to finish
        waiting = dict(build_urls)
        while waiting:
            for qu, burl in list(waiting.items()):
                try:
                    d = self.s.get(burl.rstrip('/') + "/api/json",
                                   params={"tree": "building,result"}, timeout=15).json()
                except Exception:
                    continue
                if d and not d.get("building") and d.get("result"):
                    results.append(d["result"]); del waiting[qu]
            if waiting:
                time.sleep(poll)
        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Backfill Test Analysis over a window of builds.")
    ap.add_argument("--version", required=True, help="e.g. 8.1.0")
    ap.add_argument("--builds", type=int, default=10, help="how many most-recent builds (default 10)")
    ap.add_argument("--build-list", help="explicit comma-separated builds (overrides --builds)")
    ap.add_argument("--skip-existing", action="store_true", help="skip (job,build) already analyzed")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; trigger nothing")
    ap.add_argument("--ignore-droid-failure", action="store_true",
                    help="don't stop the backfill when runs fail (default: STOP on the first "
                         "failed batch — usually droid out of tokens — so you can resume with "
                         "--skip-existing after restoring tokens)")
    ap.add_argument("--poll-interval", type=int, default=20)
    # couchbase
    ap.add_argument("--cb-host", default=os.environ.get("CB_HOST", "172.23.105.219"))
    ap.add_argument("--cb-user", default=os.environ.get("CB_USER", "Administrator"))
    ap.add_argument("--cb-pass", default=os.environ.get("CB_PASS", "esabhcuoc"))
    # jenkins
    ap.add_argument("--jenkins-url", default=os.environ.get("JENKINS_URL", "http://qe-jenkins1.sc.couchbase.com"))
    ap.add_argument("--jenkins-user", default=os.environ.get("JENKINS_USER", ""))
    ap.add_argument("--jenkins-token", default=os.environ.get("JENKINS_TOKEN", ""))
    ap.add_argument("--runner-job", default=os.environ.get("RUNNER_JOB", "test-analysis-runner"))
    args = ap.parse_args()

    store = AnalysisStore(args.cb_host, args.cb_user, args.cb_pass)

    # --- choose builds (ascending) ---
    if args.build_list:
        builds = [b.strip() for b in args.build_list.split(",") if b.strip()]
    else:
        all_builds = store.list_version_builds(args.version)
        if not all_builds:
            logger.error("No greenboard builds found for %s (need a primary/GSI index on "
                         "`greenboard`?). Use --build-list to bypass.", args.version)
            return 1
        all_builds = sorted(set(all_builds), key=build_sort_key)
        builds = all_builds[-args.builds:]
    builds = sorted(builds, key=build_sort_key)
    logger.info("Target builds (ascending): %s", builds)

    jenkins = None
    if not args.dry_run:
        jenkins = Jenkins(args.jenkins_url, args.jenkins_user, args.jenkins_token, args.runner_job)

    grand_total = 0
    for build in builds:                       # ASCENDING — history correctness
        doc = store.get_build_doc(build)
        if not doc:
            logger.warning("No greenboard doc for %s_server — skipping", build)
            continue
        runs = enumerate_failing_runs(doc, build)

        # skip-existing: drop runs whose (job,build) already has an analysis doc
        if args.skip_existing and runs:
            done = set()
            checked = {}
            kept = []
            for r in runs:
                jk = (r["os"], r["component"], r["name"])
                if jk not in checked:
                    checked[jk] = store.get_analysis(
                        key_analysis(r["os"], r["component"], r["name"], build)) is not None
                if checked[jk]:
                    done.add(jk)
                else:
                    kept.append(r)
            if done:
                logger.info("  %s: skipping %d already-analyzed job(s)", build, len(done))
            runs = kept

        logger.info("Build %s: %d failing run(s) to dispatch", build, len(runs))
        grand_total += len(runs)
        if not runs:
            continue

        if args.dry_run:
            for r in runs[:5]:
                logger.info("    would trigger: %s/%s  console=%s",
                            r["component"], r["name"], r["console_url"])
            if len(runs) > 5:
                logger.info("    … +%d more", len(runs) - 5)
            continue

        queue_urls = []
        for r in runs:
            params = {
                "CONSOLE_URL": r["console_url"], "OS": r["os"],
                "COMPONENT": r["component"], "NAME": r["name"],
                "DISPLAY_NAME": r["display_name"], "BUILD": r["build"],
                "BUILD_ID": str(r["build_id"]),
                "IGNORE_DROID_FAILURE": "1" if args.ignore_droid_failure else "0",
            }
            try:
                queue_urls.append(jenkins.trigger(params))
            except Exception as exc:
                logger.warning("  trigger failed for %s/%s: %s", r["component"], r["name"], exc)

        logger.info("Build %s: dispatched %d — waiting for the batch to finish…", build, len(queue_urls))
        results = jenkins.wait_batch(queue_urls, args.poll_interval)
        failed = sum(1 for x in results if x == "FAILURE")
        logger.info("Build %s: %d done, %d FAILURE, %d other",
                    build, len(results), failed, len(results) - failed - results.count("SUCCESS"))

        if failed and not args.ignore_droid_failure:
            logger.error("%d run(s) FAILED in build %s — almost certainly droid out of tokens. "
                         "STOPPING the backfill. Restore tokens, then re-run the SAME command with "
                         "--skip-existing to resume (completed jobs are skipped).", failed, build)
            return 2

    logger.info("Done. %s %d run(s) across %d build(s).",
                "Would dispatch" if args.dry_run else "Dispatched", grand_total, len(builds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
