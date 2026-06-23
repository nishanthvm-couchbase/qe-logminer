#!/usr/bin/python3
"""
Couchbase storage + queries for the Test Analysis feature.

All buckets live on the SAME temp greenboard cluster (172.23.105.219):
  - test_analysis : our two doc tiers (summary + analysis)
  - server        : the collector's per-run docs (used for job-level pass/fail trend)

Key formulas (the greenboard backend MUST use the same analysis key formula):
  Tier-1 summary :  tfa_<md5("{job_name}-{build_id}-{test_name}")>
  Tier-2 analysis:  analysis_<md5("{os}|{component}|{name}|{build}")>
where `job_name`/`name` are the greenboard name-WITH-variants.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions, QueryOptions
from couchbase.auth import PasswordAuthenticator

logger = logging.getLogger(__name__)

ANALYSIS_BUCKET = "test_analysis"
SERVER_BUCKET   = "server"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

def key_summary(job_name: str, build_id: Any, test_name: str) -> str:
    return "tfa_" + hashlib.md5(f"{job_name}-{build_id}-{test_name}".encode()).hexdigest()


def key_analysis(os_name: str, component: str, name: str, build: str) -> str:
    raw = f"{os_name}|{component}|{name}|{build}"
    return "analysis_" + hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class AnalysisStore:
    def __init__(self, host: str, user: str = "Administrator",
                 password: str = "esabhcuoc", timeout: int = 20) -> None:
        to = ClusterTimeoutOptions(
            connect_timeout=timedelta(seconds=timeout),
            kv_timeout=timedelta(seconds=timeout),
            query_timeout=timedelta(seconds=timeout),
        )
        self._cluster = Cluster(
            f"couchbase://{host}",
            ClusterOptions(PasswordAuthenticator(user, password), timeout_options=to),
        )
        self._cluster.wait_until_ready(timedelta(seconds=timeout))
        self._cols: Dict[str, Any] = {}

    def _col(self, bucket: str) -> Any:
        if bucket not in self._cols:
            self._cols[bucket] = self._cluster.bucket(bucket).default_collection()
        return self._cols[bucket]

    # --- writes ---

    def upsert(self, bucket: str, key: str, doc: Dict[str, Any], retries: int = 5) -> bool:
        col = self._col(bucket)
        for attempt in range(1, retries + 1):
            try:
                col.upsert(key, doc)
                return True
            except Exception as exc:
                logger.warning("upsert %s attempt %d/%d failed: %s", key, attempt, retries, exc)
        return False

    def upsert_summary(self, key: str, doc: Dict[str, Any]) -> bool:
        return self.upsert(ANALYSIS_BUCKET, key, doc)

    def upsert_analysis(self, key: str, doc: Dict[str, Any]) -> bool:
        return self.upsert(ANALYSIS_BUCKET, key, doc)

    # --- reads ---

    def get(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        try:
            return self._col(bucket).get(key).value
        except Exception:
            return None

    def get_analysis(self, key: str) -> Optional[Dict[str, Any]]:
        return self.get(ANALYSIS_BUCKET, key)

    def _query(self, statement: str, **named) -> List[Dict[str, Any]]:
        """Run N1QL; degrade to [] on any error (missing index, etc.) so the
        pipeline never hard-fails on best-effort history."""
        try:
            res = self._cluster.query(statement, QueryOptions(named_parameters=named))
            return list(res)
        except Exception as exc:
            logger.warning("N1QL failed (%s): %s", statement.split("WHERE")[0].strip(), exc)
            return []

    def related_summaries(self, job_name: str, build: str) -> List[Dict[str, Any]]:
        """All Tier-1 summary docs for this (job, product-build), across reruns/build_ids."""
        # `build` is a reserved word in N1QL → must be backticked.
        stmt = (
            f"SELECT a.* FROM `{ANALYSIS_BUCKET}` a "
            f"WHERE a.type = 'test_failure_analysis' "
            f"AND a.job_name = $job AND a.`build` = $build"
        )
        return self._query(stmt, job=job_name, build=build)

    def test_failure_history(self, job_name: str, limit: int = 400) -> List[Dict[str, Any]]:
        """Per-test failure records for this job across ALL builds (for first-seen /
        recurring / streak signals). Returns {test_name, build, build_id, category}."""
        stmt = (
            f"SELECT a.test_name, a.`build`, a.build_id, a.category "
            f"FROM `{ANALYSIS_BUCKET}` a "
            f"WHERE a.type = 'test_failure_analysis' AND a.job_name = $job "
            f"LIMIT {int(limit)}"
        )
        return self._query(stmt, job=job_name)

    def job_trend(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Job-level pass/fail per recent build from the server bucket (name = with-variants).
        Best-effort: needs a server-bucket index; returns [] if unavailable."""
        stmt = (
            f"SELECT s.`build`, s.`result`, s.totalCount, s.failCount, s.build_id "
            f"FROM `{SERVER_BUCKET}` s "
            f"WHERE s.name = $name "
            f"ORDER BY s.`build` DESC LIMIT {int(limit) * 4}"
        )
        return self._query(stmt, name=name)
