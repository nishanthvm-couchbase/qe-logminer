#!/usr/bin/python3
"""
Reconstruct a job's greenboard identity from a Jenkins executor build's params.

WHY THIS EXISTS
───────────────
The downstream analysis job must key its docs by the SAME identity greenboard
displays, otherwise the "Test Analysis" button can't find them. greenboard rows
are keyed by name-WITH-variants (e.g. "debian-2i_..._bucket_storage=COUCHSTOREGSI_type=PLASMA")
under (os, component, build). That identity is produced by the collector from the
executor build's params.

This module replicates the collector's name/variant construction so the analysis
pipeline derives the exact same strings.

  ⚠ MUST stay in sync with jinja/collector/parsing.py:
      build_test_name(), get_variants(), add_variants_to_name(), parse_build_version()
    Kept as a small self-contained copy (not an import) so the downstream job is
    portable to droid machines without the whole jinja package.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DEFAULT_ARCHITECTURE   = "x86_64"
DEFAULT_BUCKET_STORAGE = "COUCHSTORE"
DEFAULT_GSI_TYPE       = "PLASMA"
# Components for which GSI_type defaults to PLASMA (raw param values — see collector fix).
GSI_COMPONENTS = {"2I", "2I_MOI", "2I_REBALANCE", "GSI", "MEMDB", "PLASMA"}

DEFAULT_BUILD_PARAM_NAMES = [
    "version_number", "cluster_version", "build",
    "COUCHBASE_SERVER_VERSION", "columnar_version_number", "cbs_ver",
]

_VERSION_RE  = re.compile(r"^\d\.\d\.\d{1,5}")
_BUILD_NO_RE = re.compile(r"^\d{1,10}")


# ---------------------------------------------------------------------------
# Jenkins action / parameter extraction (mirrors collector get_action)
# ---------------------------------------------------------------------------

def get_action(actions: Any, key: str, value: Optional[str] = None) -> Optional[Any]:
    if not actions:
        return None
    for a in actions:
        if a is None:
            continue
        if hasattr(a, "keys"):
            keys = a.keys()
        elif a and hasattr(a[0], "keys"):
            keys = a[0].keys()
        else:
            continue
        if "urlName" in keys and a.get("urlName") not in ("robot", "testReport", "tapTestReport"):
            continue
        if key in keys:
            if value is not None:
                if a.get("name") == value:
                    return a.get("value")
            else:
                return a[key]
    return None


def extract_params(actions: Any) -> Any:
    params = get_action(actions, "parameters")
    if params is None and actions and not hasattr(actions, "keys"):
        for a in actions:
            if not hasattr(a, "keys"):
                return a
    return params


# ---------------------------------------------------------------------------
# Version / variant / name construction (mirrors collector)
# ---------------------------------------------------------------------------

def parse_build_version(raw: str) -> Optional[str]:
    raw = raw.replace("-rel", "").split(",")[0].strip()
    try:
        parts = raw.split("-")
        if len(parts) < 2:
            return None
        rel, bno = parts[0], parts[1]
        while rel.count(".") < 2:
            rel += ".0"
        if not _VERSION_RE.match(rel) or not _BUILD_NO_RE.match(bno):
            return None
        return f"{rel}-{bno.zfill(4)}"
    except Exception:
        return None


def get_build(params: Any, param_names: List[str] = None) -> Optional[str]:
    for name in (param_names or DEFAULT_BUILD_PARAM_NAMES):
        raw = get_action(params, "name", name)
        if raw:
            build = parse_build_version(raw)
            if build:
                return build
    return None


def _get_variant_from_params(name: str, params: Any) -> Optional[str]:
    raw = get_action(params, "name", "parameters")
    if not raw:
        return None
    for part in raw.split(","):
        if part.startswith(name):
            pieces = part.split("=")
            return pieces[1].upper() if len(pieces) > 1 else None
    return None


def get_variants(params: Any, component: str) -> Dict[str, str]:
    storage = _get_variant_from_params("bucket_storage", params) or DEFAULT_BUCKET_STORAGE
    gsi = _get_variant_from_params("gsi_type", params)
    if gsi is None:
        gsi = DEFAULT_GSI_TYPE if (component or "").upper() in GSI_COMPONENTS else "UNDEFINED"
    return {"bucket_storage": storage, "GSI_type": gsi}


def add_variants_to_name(doc_name: str, variants: Dict[str, str]) -> str:
    for k, v in variants.items():
        doc_name += f"{k}={v}"
    return doc_name


def build_test_name(params: Any, fallback_os: Optional[str] = None) -> Optional[str]:
    """<os>-<component>_<subcomponent> from raw params (pre-variant displayName)."""
    component = get_action(params, "name", "component")
    if not component:
        test_yml = get_action(params, "name", "test")
        if test_yml and ".yml" in test_yml:
            import os as _os
            stem = _os.path.splitext(_os.path.basename(test_yml.split()[-1]))[0]
            component = f"systest-{stem}"
    if not component:
        return None
    os_param = (get_action(params, "name", "OS") or
                get_action(params, "name", "os") or fallback_os or "")
    arch = get_action(params, "name", "arch")
    if arch and arch != DEFAULT_ARCHITECTURE:
        os_param = f"{os_param}-{arch}"
    subcomponent = get_action(params, "name", "subcomponent") or "server"
    return f"{os_param}-{component}_{subcomponent}"


# ---------------------------------------------------------------------------
# Public: full identity from a build's actions
# ---------------------------------------------------------------------------

def identity_from_actions(
    actions: Any, build_id: Optional[int] = None,
    build_param_names: List[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the greenboard identity for an executor build, or None if it isn't a
    real per-test executor run (no component param / no build).

    Keys: os, component, display_name (no variants), name (with variants),
          variants, build (product, e.g. 8.1.0-2299), build_id.
    """
    params = extract_params(actions)
    if not params:
        return None

    component = get_action(params, "name", "component")
    if not component:
        return None  # not a categorizable test run
    component = component.upper()

    os_name = (get_action(params, "name", "OS") or get_action(params, "name", "os"))
    if os_name:
        os_name = os_name.upper()

    display_name = build_test_name(params, fallback_os=os_name)
    if not display_name:
        return None

    arch = get_action(params, "name", "arch")
    if arch and arch != DEFAULT_ARCHITECTURE and os_name:
        os_name = f"{os_name}-{arch}".upper()

    build = get_build(params, build_param_names)
    if not build:
        return None

    variants = get_variants(params, component)
    name = add_variants_to_name(display_name, variants)

    return {
        "os": os_name,
        "component": component,
        "display_name": display_name,
        "name": name,
        "variants": variants,
        "build": build,
        "build_id": build_id,
    }
