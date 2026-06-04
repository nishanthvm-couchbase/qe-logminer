# qe-logminer

Parses Jenkins `consoleText` logs from Couchbase QE testrunner jobs and extracts structured failure data for downstream pipelines.

## What it does

For each test in a Jenkins build log it determines pass / fail / abort, then for every failed test it captures:
- the test name and input params
- the Python traceback
- `error_lines` — a compact slice of the filtered log containing only ERROR/WARNING lines and ±3 lines of context around each one

The output is a single JSON file, small enough to feed directly into an LLM or analysis pipeline.

## Usage

```bash
# from a local file
python3 parse_console_log.py consoleText

# from a Jenkins URL
python3 parse_console_log.py http://qe-jenkins1.sc.couchbase.com/job/test_suite_executor/44388/consoleText
```

Output is written to `parse_result_<source>.json` in the current directory.

## Output format

```json
{
  "source": "consoleText",
  "total_tests": 15,
  "passed": 9,
  "failed": 6,
  "aborted": 0,
  "failed_tests": [
    {
      "test_name": "newupgradetests.MultiNodesUpgradeTests.test_offline_upgrade_with_add_new_services",
      "params": "{'initial_version': '7.6.7-6706', 'after_upgrade_services_in': 'eventing', ...}",
      "traceback": "Traceback (most recent call last):\n  ...\nException: 13 out of 20 queries failed!",
      "error_lines": "... ERROR and WARNING lines with surrounding context ..."
    }
  ]
}
```

### `error_lines` pipeline

```
raw consoleText
    └── parse_tests()              # split into per-test blocks
        └── filter_log()           # strip known infra noise  →  full_log
            └── extract_error_lines()  # ERROR/WARNING anchors + ±3 context  →  error_lines
```

`error_lines` is always a strict subset of `full_log` lines.

## Extending for other components

The core parsing logic is generic — it only looks for standard testrunner markers (`Test Input params:`, `Ran 1 test in Xs`). Component-specific tuning is done entirely through `NOISE_PATTERNS` at the top of the script.

If a new component's log has recurring infrastructure noise surviving into `error_lines`, add a pattern there:

```python
NOISE_PATTERNS = [
    ...
    re.compile(r".*<component-specific noise pattern>"),
]
```

## Requirements

```bash
pip install requests
```
