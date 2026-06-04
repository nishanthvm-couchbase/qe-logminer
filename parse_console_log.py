#!/usr/bin/python3

import json
import os
import re
import sys

import requests

test_complete_pattern = re.compile(r"Ran 1 test in (\d+\.\d+s)")
test_report_delimiter = '=' * 70
build_aborted_pattern = re.compile(
    r"Build timed out \(after ([0-9]+ [a-zA-Z]+)\)\. "
    r"Marking the build as aborted.")

# Noise patterns to strip from test logs
NOISE_PATTERNS = [
    re.compile(r".*SSH Connecting to .* with username"),
    re.compile(r".*SSH Connected to .* as "),
    re.compile(r".*os_distro: .*, os_version: .*, is_linux_distro"),
    re.compile(r".*extract_remote_info-->distribution_type"),
    re.compile(r".*CryptographyDeprecationWarning"),
    re.compile(r".*\"cipher\": algorithms\.TripleDES"),
    re.compile(r".*\"class\": algorithms\.TripleDES"),
    re.compile(r".*command executed successfully with root"),
    re.compile(r".*running command\.raw on \d+\.\d+\.\d+\.\d+"),
    re.compile(r".*is_ns_server_running"),
    re.compile(r".*waiting for ns_server"),
    re.compile(r".*ns_server @ .* is running"),
    re.compile(r".*Trying to check is this url alive"),
    re.compile(r".*This url .* is live"),
    re.compile(r".*closing all ssh connections"),
    re.compile(r".*closing all memcached connections"),
    re.compile(r"Cluster instance shutdown with force"),
    re.compile(r".*sleep for \d+ secs\. sleep \d+ seconds before run next test"),
    re.compile(r".*socket error while connecting to .*Connection refused"),
    re.compile(r".*GET http://.*pools/default.*unknown pool"),
    re.compile(r".*with status False: unknown pool"),
    re.compile(r".*Collecting logs from \d+\.\d+"),
    re.compile(r".*cbcollect_info .* diag\.zip"),
    re.compile(r".*dpkg: warning: while removing couchbase-server"),
    re.compile(r".*Directory at /tmp DOES exist\. Fx returns True"),
]


ERROR_LEVEL_PATTERN = re.compile(
    r"\] (?:ERROR|WARNING|WARN|CRITICAL) - "
    r"|Traceback \(most recent call last\)"
    r"|^Exception:"
)


def is_noise(line):
    for pattern in NOISE_PATTERNS:
        if pattern.match(line):
            return True
    return False


def filter_log(lines):
    return [l for l in lines if not is_noise(l)]


def extract_error_lines(filtered_lines, context=3):
    """Derives from the already-filtered full_log lines, not the raw test block."""
    anchors = {i for i, l in enumerate(filtered_lines) if ERROR_LEVEL_PATTERN.search(l)}
    included = set()
    for idx in anchors:
        for j in range(max(0, idx - context), min(len(filtered_lines), idx + context + 1)):
            included.add(j)
    return [filtered_lines[i] for i in sorted(included)]


def fetch_content(source):
    if os.path.isfile(source):
        with open(source, "r") as f:
            return f.read()
    resp = requests.get(source)
    resp.raise_for_status()
    return resp.text


def parse_tests(content):
    lines = content.split("\n")
    tests = []
    current_test_lines = []
    in_test = False
    test_num = 0
    lookback = 5

    for i, line in enumerate(lines):
        if line.startswith("Test Input params:"):
            if in_test and current_test_lines:
                tests.append(current_test_lines)
            test_num += 1
            in_test = True
            start = max(0, i - lookback)
            current_test_lines = lines[start:i] + [line]
            continue

        if in_test:
            current_test_lines.append(line)
            m = test_complete_pattern.match(line)
            if m:
                tests.append(current_test_lines)
                in_test = False
                current_test_lines = []
            elif build_aborted_pattern.findall(line):
                tests.append(current_test_lines)
                in_test = False
                current_test_lines = []

    if in_test and current_test_lines:
        tests.append(current_test_lines)

    return tests


def classify_test(test_lines):
    for line in test_lines:
        if test_report_delimiter in line:
            has_error_block = False
            for l2 in test_lines:
                if "FAILED" in l2 or "ERROR" in l2:
                    has_error_block = True
                    break
            if has_error_block:
                return "FAIL"
        if build_aborted_pattern.findall(line):
            return "ABORT"
    for line in test_lines:
        m = test_complete_pattern.match(line)
        if m:
            return "PASS"
    return "UNKNOWN"


def extract_test_name(test_lines):
    # Try ./testrunner -t module.Class.method pattern (from lookback lines)
    for line in test_lines:
        m = re.search(r"\./testrunner .* -t\s+(\S+?)(?:,|$)", line)
        if m:
            return m.group(1)
    # Try "test_method (module.Class) ..." pattern
    for line in test_lines:
        m = re.match(r"^(\S+)\s+\((\S+)\)", line)
        if m:
            return "%s.%s" % (m.group(2), m.group(1))
    # Try basetestcase setup line
    for line in test_lines:
        m = re.search(r"basetestcase setup was started for test #\d+\s+(\S+)", line)
        if m:
            return m.group(1)
    return "unknown_test"


def extract_test_params(test_lines):
    for i, line in enumerate(test_lines):
        if line.startswith("Test Input params:"):
            if i + 1 < len(test_lines):
                try:
                    return test_lines[i + 1]
                except Exception:
                    pass
    return ""


def extract_traceback(test_lines):
    tb_lines = []
    in_tb = False
    for line in test_lines:
        if line.startswith("Traceback (most recent call last):"):
            in_tb = True
            tb_lines = [line]
        elif in_tb:
            tb_lines.append(line)
            if line and not line.startswith(" ") \
                    and not line.startswith("Traceback"):
                in_tb = False
    return "\n".join(tb_lines)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 parse_console_log.py <console_log_url_or_file>")
        print("Example: python3 parse_console_log.py "
              "http://qe-jenkins1.sc.couchbase.com/job/"
              "test_suite_executor/44388/consoleText")
        sys.exit(1)

    source = sys.argv[1]

    print("=" * 80)
    print("Fetching: %s" % source)
    print("=" * 80)

    content = fetch_content(source)
    tests = parse_tests(content)

    results = {"source": source, "total_tests": len(tests),
               "passed": 0, "failed": 0, "aborted": 0,
               "failed_tests": []}

    for idx, test_lines in enumerate(tests):
        status = classify_test(test_lines)
        test_name = extract_test_name(test_lines)

        if status == "PASS":
            results["passed"] += 1
            print("Test #%d [PASS] %s" % (idx + 1, test_name))
        elif status == "FAIL":
            results["failed"] += 1
            filtered = filter_log(test_lines)
            traceback = extract_traceback(test_lines)
            params = extract_test_params(test_lines)
            error_lines = extract_error_lines(filtered)
            results["failed_tests"].append({
                "test_num": idx + 1,
                "test_name": test_name,
                "params": params,
                "traceback": traceback,
                "error_lines": "\n".join(error_lines),
                "full_log": "\n".join(filtered)
            })
            print("Test #%d [FAIL] %s" % (idx + 1, test_name))
        elif status == "ABORT":
            results["aborted"] += 1
            print("Test #%d [ABORT] %s" % (idx + 1, test_name))
        else:
            print("Test #%d [UNKNOWN] %s" % (idx + 1, test_name))

    # Build output filename
    if os.path.isfile(source):
        base = os.path.splitext(os.path.basename(source))[0]
    else:
        url_parts = source.rstrip("/").split("/")
        base = "_".join(url_parts[-3:]).replace(".", "_")
    output_file = "parse_result_%s.json" % base

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Summary: %d passed, %d failed, %d aborted (out of %d)"
          % (results["passed"], results["failed"],
             results["aborted"], results["total_tests"]))
    print("Results saved to: %s" % output_file)
    print("=" * 80)
    