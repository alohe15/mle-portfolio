#!/usr/bin/env python3
"""Query every Fraud API dashboard expression against local Prometheus."""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DASHBOARD = Path("deploy/monitoring/grafana/dashboards/fraud-api.json")
QUERY_URL = "http://localhost:9090/api/v1/query"


def iter_panels(node):
    if isinstance(node, dict):
        if "targets" in node and "title" in node:
            yield node
        for value in node.values():
            yield from iter_panels(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_panels(item)


def query(expr):
    url = QUERY_URL + "?" + urllib.parse.urlencode({"query": expr})
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        return "ERROR " + str(payload.get("error", "query failed"))
    result = payload.get("data", {}).get("result", [])
    if not result:
        return "EMPTY"
    parts = []
    for series in result:
        metric = series.get("metric", {})
        labels = ",".join(
            f'{key}="{value}"' for key, value in sorted(metric.items()) if key != "__name__"
        )
        value = series.get("value", [None, None])[1]
        parts.append(f"{{{labels}}} {value}" if labels else str(value))
    return "; ".join(parts)


def main():
    try:
        dashboard = json.loads(DASHBOARD.read_text())
        found = False
        for panel in iter_panels(dashboard):
            for target in panel.get("targets", []):
                expr = target.get("expr")
                if not expr:
                    continue
                found = True
                legend = target.get("legendFormat", "")
                try:
                    outcome = query(expr)
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                    outcome = f"ERROR {exc}"
                print(f"{panel.get('title', '')}\t{legend}\t{outcome}")
        if not found:
            print("ERROR\t\tno panel expressions found")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR\t\t{exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
