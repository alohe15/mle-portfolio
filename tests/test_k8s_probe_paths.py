"""Release probes must name real GET routes.

A readiness path such as /readyz-does-not-exist fails this test before a
release, which is the fault injected in the release-incident drill.
"""

from __future__ import annotations

import re
from pathlib import Path

from services.api.app import app

DEPLOYMENT = Path(__file__).resolve().parent.parent / "deploy" / "k8s" / "local" / "deployment.yaml"
PROBE_HEADER = re.compile(r"^(livenessProbe|readinessProbe):\s*$")
PATH_LINE = re.compile(r"^path:\s*(\S+)\s*$")


def probe_http_paths(text: str) -> dict[str, str]:
    """httpGet path for each probe, read from the deployment text."""
    paths: dict[str, str] = {}
    probe: str | None = None
    in_http = False
    for raw in text.splitlines():
        line = raw.strip()
        header = PROBE_HEADER.match(line)
        if header:
            probe = header.group(1)
            in_http = False
            continue
        if probe is None:
            continue
        if line == "httpGet:":
            in_http = True
            continue
        if in_http:
            found = PATH_LINE.match(line)
            if found:
                paths[probe] = found.group(1)
                in_http = False
    return paths


def test_probe_paths_are_real_get_routes():
    paths = probe_http_paths(DEPLOYMENT.read_text())
    assert set(paths) == {"livenessProbe", "readinessProbe"}
    get_routes = {
        route.path
        for route in app.routes
        if "GET" in (getattr(route, "methods", None) or set())
    }
    for probe, path in paths.items():
        assert path in get_routes, probe
