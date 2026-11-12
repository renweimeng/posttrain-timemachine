"""Portable installed-package smoke test. Runs on macOS, Linux and Windows."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from ptm import __version__
from ptm.api import create_app
from ptm.demo import create_demo
from ptm.storage import Store

with tempfile.TemporaryDirectory(prefix="ptm-smoke-") as directory:
    root = Path(directory) / "workspace"
    store = Store(root)
    create_demo(store)
    with TestClient(create_app(root)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/v1/runs").json()["items"]
        assert client.get("/openapi.json").json()["info"]["version"] == __version__
    result = subprocess.run(
        [sys.executable, "-I", "-m", "ptm", "doctor", "demo-format-truth", "--root", str(root)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    assert json.loads(result.stdout)["capabilities"]["reward_diff"]
    assert "torch" not in sys.modules
    print(json.dumps({"version": __version__, "platform": sys.platform,
                      "demo_api_cli": "passed", "torch_imported": False}))
