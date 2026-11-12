"""A real two-process CPU GRPO run, not a fabricated rank merge fixture."""
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from ptm.storage import Store, doctor


@pytest.mark.trl
def test_actual_two_rank_grpo_capture_and_global_advantage_parity(tmp_path):
    pytest.importorskip("trl")
    project = Path(__file__).resolve().parents[1]
    with socket.socket() as socket_:
        socket_.bind(("127.0.0.1", 0))
        port = socket_.getsockname()[1]
    output = tmp_path / "distributed"
    root = tmp_path / "store"
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
         "--master_addr=127.0.0.1", f"--master_port={port}",
         str(project / "examples/quickstart-trl/train_distributed.py"),
         "--output", str(output), "--root", str(root)],
        cwd=project, env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout[-12000:]
    result = json.loads((output / "result.json").read_text())
    assert result["reference_parity"] and result["ranks"] == 2
    assert result["groups"] == 1 and result["completions"] == 4
    snapshot = Store(root).load("tiny-trl-distributed")
    assert {c.worker_rank for c in snapshot.completions} == {0, 1}
    assert snapshot.run.drop_count == 0
    assert doctor(snapshot)["capabilities"]["advantage"]
