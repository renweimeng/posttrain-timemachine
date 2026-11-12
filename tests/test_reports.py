import io
import zipfile
from pathlib import Path

import pytest

from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, GroupFilter, load_spec, run_analysis
from ptm.reports import build_bundle, verify_bundle
from ptm.storage import Store


def test_deterministic_export_offline_recompute_and_tamper(tmp_path):
    snapshot = demo_snapshot()
    snapshot.run.config["api_key"] = "this-secret-must-not-escape"
    store = Store(tmp_path/"data")
    store.commit(snapshot)
    spec = load_spec(Path(__file__).resolve().parents[1]/"examples/format-over-truth/patch.yaml")
    request = AnalysisRequest(run_id=snapshot.run.run_id, reward_spec=spec,
                              group_filter=GroupFilter(group_ids=[snapshot.groups[0].group_id]))
    result = run_analysis(store, request)
    a, b = build_bundle(store, result["analysis_id"]), build_bundle(store, result["analysis_id"])
    assert a == b
    path = tmp_path/"report.zip"
    path.write_bytes(a)
    assert verify_bundle(path, recompute=True)["verified"]
    with zipfile.ZipFile(io.BytesIO(a)) as archive:
        contents = {n: archive.read(n) for n in archive.namelist()}
    assert all(b"this-secret-must-not-escape" not in v for v in contents.values())
    assert b"retrain validation pending" in contents["report.md"].lower()
    contents["report.md"] += b"tamper"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    with pytest.raises(PTMError, match="Checksum"):
        verify_bundle(path)


def test_archive_traversal_never_extracted(tmp_path):
    path = tmp_path/"evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../outside", "bad")
    with pytest.raises(PTMError, match="archive"):
        verify_bundle(path)
    assert not (tmp_path.parent/"outside").exists()
