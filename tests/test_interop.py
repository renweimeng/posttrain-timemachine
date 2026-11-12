import json

import pytest

from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.importer import export_events
from ptm.interop import merge_rank_manifest, read_verl
from ptm.storage import file_hash


def test_explicit_verl_mapping_and_missing_group_refusal(tmp_path):
    s = demo_snapshot()
    s.run.framework = "verl"
    s.run.framework_version = "declared-source-1"
    b = s.batches[0]
    g = s.groups[0]
    header = {"ptm_run": s.run.model_dump(mode="json"), "field_map": {"group_id": "metadata.explicit_group"}}
    rows = []
    for c in s.completions[:4]:
        row = {**c.model_dump(mode="json"), **g.model_dump(mode="json"), **b.model_dump(mode="json")}
        row.pop("group_id")
        row["metadata"] = {"explicit_group": g.group_id}
        rows.append(row)
    p = tmp_path / "verl.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [header, *rows]))
    parsed = read_verl(p)
    assert len(parsed.groups) == 1 and len(parsed.completions) == 4
    assert read_verl(p, "hash_only").completions[0].text is None
    del rows[0]["metadata"]["explicit_group"]
    p.write_text("\n".join(json.dumps(r) for r in [header, *rows]))
    with pytest.raises(PTMError, match="explicit verl fields"):
        read_verl(p)


def test_rank_manifest_checksums_and_missing_rank(tmp_path):
    s = demo_snapshot()
    events = tmp_path / "events.jsonl"
    export_events(s, events)
    lines = [json.loads(line) for line in events.read_text().splitlines()]
    shards = []
    for rank in [0, 1]:
        path = tmp_path / f"rank-{rank}.jsonl"
        selected = [v for i, v in enumerate(lines) if i % 2 == rank]
        path.write_text("\n".join(json.dumps(v) for v in selected) + "\n")
        shards.append({"worker_rank": rank, "file": path.name, "sha256": file_hash(path)})
    manifest = tmp_path / "ranks.json"
    manifest.write_text(json.dumps({"world_size": 2, "shards": shards}))
    merged = merge_rank_manifest(manifest)
    assert len(merged.completions) == len(s.completions)
    assert merged.run.config["world_size"] == 2
    manifest.write_text(json.dumps({"world_size": 3, "shards": shards}))
    with pytest.raises(PTMError, match="expected rank"):
        merge_rank_manifest(manifest)
    shards[0]["sha256"] = "bad"
    manifest.write_text(json.dumps({"world_size": 2, "shards": shards}))
    with pytest.raises(PTMError, match="checksum"):
        merge_rank_manifest(manifest)
