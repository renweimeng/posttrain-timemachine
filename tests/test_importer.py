import json

import pytest

from ptm.demo import demo_snapshot
from ptm.errors import PTMError
from ptm.importer import export_events, read_snapshot
from ptm.replay import load_spec


def test_event_deduplication_and_redaction(tmp_path):
    path = tmp_path / "events.jsonl"
    snapshot = demo_snapshot()
    export_events(snapshot, path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n" + lines[0] + "\n")
    assert read_snapshot(path).ordered() == snapshot.ordered()
    hidden = read_snapshot(path, "hash_only")
    assert hidden.completions[0].text is None
    assert hidden.completions[0].text_hash
    assert hidden.groups[0].prompt is None


def test_conflicting_event_and_version(tmp_path):
    path = tmp_path / "events.jsonl"
    export_events(demo_snapshot(), path)
    first = json.loads(path.read_text().splitlines()[0])
    first["data"]["name"] = "conflicting"
    with path.open("a") as stream:
        stream.write(json.dumps(first) + "\n")
    with pytest.raises(PTMError, match="Conflicting"):
        read_snapshot(path)
    path = tmp_path / "snapshot.json"
    data = demo_snapshot().ordered()
    data["run"]["schema_version"] = "0.0.1"
    path.write_text(json.dumps(data))
    assert read_snapshot(path).run.schema_version == "0.1.0"


def test_yaml_alias_graph_is_rejected_before_expansive_traversal(tmp_path):
    path = tmp_path/'aliases.yaml'
    path.write_text('a: &a [1, 2]\nb: &b [*a, *a, *a]\nc: [*b, *b, *b]\n')
    with pytest.raises(PTMError, match='Shared/cyclic'):
        load_spec(path)
