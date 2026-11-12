from ptm.demo import demo_snapshot
from ptm.diagnostics import detect


def test_numeric_grades_are_not_silently_boolean_and_sources_must_match():
    snapshot = demo_snapshot()
    assert any(a['id']=='reward-truth-mismatch' for a in detect(snapshot))
    for c in snapshot.completions:
        c.grade.value = .5
    assert not any(a['id']=='reward-truth-mismatch' for a in detect(snapshot))
    snapshot = demo_snapshot()
    for c in snapshot.completions:
        if c.grade.value == 1:
            c.grade.source = 'different-verifier'
    assert not any(a['id']=='reward-truth-mismatch' for a in detect(snapshot))
