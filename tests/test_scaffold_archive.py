import pytest

from sicas.archive import Archive
from sicas.scaffold import DiffError, Scaffold, apply_diff


def test_apply_diff_is_pure_and_changes_digest():
    sc = Scaffold()
    new = apply_diff(sc, {"ops": [{"op": "add_rule", "text": "Handle empty input."}]})
    assert sc.rules == [] and new.rules == ["Handle empty input."]
    assert new.digest != sc.digest


def test_component_restriction():
    with pytest.raises(DiffError):
        apply_diff(Scaffold(), {"ops": [{"op": "add_rule", "text": "x"}]}, allowed={"skills"})


@pytest.mark.parametrize("op", [
    {"op": "disable_tool", "name": "write_file"}, {"op": "enable_tool", "name": "rm_rf"},
    {"op": "set_retrieval", "k": 99}, {"op": "nope"}, {"op": "add_rule", "text": ""},
    {"op": "remove_rule", "index": 3}])
def test_invalid_ops_rejected(op):
    with pytest.raises(DiffError):
        apply_diff(Scaffold(), {"ops": [op]})


def test_skill_retrieval_topk_by_overlap():
    sc = apply_diff(Scaffold(), {"ops": [
        {"op": "add_skill", "name": "empty", "description": "empty list handling", "content": "A"},
        {"op": "add_skill", "name": "case", "description": "case insensitive strings", "content": "B"}]})
    got = sc.retrieve_skills("Return None for an empty list.")
    assert [s["name"] for s in got] == ["empty"]
    assert "A" in sc.render_system("empty list") and "B" not in sc.render_system("empty list")


def test_archive_lineage_and_rollback(tmp_path):
    a = Archive(tmp_path)
    mk = lambda i, p, acc: {"id": i, "parent": p, "accepted": acc, "scaffold": {}}
    a.add(mk(0, None, True)); a.add(mk(1, 0, True)); a.add(mk(2, 1, False)); a.add(mk(3, 1, True))
    assert a.lineage(3) == [0, 1, 3]
    a.rollback(1)
    assert a.current_id() == 1
    with pytest.raises(ValueError):
        a.rollback(2)
