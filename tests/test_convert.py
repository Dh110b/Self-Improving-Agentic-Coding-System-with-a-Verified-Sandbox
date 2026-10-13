import json

from sicas.convert import _examples_text, _literal, assign_splits, collect_files, convert_sources, map_type
from sicas.sandbox import LocalSandbox
from sicas.tasks import load_jsonl

PY = '''
from typing import List

def total(nums: List[int]) -> int:
    """Sum.

    >>> total([1, 2])
    3
    """
    return sum(nums)

def is_even(n: int) -> bool:
    return n % 2 == 0

def shout(text):
    return text.upper() + "!"

def clamp(x, lo=0, hi=10):
    return max(lo, min(hi, x))

def mystery(thing):
    return thing

def noisy(n: int):
    print(n)

def reads(n: int):
    return open("x").read()

assert is_even(4) == True
'''
JAVA = '''
public class Calc {
    /** Adds. add(2, 3) => 5 */
    public static int add(int a, int b) { return a + b; }
    public static String greet(String name) { return "hi " + name; }
    public static void main(String[] args) {
        assertEquals(5, add(2, 3));
        assert(add(-1, 1) == 0);
        assertEquals("hi bob", greet("bob"));
    }
}
'''


def convert(files, run=True):
    return convert_sources(files, LocalSandbox() if run else None)


def test_python_function_becomes_verified_task():
    r = convert([("m.py", PY)])
    ids = {t["id"]: t for t in r.tasks}
    assert {"m_total", "m_is_even", "m_shout"} <= set(ids)
    t = ids["m_total"]
    assert "from typing import List" in t["prompt"] and t["prompt"].rstrip().endswith('"""')
    assert "assert total([1, 2]) == 3" in t["visible_tests"]          # doctest example is visible
    assert "recorded" in t["meta"]["tests_from"]
    ns = {}
    exec("def total(nums):\n    return sum(nums)\n", ns)                # a correct solution passes the hidden tests
    exec(t["hidden_tests"], ns)


def test_skips_have_reasons():
    r = convert([("m.py", PY)])
    why = {k["name"]: k["reason"] for k in r.skipped}
    assert "returns nothing" in why["noisy"]
    assert "reads" in why
    assert "types" in why["mystery"]                                   # unknown parameter types
    assert "m_clamp" in {t["id"] for t in r.tasks}                     # types inferred from names/defaults


def test_no_sandbox_uses_only_written_examples():
    r = convert([("m.py", PY)], run=False)
    assert r.tasks == []                                               # at most 1 written example per function
    assert any("not run" in w for w in r.warnings)


def test_other_language_needs_written_examples():
    r = convert([("Calc.java", JAVA)])
    assert [t["id"] for t in r.tasks] == ["Calc_add"]
    t = r.tasks[0]
    assert t["prompt"].startswith("def add(a: int, b: int) -> int:")
    assert "assert add(2, 3) == 5" in t["visible_tests"] and "add(-1, 1)" in t["hidden_tests"]
    assert any(k["name"] == "greet" for k in r.skipped)                # only one example


def test_example_forms_and_literals():
    txt = ('assert(f(1, "a") == true);\nassertEquals(new int[]{1,2}, f(3, "b"));\n'
           'EXPECT_EQ(f(5, "c"), 7);\n// f(0, "d") => -1\nexpect(f(2, "e")).toBe(9);\n')
    got = _examples_text(txt, "f")
    assert (["1", "'a'"], "True") in got and (["0", "'d'"], "-1") in got and (["2", "'e'"], "9") in got
    assert (["3", "'b'"], "[1, 2]") in got and (["5", "'c'"], "7") in got
    assert _literal("{1, 2L, 3}") == [1, 2, 3] and _literal("Arrays.asList(1, 2)") == [1, 2]
    assert map_type("const char *") == "str" and map_type("vector<int>") == "List[int]"
    assert map_type("&str") == "str" and map_type("unsigned long long") == "int" and map_type("Foo") is None


def test_splits_and_load(tmp_path):
    tasks = [{"id": f"t{i}", "split": "train"} for i in range(4)]
    assert assign_splits(tasks) and {t["split"] for t in tasks} == {"train", "val", "test", "reg"}
    assert not assign_splits(tasks[:3])
    (tmp_path / "m.py").write_text(
        PY.replace("def noisy", "def other(n: int) -> int:\n    return n + 1\n\ndef noisy"), encoding="utf-8")
    r = convert(collect_files([str(tmp_path)]))
    out = tmp_path / "t.jsonl"
    out.write_text(r.to_jsonl(), encoding="utf-8")
    loaded = load_jsonl(out)                                           # accepted by the normal task loader
    assert len(loaded) == len(r.tasks) >= 4
    json.loads(r.to_jsonl().splitlines()[0])
