"""Convert source code into a tasks JSONL that the agent can practise on.

    python -m sicas convert myfile.py other.c Calc.java --out my_tasks.jsonl

Every task is a *Python* task (the agent and the sandbox only run Python):
  * Python files: each top-level function becomes a task. The expected outputs come from, in order,
    doctests in the docstring, `assert f(...) == ...` lines anywhere in the uploaded files, and finally
    inputs generated from the type hints / parameter names whose outputs are recorded by running YOUR
    function (inside the Docker sandbox). The task asks the agent to re-implement the function from
    its signature and docstring, so the scores say how well the agent reproduces your code's behaviour.
  * C, C++, Java, C#, JavaScript, TypeScript, Go, Rust, Kotlin, Swift, PHP: each function becomes a
    "port it to Python" task. Nothing is compiled or run, so the expected outputs must be written in
    the source as examples, e.g. `assert(add(2, 3) == 5);`, `assertEquals(5, add(2, 3));`
    or a comment `// add(2, 3) => 5`.
Functions that cannot be turned into verifiable tasks are skipped, with the reason reported.
"""
from __future__ import annotations

import ast
import copy
import doctest
import hashlib
import json
import keyword
import random
import re
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox import Sandbox

LANGS = {".py": "python", ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
         ".java": "java", ".cs": "csharp", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
         ".go": "go", ".rs": "rust", ".kt": "kotlin", ".swift": "swift", ".php": "php"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "build", "dist", "target", "out",
             ".idea", ".vscode", "site-packages"}
MAX_FILE = 400_000
MAX_CASES = 25
MAX_PROMPT = 8000
NOT_VERIFIABLE_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "requests", "urllib", "pathlib",
                          "random", "time", "datetime", "threading", "asyncio", "http", "ftplib", "smtplib"}
DENY_CALLS = {"input", "open", "exec", "eval", "compile", "__import__", "exit", "quit"}

EQ_HELPER = (
    "import math as _m\n"
    "def _eq(a, b):\n"
    "    if isinstance(a, float) or isinstance(b, float):\n"
    "        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and "
    "_m.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)\n"
    "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):\n"
    "        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))\n"
    "    return a == b\n")


@dataclass
class ConvertResult:
    tasks: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sandbox: str = "none"

    def skip(self, file: str, name: str, reason: str) -> None:
        self.skipped.append({"file": file, "name": name, "reason": reason})

    def to_jsonl(self) -> str:
        return "".join(json.dumps(t) + "\n" for t in self.tasks)

    def summary(self) -> dict:
        return {"n_tasks": len(self.tasks), "n_skipped": len(self.skipped), "sandbox": self.sandbox,
                "tasks": [{"id": t["id"], "file": t["meta"]["file"], "language": t["meta"]["language"],
                           "n_tests": t["meta"]["n_tests"], "tests_from": t["meta"]["tests_from"],
                           "split": t["split"]} for t in self.tasks],
                "skipped": self.skipped, "warnings": self.warnings}


# --------------------------------------------------------------------------- literals
_STR = re.compile(r'("(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\')')


def _outside_strings(s: str, fn) -> str:
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(_STR.split(s)))


def _split_top(s: str, angle: bool = False) -> list[str]:
    out, depth, cur, q, i = [], 0, [], None, 0
    while i < len(s):
        c = s[i]
        if q:
            cur.append(c)
            if c == "\\" and i + 1 < len(s):
                cur.append(s[i + 1]); i += 1
            elif c == q:
                q = None
        elif c in "\"'":
            q = c; cur.append(c)
        elif c in "([{" or (angle and c == "<"):
            depth += 1; cur.append(c)
        elif c in ")]}" or (angle and c == ">"):
            depth -= 1; cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _read_balanced(text: str, i: int):
    """text[i] is an opening bracket; return (inner, index_of_closing) or None."""
    depth, j, q = 0, i, None
    while j < len(text):
        c = text[j]
        if q:
            if c == "\\":
                j += 2; continue
            if c == q:
                q = None
        elif c in "\"'":
            q = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j
        j += 1
    return None


def _safe_eval(n: ast.AST):
    if isinstance(n, ast.Constant) and n.value is not Ellipsis:
        return n.value
    if isinstance(n, (ast.List, ast.Tuple)):
        v = [_safe_eval(e) for e in n.elts]
        return v if isinstance(n, ast.List) else tuple(v)
    if isinstance(n, ast.Set):
        return {_safe_eval(e) for e in n.elts}
    if isinstance(n, ast.Dict) and None not in n.keys:
        return {_safe_eval(k): _safe_eval(v) for k, v in zip(n.keys, n.values)}
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _safe_eval(n.operand)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_L" and not n.keywords:
        return [_safe_eval(a) for a in n.args]
    raise ValueError("not a literal")


_WORDS = {"true": "True", "false": "False", "null": "None", "nil": "None", "NULL": "None",
          "nullptr": "None", "None": "None", "True": "True", "False": "False"}


def _literal(src: str):
    """Translate a literal written in C/Java/JS/Go/Rust/... into a Python value; None-sentinel on failure."""
    def fix(t: str) -> str:
        t = re.sub(r"\bnew\s+[\w.]+\s*(?:<[^<>()]*>)?\s*(?:\[\s*\d*\s*\])+", "", t)
        t = re.sub(r"(?:std::)?(?:vector|array)\s*<[^<>]*>", "", t)
        t = re.sub(r"\[\]\w+(?=\s*\{)", "", t)
        t = re.sub(r"\b(?:Arrays\.asList|List\.of|Set\.of|listOf|arrayOf|intArrayOf|mutableListOf)\s*\(", "_L(", t)
        t = t.replace("vec!", "").replace(".to_string()", "").replace(".to_owned()", "")
        t = re.sub(r"\b(\d+(?:\.\d+)?)(?:[LlUuFfDd]+|[iu](?:8|16|32|64|128|size)|f32|f64)\b", r"\1", t)
        t = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: _WORDS.get(m.group(0), m.group(0)), t)
        return t.replace("{", "[").replace("}", "]")
    expr = _outside_strings(src.strip(), fix)
    try:
        return _safe_eval(ast.parse(expr, mode="eval").body)
    except Exception:
        return _NOPE


_NOPE = object()


def _clean_expected(s: str) -> str:
    s = _outside_strings(s, lambda t: re.split(r"//|#(?!\w)", t)[0]).strip()
    s = s.rstrip(" \t;,{")
    while s.count(")") > s.count("(") and s.endswith(")"):
        s = s[:-1].rstrip(" ;,")
    return s.strip()


# --------------------------------------------------------------------------- types (non-Python)
_INTS = {"int", "long", "short", "byte", "size_t", "ssize_t", "integer", "i8", "i16", "i32", "i64", "i128",
         "isize", "u8", "u16", "u32", "u64", "u128", "usize", "uint", "int8", "int16", "int32", "int64",
         "uint8", "uint16", "uint32", "uint64", "bigint", "uint8_t", "int8_t", "int16_t", "int32_t",
         "int64_t", "uint16_t", "uint32_t", "uint64_t"}
_FLOATS = {"float", "double", "f32", "f64", "float32", "float64", "decimal"}
_STRS = {"string", "str", "string_view", "charsequence", "char"}
_BOOLS = {"bool", "boolean"}
_ARRAYS = {"intarray": "List[int]", "longarray": "List[int]", "doublearray": "List[float]",
           "stringarray": "List[str]", "booleanarray": "List[bool]"}
_CONTAINERS = r"(?:vector|list|ArrayList|LinkedList|Vec|Array|Slice|IEnumerable|Iterable|Collection|Set|HashSet|deque|Sequence|MutableList|IList)"


def map_type(t: str | None) -> str | None:
    """Best-effort Python annotation for a C/Java/Go/Rust/... type; None when unknown."""
    t = (t or "").strip()
    t = re.sub(r"\b(const|final|mut|unsigned|signed|struct|static|volatile|register|inline|var|val|let|"
               r"readonly|out|ref|in)\b", " ", t)
    t = re.sub(r"\s+", " ", t.replace("&", " ").replace("std::", "")).strip()
    if not t:
        return None
    if t.endswith("*"):
        base = t.rstrip("* ").strip()
        if base == "char":
            return "str"
        inner = map_type(base)
        return f"List[{inner}]" if inner else "list"
    if t.endswith("[]"):
        inner = map_type(t[:-2])
        return f"List[{inner}]" if inner else "list"
    m = re.fullmatch(r"\[\](.+)", t)
    if m:
        inner = map_type(m.group(1))
        return f"List[{inner}]" if inner else "list"
    m = re.fullmatch(_CONTAINERS + r"\s*<\s*(.+?)\s*>", t)
    if m:
        inner = map_type(m.group(1))
        return f"List[{inner}]" if inner else "list"
    m = re.fullmatch(r"\[\s*([^\];]+?)\s*\]", t)
    if m:
        inner = map_type(m.group(1))
        return f"List[{inner}]" if inner else "list"
    low = t.lower()
    if low in _ARRAYS:
        return _ARRAYS[low]
    toks = low.split()
    if all(x in _INTS for x in toks):
        return "int"
    if low in _FLOATS:
        return "float"
    if low in _STRS:
        return "str"
    if low in _BOOLS:
        return "bool"
    return None


def _is_void(ret: str | None) -> bool:
    return (ret or "").strip().lower() in ("void", "()", "unit", "nothing", "never", "!")


def _pyname(n: str) -> str:
    return n + "_" if keyword.iskeyword(n) or n in ("None", "True", "False") else n


# --------------------------------------------------------------------------- non-Python signatures
_CKW = {"if", "else", "for", "while", "switch", "catch", "return", "sizeof", "new", "delete", "throw", "case",
        "do", "try", "using", "namespace", "typedef", "struct", "class", "enum", "union", "goto", "elif"}
_MODS = {"public", "private", "protected", "static", "final", "inline", "extern", "virtual", "abstract",
         "synchronized", "override", "const", "constexpr", "unsafe", "async", "internal", "sealed",
         "readonly", "native", "strictfp", "friend", "explicit", "export", "default"}

_C_FN = re.compile(r"^[ \t]*(?P<head>(?:[\w:<>,\[\]*&.]+[ \t*&]+)+)(?P<name>[A-Za-z_]\w*)[ \t]*"
                   r"\((?P<params>[^;{}()]*)\)[^;{}()]*\{", re.M)
_JS_FN = re.compile(r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>\w+)\s*"
                    r"\((?P<params>[^()]*)\)\s*(?::\s*(?P<ret>[^{]+?))?\s*\{", re.M)
_JS_ARROW = re.compile(r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?"
                       r"\((?P<params>[^()]*)\)\s*(?::\s*(?P<ret>[^=]+?))?\s*=>", re.M)
_GO_FN = re.compile(r"^func\s+(?P<name>\w+)\s*\((?P<params>[^()]*)\)\s*(?P<ret>[^\n{]*)\{", re.M)
_RS_FN = re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?fn\s+(?P<name>\w+)\s*(?:<[^>]*>)?\s*"
                    r"\((?P<params>[^()]*)\)\s*(?:->\s*(?P<ret>[^{]+?))?\s*(?:where[^{]*)?\{", re.M)
_KT_FN = re.compile(r"^[ \t]*(?:(?:public|private|internal|inline|tailrec)\s+)*fun\s+(?:<[^>]*>\s*)?(?P<name>\w+)\s*"
                    r"\((?P<params>[^()]*)\)\s*(?::\s*(?P<ret>[^{=]+?))?\s*(?:\{|=)", re.M)
_SW_FN = re.compile(r"^[ \t]*(?:(?:public|private|internal|static|fileprivate|open)\s+)*func\s+(?P<name>\w+)\s*"
                    r"(?:<[^>]*>)?\((?P<params>[^()]*)\)\s*(?:->\s*(?P<ret>[^{]+?))?\s*\{", re.M)
_PHP_FN = re.compile(r"^[ \t]*(?:(?:public|private|protected|static|final)\s+)*function\s+(?P<name>\w+)\s*"
                     r"\((?P<params>[^()]*)\)\s*(?::\s*(?P<ret>\??\w+))?\s*\{", re.M)


def _params_ctype(s: str) -> list[tuple[str, str | None]]:
    out = []
    for p in _split_top(s, angle=True):
        p = re.sub(r"\s*=\s*.*$", "", p).strip()
        if not p or p in ("void", "...") or p.startswith("..."):
            continue
        m = re.match(r"^(?P<type>.*?)(?P<name>[A-Za-z_]\w*)\s*(?P<arr>(?:\[\s*\d*\s*\])*)$", p)
        if not m or not m.group("type").strip():
            return []
        out.append((m.group("name"), (m.group("type") + ("[]" * m.group("arr").count("["))).strip()))
    return out


def _params_colon(s: str) -> list[tuple[str, str | None]]:
    out = []
    for p in _split_top(s, angle=True):
        p = re.sub(r"\s*=\s*.*$", "", p).strip()
        if not p or p in ("self", "&self", "&mut self", "mut self") or p.startswith("..."):
            continue
        m = re.match(r"^(?:[\w]+\s+)?(?:(?:inout|mut|var|val|vararg)\s+)?(?P<name>[A-Za-z_]\w*)\??\s*:\s*(?P<type>.+)$", p)
        if not m:
            return []
        out.append((m.group("name"), m.group("type").strip()))
    return out


def _params_go(s: str) -> list[tuple[str, str | None]]:
    items, out, pending = _split_top(s), [], []
    for it in items:
        parts = it.split(None, 1)
        if len(parts) == 1:
            pending.append(parts[0])
        else:
            for n in pending + [parts[0]]:
                out.append((n, parts[1].strip()))
            pending = []
    return out if not pending else []


def _params_php(s: str) -> list[tuple[str, str | None]]:
    out = []
    for p in _split_top(s):
        p = re.sub(r"\s*=\s*.*$", "", p).strip()
        m = re.match(r"^(?:(?P<type>[\w?\\|]+)\s+)?&?\$(?P<name>\w+)$", p)
        if not m:
            return []
        out.append((m.group("name"), m.group("type")))
    return out


def _params_js(s: str) -> list[tuple[str, str | None]]:
    out = []
    for p in _split_top(s, angle=True):
        p = re.sub(r"\s*=\s*.*$", "", p).strip()
        if p.startswith("...") or not p:
            return [] if p else out
        m = re.match(r"^(?P<name>[A-Za-z_$][\w$]*)\??\s*(?::\s*(?P<type>.+))?$", p)
        if not m:
            return []
        out.append((m.group("name"), m.group("type")))
    return out


def _doc_before(text: str, pos: int, lang: str) -> str:
    lines = text[:pos].rstrip().split("\n")
    while lines and re.match(r"\s*(@\w+.*|#\[.*\]|\[\w+.*\])\s*$", lines[-1]):
        lines.pop()
    if not lines:
        return ""
    if lines[-1].rstrip().endswith("*/"):
        blob = "\n".join(lines)
        s = blob.rfind("/*")
        body = blob[s:] if s >= 0 else ""
        body = body.strip()[2:-2]
        cleaned = [re.sub(r"^\s*\*+ ?", "", ln) for ln in body.split("\n")]
        return "\n".join(cleaned).strip().lstrip("*").strip()
    prefixes = ("//",) + (("#",) if lang == "php" else ())
    got = []
    for ln in reversed(lines):
        st = ln.strip()
        if st.startswith(prefixes):
            got.append(re.sub(r"^(?:///?|//!|#)\s?", "", st))
        else:
            break
    return "\n".join(reversed(got)).strip()


def _find_functions(lang: str, text: str):
    """Yield dicts: name, params [(name, type)], ret, doc, line."""
    pats = {"c": [_C_FN], "cpp": [_C_FN], "java": [_C_FN], "csharp": [_C_FN],
            "javascript": [_JS_FN, _JS_ARROW], "typescript": [_JS_FN, _JS_ARROW], "go": [_GO_FN],
            "rust": [_RS_FN], "kotlin": [_KT_FN], "swift": [_SW_FN], "php": [_PHP_FN]}[lang]
    seen = set()
    for rx in pats:
        for m in rx.finditer(text):
            name, raw = m.group("name"), m.group("params")
            ret = m.groupdict().get("ret")
            if lang in ("c", "cpp", "java", "csharp"):
                head = m.group("head")
                toks = re.findall(r"[\w:<>,\[\]*&.]+", head)
                if name in _CKW or any(t in _CKW for t in toks):
                    continue
                typ = [t for t in toks if t not in _MODS]
                if not typ:
                    continue
                ret, params = " ".join(typ), _params_ctype(raw)
            elif lang == "go":
                params, ret = _params_go(raw), (ret or "").strip()
                if ret.startswith("("):
                    ret = "(multi)"
            elif lang in ("rust", "kotlin", "swift"):
                params = _params_colon(raw) if lang != "swift" else _params_colon(
                    re.sub(r"\b(?:_|\w+)\s+(\w+)\s*:", r"\1:", raw))
                ret = (ret or "").strip() or "()"
            elif lang == "php":
                params = _params_php(raw)
            else:
                params = _params_js(raw)
            if name in seen:
                continue
            seen.add(name)
            yield {"name": name, "params": params, "ret": ret, "line": text.count("\n", 0, m.start()) + 1,
                   "doc": _doc_before(text, m.start(), lang), "raw_params": raw.strip()}


_ASSERT_HINT = re.compile(r"assert|expect|equal|test|check|should", re.I)
_EXPECT_FIRST = re.compile(r"(?:assertEquals?|assertEqual|assert_eq!?|EXPECT_EQ|ASSERT_EQ|AreEqual)\s*\(\s*(.+?)\s*,\s*$")


def _examples_text(text: str, name: str) -> list[tuple[list[str], str]]:
    """(arg literals as Python source, expected literal as Python source) for `name(...)` calls in the text."""
    out = []
    for m in re.finditer(r"(?<![\w$])" + re.escape(name) + r"\s*\(", text):
        rb = _read_balanced(text, m.end() - 1)
        if not rb:
            continue
        inner, end = rb
        ls = text.rfind("\n", 0, m.start()) + 1
        prefix = text[ls:m.start()]
        le = text.find("\n", end)
        rest = text[end + 1:le if le != -1 else len(text)]
        exp = None
        mm = re.match(r"\s*(?:===?|=>|->|→|⇒|returns?\b|gives?\b|=|:)\s*(.+)", rest)
        if mm:
            exp = mm.group(1)
        else:
            mm = re.match(r"\s*\)\s*\.\s*(?:toBe|toEqual|toStrictEqual)\s*\((.+)", rest)
            if mm:
                exp = mm.group(1)
            elif re.match(r"\s*,\s*\S", rest) and _ASSERT_HINT.search(prefix):
                exp = re.match(r"\s*,\s*(.+)", rest).group(1)
            else:
                pm = _EXPECT_FIRST.search(prefix)
                if pm:
                    exp = pm.group(1)
        if exp is None:
            continue
        args = [_literal(a) for a in _split_top(inner)]
        val = _literal(_clean_expected(exp))
        if val is _NOPE or any(a is _NOPE for a in args):
            continue
        out.append(([repr(a) for a in args], repr(val)))
    return out


# --------------------------------------------------------------------------- Python analysis
_POOLS = {
    "int": [0, 1, 2, 3, -1, 5, 7, 10, 12, 100, -20], "float": [0.0, 1.5, -2.25, 3.0, 10.5],
    "str": ["", "a", "abc", "Hello World", "racecar", "a b  c", "AbC123", "  padded  "],
    "bool": [True, False],
    "list_int": [[], [1], [3, 1, 2], [5, 5, 5], [-1, 0, 1, 2], [10, 20, 30, 40, 50], [4, -3, 9, 0, 2]],
    "list_float": [[], [1.5], [2.5, 0.5, 1.0]],
    "list_str": [[], ["a"], ["b", "a", "c"], ["hello", "world"], ["x", "x", "y"]],
}
_NAME_HINTS = [
    (r"^(n|m|k|i|j|x|y|a|b|c|num|number|count|size|length|limit|index|idx|start|end|value|val|total|"
     r"width|height|depth|base|exp|power|year|age)$", "int"),
    (r"(nums|numbers|arr|array|lst|values|xs|ints|items|data|seq|sequence|numlist)$", "list_int"),
    (r"^(s|s1|s2|st|text|word|string|name|sentence|line|msg|message|phrase|sub|prefix|suffix|pattern)$", "str"),
    (r"^(words|names|strings|lines|strs|tokens)$", "list_str"),
    (r"^(flag|is_\w+|has_\w+|reverse|ascending|verbose)$", "bool"),
]


def _kind(ann: str | None, default: ast.AST | None, pname: str) -> str | None:
    if ann:
        a = ann.replace("typing.", "")
        m = re.fullmatch(r"(?:List|list|Sequence|Iterable|Tuple|tuple)\[(\w+)(?:, \.\.\.)?\]", a)
        k = ("list_" + m.group(1)) if m else a
        return k if k in _POOLS else None
    if isinstance(default, ast.Constant):
        for t, k in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
            if isinstance(default.value, t):
                return k
    for rx, k in _NAME_HINTS:
        if re.search(rx, pname):
            return k
    return None


def _gen_cases(fn: ast.FunctionDef) -> list[str] | None:
    """Call-argument sources for generated inputs, or None if some parameter type is unknown."""
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs:
        return None
    params = a.posonlyargs + a.args
    defaults = [None] * (len(params) - len(a.defaults)) + list(a.defaults)
    pools = []
    for p, d in zip(params, defaults):
        k = _kind(ast.unparse(p.annotation) if p.annotation else None, d, p.arg)
        if k is None:
            return None
        pools.append(_POOLS[k])
    if not pools:
        return None
    rng = random.Random(int(hashlib.sha1(fn.name.encode()).hexdigest()[:8], 16))
    combos = [tuple(p[i % len(p)] for p in pools) for i in range(max(len(p) for p in pools))]
    combos += [tuple(rng.choice(p) for p in pools) for _ in range(30)]
    seen, out = set(), []
    for c in combos:
        src = ", ".join(repr(x) for x in c)
        if src not in seen:
            seen.add(src); out.append(src)
    return out[:16]


_DRIVER = r'''
import ast, json, math, sys, importlib.util
cfg = json.load(open("cases.json"))
spec = importlib.util.spec_from_file_location("ref", "ref.py")
m = importlib.util.module_from_spec(spec); sys.modules["ref"] = m
try:
    spec.loader.exec_module(m)
    fn = getattr(m, cfg["name"])
except BaseException as e:
    print("__SICAS__" + json.dumps({"error": ("import failed: %s: %s" % (type(e).__name__, e))[:200]})); sys.exit(0)
out = []
for src in cfg["cases"]:
    try:
        r1 = eval("fn(" + src + ")", {"fn": fn}); r2 = eval("fn(" + src + ")", {"fn": fn})
        t = repr(r1)
        ok = (r1 == r2 and r1 is not None and len(t) <= 300 and ast.literal_eval(t) == r1
              and not (isinstance(r1, float) and (math.isnan(r1) or math.isinf(r1))))
    except BaseException:
        ok = False
    out.append(t if ok else None)
print("__SICAS__" + json.dumps({"results": out}))
'''


def _run_reference(sandbox: Sandbox, module_src: str, name: str, arg_srcs: list[str], timeout: int = 20):
    """Return ([expected_src or None per case], error_or_None). Runs the user's code inside the sandbox."""
    with tempfile.TemporaryDirectory(prefix="sicas_conv_") as d:
        d = Path(d)
        (d / "ref.py").write_text(module_src, encoding="utf-8")
        (d / "cases.json").write_text(json.dumps({"name": name, "cases": arg_srcs}), encoding="utf-8")
        (d / "driver.py").write_text(_DRIVER, encoding="utf-8")
        r = sandbox.run_python(["driver.py"], d, timeout, readonly=True)
    if r.timed_out:
        return None, "the function did not finish in time (infinite loop or too slow)"
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("__SICAS__")), None)
    if not line:
        return None, "could not run it: " + (r.stderr.strip().splitlines() or ["no output"])[-1][:160]
    data = json.loads(line[len("__SICAS__"):])
    return data.get("results"), data.get("error")


def _lit_call_args(call: ast.Call) -> str | None:
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None
    try:
        for a in call.args:
            ast.literal_eval(a)
        for k in call.keywords:
            ast.literal_eval(k.value)
    except Exception:
        return None
    return ", ".join([ast.unparse(a) for a in call.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in call.keywords])


def _is_call_to(n: ast.AST, name: str) -> bool:
    return isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name


def _lit(n: ast.AST):
    try:
        return repr(ast.literal_eval(n))
    except Exception:
        return None


def _py_given_cases(trees: list[ast.AST], fn: ast.FunctionDef) -> list[tuple[str, str]]:
    name, cases = fn.name, []

    def add(call, exp):
        a = _lit_call_args(call)
        if a is not None and exp is not None:
            cases.append((a, exp))

    doc = ast.get_docstring(fn) or ""
    for ex in doctest.DocTestParser().get_examples(doc):
        try:
            node = ast.parse(ex.source.strip(), mode="eval").body
            if _is_call_to(node, name) and ex.want.strip() and not ex.exc_msg:
                add(node, repr(ast.literal_eval(ex.want.strip())))
        except Exception:
            pass
    for tree in trees:
        for n in ast.walk(tree):
            if isinstance(n, ast.Assert):
                t = n.test
                if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], (ast.Eq, ast.Is)):
                    l, r = t.left, t.comparators[0]
                    if _is_call_to(l, name):
                        add(l, _lit(r))
                    elif _is_call_to(r, name):
                        add(r, _lit(l))
                elif _is_call_to(t, name):
                    add(t, "True")
                elif isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not) and _is_call_to(t.operand, name):
                    add(t.operand, "False")
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and len(n.args) == 2:
                if n.func.attr in ("assertEqual", "assertEquals"):
                    a, b = n.args
                    if _is_call_to(a, name):
                        add(a, _lit(b))
                    elif _is_call_to(b, name):
                        add(b, _lit(a))
    return cases


def _py_task(file: str, src: str, tree: ast.Module, fn: ast.FunctionDef, trees, sandbox, res):
    name = fn.name
    mod_defs, mod_imports = {}, {}
    for s in tree.body:
        if isinstance(s, (ast.FunctionDef, ast.ClassDef)):
            mod_defs[s.name] = s
        elif isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
            mod_defs[s.targets[0].id] = s
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for al in s.names:
                mod_imports[(al.asname or al.name).split(".")[0]] = s
    # dependencies on other code in the same file are copied into the prompt (like HumanEval helpers)
    deps, todo, used = {}, [fn], set()
    while todo:
        node = todo.pop()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                used.add(n.id)
                if n.id in mod_defs and n.id != name and n.id not in deps:
                    deps[n.id] = mod_defs[n.id]; todo.append(mod_defs[n.id])
    for m in used & set(mod_imports):
        if m in NOT_VERIFIABLE_MODULES:
            res.skip(file, name, f"uses `{m}`, which cannot be verified deterministically in the sandbox")
            return None
    for n in ast.walk(fn):
        if isinstance(n, (ast.Yield, ast.YieldFrom)):
            res.skip(file, name, "is a generator"); return None
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in DENY_CALLS:
            res.skip(file, name, f"calls {n.func.id}() (needs input, files or dynamic code)"); return None
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
            if any(x.split(".")[0] in NOT_VERIFIABLE_MODULES for x in mods):
                res.skip(file, name, "imports a module that cannot be verified deterministically"); return None
    if not any(isinstance(n, ast.Return) and n.value is not None and
               not (isinstance(n.value, ast.Constant) and n.value.value is None) for n in ast.walk(fn)):
        res.skip(file, name, "returns nothing, so there is no output to check"); return None
    if fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        res.skip(file, name, "uses *args/**kwargs/keyword-only parameters"); return None

    doc = ast.get_docstring(fn, clean=True)
    if not doc:
        lines = src.splitlines()
        above, i = [], fn.lineno - 2
        while i >= 0 and lines[i].strip().startswith("#"):
            above.append(lines[i].strip().lstrip("#").strip()); i -= 1
        doc = "\n".join(reversed(above)) or f"Implement `{name}` ({name.replace('_', ' ')})."
    hdr = copy.deepcopy(fn)
    hdr.decorator_list, hdr.body = [], [ast.Expr(ast.Constant(doc))]
    ast.fix_missing_locations(hdr)
    dep_nodes = sorted(deps.values(), key=lambda s: s.lineno)
    names_needed = set(used)
    for d in dep_nodes:
        names_needed |= {n.id for n in ast.walk(d) if isinstance(n, ast.Name)}
    seen_imp = []
    for m in sorted(names_needed & set(mod_imports), key=lambda k: mod_imports[k].lineno):
        if mod_imports[m] not in seen_imp:
            seen_imp.append(mod_imports[m])
    prompt = "\n".join(ast.unparse(x) for x in seen_imp) + ("\n\n" if seen_imp else "")
    prompt += "".join(ast.unparse(d) + "\n\n" for d in dep_nodes) + ast.unparse(hdr) + "\n"
    if len(prompt) > MAX_PROMPT:
        res.skip(file, name, "needs too much surrounding code to include in the prompt"); return None

    given = _py_given_cases(trees, fn)
    cases = [(a, e, "given") for a, e in given]
    if sandbox is not None:
        gen = _gen_cases(fn)
        if gen is None:
            if len(set(a for a, _ in given)) < 2:
                res.skip(file, name, "parameter types are unknown: add type hints (e.g. `n: int`) or "
                         "`assert f(...) == ...` examples")
                return None
        else:
            have = {a for a, _ in given}
            todo_cases = [g for g in gen if g not in have]
            outs, err = _run_reference(sandbox, src, name, todo_cases)
            if outs is None or err:
                res.warnings.append(f"{file}:{name}: could not record outputs ({err}).")
            else:
                cases += [(a, e, "recorded") for a, e in zip(todo_cases, outs) if e is not None]
    return prompt, cases


# --------------------------------------------------------------------------- assembling
def _assemble(name: str, cases: list[tuple[str, str, str]]):
    seen, uniq = set(), []
    for a, e, k in cases:
        if a not in seen:
            seen.add(a); uniq.append((a, e, k))
    uniq = uniq[:MAX_CASES]
    if len(uniq) < 2:
        return None
    nvis = 1 if len(uniq) < 4 else 2 if len(uniq) < 8 else 3
    vis = [c for c in uniq if "." not in c[1]][:nvis] or uniq[:1]
    visible = "\n".join(f"assert {name}({a}) == {e}" for a, e, _ in vis)
    hidden = EQ_HELPER + "\n".join(f"assert _eq({name}({a}), {e})" for a, e, _ in uniq) + "\n"
    kinds = sorted({k for _, _, k in uniq})
    return visible, hidden, len(uniq), "+".join(kinds)


def assign_splits(tasks: list[dict]) -> bool:
    """Deterministic train/val/test/reg split; every split non-empty when there are >= 4 tasks."""
    n = len(tasks)
    if n < 4:
        return False
    order = sorted(tasks, key=lambda t: hashlib.sha1(t["id"].encode()).hexdigest())
    labels = ["reg"] * max(1, round(.10 * n)) + ["val"] * max(1, round(.15 * n)) + ["test"] * max(1, round(.25 * n))
    labels += ["train"] * (n - len(labels))
    for t, lab in zip(order, labels):
        t["split"] = lab
    return True


def _is_test_file(path: str) -> bool:
    b = Path(path).name.lower()
    return bool(re.match(r"(test_.*|.*_test\..*|.*\.test\..*|.*\.spec\..*|.*tests?\.\w+|conftest\.py)$", b))


def collect_files(paths: list[str]) -> list[tuple[str, str]]:
    """Read files / folders from disk into (name, text) pairs."""
    out = []
    for p in map(Path, paths):
        cands = [p] if p.is_file() else sorted(
            f for f in p.rglob("*") if f.is_file() and not (set(f.relative_to(p).parts[:-1]) & SKIP_DIRS))
        for f in cands:
            if f.suffix.lower() in LANGS:
                out.append((str(f), f.read_text(encoding="utf-8", errors="replace")))
    return out


def convert_sources(files: list[tuple[str, str]], sandbox: Sandbox | None = None) -> ConvertResult:
    """files: (path_or_name, text). sandbox: where to run Python code to record expected outputs
    (None = never run anything; only examples written in the source are used)."""
    res = ConvertResult(sandbox=sandbox.name if sandbox else "none")
    parsed = []
    for name, text in files:
        lang = LANGS.get(Path(name).suffix.lower())
        if not lang:
            res.skip(name, "", "unsupported file type"); continue
        if len(text) > MAX_FILE:
            res.skip(name, "", f"file is larger than {MAX_FILE // 1000} KB"); continue
        parsed.append((name, lang, text.replace("\r\n", "\n")))
    py_trees = {}
    for name, lang, text in parsed:
        if lang == "python":
            try:
                py_trees[name] = ast.parse(text)
            except SyntaxError as e:
                res.skip(name, "", f"Python syntax error on line {e.lineno}")
    if sandbox is None and any(py_trees):
        res.warnings.append("Python code was not run (no Docker, and running locally was not allowed), so only "
                            "examples written in the code (doctests / assert lines) were used.")

    ids = set()

    def new_id(file: str, fname: str) -> str:
        base = re.sub(r"\W+", "_", f"{Path(file).stem}_{fname}").strip("_")
        i, cand = 1, base
        while cand in ids:
            i += 1; cand = f"{base}_{i}"
        ids.add(cand)
        return cand

    all_py = list(py_trees.values())
    for name, lang, text in parsed:
        if lang == "python":
            tree = py_trees.get(name)
            if tree is None or _is_test_file(name):
                continue
            for fn in [s for s in tree.body if isinstance(s, ast.FunctionDef)]:
                if fn.name.startswith(("_", "test")) or fn.name == "main":
                    continue
                r = _py_task(name, text, tree, fn, all_py, sandbox, res)
                if r is None:
                    continue
                prompt, cases = r
                asm = _assemble(fn.name, cases)
                if asm is None:
                    res.skip(name, fn.name, "fewer than 2 checkable examples (add doctests / `assert f(...) == ...` "
                             "lines, or type hints so inputs can be generated)")
                    continue
                vis, hid, n, how = asm
                res.tasks.append({"id": new_id(name, fn.name), "prompt": prompt, "entry_point": fn.name,
                                  "visible_tests": vis, "hidden_tests": hid, "split": "train",
                                  "meta": {"file": name, "language": "python", "n_tests": n, "tests_from": how}})
        else:
            if _is_test_file(name):
                continue
            same_lang = [t for _, l, t in parsed if l == lang]
            for f in _find_functions(lang, text):
                fname = f["name"]
                if fname.startswith("_") or fname == "main" or keyword.iskeyword(fname):
                    continue
                if _is_void(f["ret"]) or (lang in ("go", "rust", "kotlin", "swift") and f["ret"] in ("", "()")):
                    res.skip(name, fname, "returns nothing, so there is no output to check"); continue
                if f["ret"] == "(multi)":
                    res.skip(name, fname, "returns several values"); continue
                if not f["params"] and f["raw_params"] not in ("", "void"):
                    res.skip(name, fname, "could not read its parameter list"); continue
                if not f["params"]:
                    res.skip(name, fname, "takes no parameters"); continue
                ex = [e for t in same_lang for e in _examples_text(t, fname) if len(e[0]) == len(f["params"])]
                cases = [(", ".join(a), e, "given") for a, e in ex]
                if len({c[0] for c in cases}) < 2:
                    res.skip(name, fname, "needs at least 2 examples written in the source, e.g. "
                             f"`assert({fname}(...) == ...)` or a comment `{fname}(...) => ...`")
                    continue
                pnames = [_pyname(n) for n, _ in f["params"]]
                sig = ", ".join(p + (f": {a}" if (a := map_type(t)) else "") for p, (_, t) in zip(pnames, f["params"]))
                ra = map_type(f["ret"]) if f["ret"] else None
                doc = f["doc"] or f"Implement `{fname}` ({re.sub(r'([a-z])([A-Z])', r'\1 \2', fname).replace('_', ' ').lower()})."
                doc = doc.replace("\\", "\\\\").replace('"""', "'''").strip()
                doc = textwrap.indent(doc, "    ").lstrip()
                body = f'    """{doc}\n\n    (Ported from {lang} file `{Path(name).name}`: write this function in Python.)\n    """\n'
                prompt = ("from typing import List\n\n" if "List[" in sig + (ra or "") else "") + \
                    f"def {fname}({sig})" + (f" -> {ra}" if ra else "") + ":\n" + body
                vis, hid, n, how = _assemble(fname, cases)
                res.tasks.append({"id": new_id(name, fname), "prompt": prompt, "entry_point": fname,
                                  "visible_tests": vis, "hidden_tests": hid, "split": "train",
                                  "meta": {"file": name, "language": lang, "n_tests": n, "tests_from": how}})

    if len(res.tasks) > 200:
        res.warnings.append("more than 200 tasks found; only the first 200 are kept")
        res.tasks = res.tasks[:200]
    if not assign_splits(res.tasks):
        res.warnings.append(f"{len(res.tasks)} task(s) found, but at least 4 are needed so that practice, check, "
                            "test and regression sets are all non-empty. Add more files or functions.")
    return res
