#!/usr/bin/env python3
"""Generate ``prototypes.py`` (ctypes restype/argtypes for every C function)
by parsing the vendored ``PicoGK.h`` header.

The header grammar is highly regular::

    PICOGK_API <return-type> <Name>( <arg>, <arg>, ... );

Run from anywhere::

    python -m picopie._native._gen_prototypes        # writes prototypes.py
    python -m picopie._native._gen_prototypes --check # parse + report only

Regenerate this whenever the bound runtime version changes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = HERE / "headers" / "PicoGK.h"
OUT = HERE / "prototypes.py"

# --- macro expansions from the header (#define lines we rely on) -------------
# All handle aliases collapse to uint64_t; PKVIEWER is void*.
MACROS = {
    "PKHANDLE": "uint64_t",
    "PKINSTANCE": "uint64_t", "PKMESH": "uint64_t", "PKLATTICE": "uint64_t",
    "PKPOLYLINE": "uint64_t", "PKVOXELS": "uint64_t", "PKVDBFILE": "uint64_t",
    "PKSCALARFIELD": "uint64_t", "PKVECTORFIELD": "uint64_t",
    "PKMETADATA": "uint64_t", "PKFILEINFO": "uint64_t",
    "PKGPUTEX": "uint64_t", "PKQUAD": "uint64_t", "PKGUI": "uint64_t",
    "PKVIEWER": "void*",
    "PKINFOSTRINGLEN": "255",
}

STRUCTS = {"PKVector2", "PKVector3", "PKVector4", "PKCoord",
           "PKTriangle", "PKBBox3", "PKMatrix4x4", "PKColorFloat"}
CALLBACKS = {"PKPFnfSdf", "PKFnTraverseActiveS", "PKFnTraverseActiveV", "PKFInfo",
             "PKPFUpdateRequested", "PKPFKeyPressed", "PKPFMouseMoved",
             "PKPFMouseButton", "PKPFScrollWheel", "PKPFWindowSize"}
# All callbacks are modeled (Viewer callbacks added in Phase 10).
CALLBACKS_MODELED = set(CALLBACKS)

SCALARS = {
    "void": "None", "bool": "c_bool", "int": "c_int",
    "int32_t": "c_int32", "int64_t": "c_int64", "uint64_t": "c_uint64",
    "float": "c_float", "double": "c_double",
}


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    # drop preprocessor lines (#define / #ifdef / #include ...)
    text = re.sub(r"(?m)^[ \t]*#.*$", "", text)
    return text


def expand_macros(type_str: str) -> str:
    """Whole-word macro substitution so e.g. PKVIEWER -> void* before we
    count pointers."""
    for key, val in MACROS.items():
        type_str = re.sub(rf"\b{key}\b", val, type_str)
    return type_str


def map_type(ctype: str, *, is_return: bool) -> str:
    """Map a single (macro-expanded) C type string to a ctypes expression."""
    t = ctype.strip()
    t = t.replace("const", " ")
    t = expand_macros(t)                     # PKVIEWER -> void*, handles -> uint64_t
    is_array = "[" in t
    t = re.sub(r"\[[^\]]*\]", "", t)        # drop array brackets
    nptr = t.count("*") + (1 if is_array else 0)
    base = t.replace("*", "").strip()
    base = " ".join(base.split())

    # char* / char[] -> c_char_p (input or output string buffer)
    if base == "char":
        if nptr >= 1:
            return "c_char_p"
        raise ValueError(f"bare char type: {ctype!r}")

    if base == "void" and nptr >= 1:
        return "c_void_p"

    if base in CALLBACKS:
        return base if base in CALLBACKS_MODELED else "c_void_p"

    if base in STRUCTS:
        return f"POINTER({base})" if nptr else base

    if base in SCALARS:
        c = SCALARS[base]
        if nptr == 0:
            if c == "None" and not is_return:
                raise ValueError("void as non-return arg")
            return c
        for _ in range(nptr):
            c = f"POINTER({c})"
        return c

    raise ValueError(f"unmapped type: base={base!r} nptr={nptr} from {ctype!r}")


def split_args(argstr: str) -> list[str]:
    argstr = argstr.strip()
    if not argstr or argstr == "void":
        return []
    return [a.strip() for a in argstr.split(",") if a.strip()]


def arg_type_only(arg: str) -> str:
    """Return the type portion of a parameter declaration (drop the name)."""
    a = " ".join(arg.split())
    a = re.sub(r"\[[^\]]*\]", "[]", a)  # normalise array markers, keep one
    has_array = "[]" in a
    a = a.replace("[]", "")
    tokens = a.split()
    # one token -> unnamed param (whole thing is the type); else drop the name
    type_str = a if len(tokens) <= 1 else " ".join(tokens[:-1])
    if has_array:
        type_str += "[]"
    return type_str


DECL_RE = re.compile(
    r"PICOGK_API\s+(?P<sig>[A-Za-z_][\w\s\*]*?\([^;]*?\))\s*;", re.S)
HEAD_RE = re.compile(
    r"^(?P<ret>[A-Za-z_][\w\s\*]*?)\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>.*)\)$",
    re.S)


def parse(header_text: str):
    text = strip_comments(header_text)
    funcs = []
    problems = []
    for m in DECL_RE.finditer(text):
        sig = " ".join(m.group("sig").split())
        hm = HEAD_RE.match(sig)
        if not hm:
            problems.append(("parse-head", sig))
            continue
        name = hm.group("name")
        try:
            restype = map_type(hm.group("ret"), is_return=True)
            argtypes = [map_type(arg_type_only(a), is_return=False)
                        for a in split_args(hm.group("args"))]
        except ValueError as e:
            problems.append((str(e), name))
            continue
        funcs.append((name, restype, argtypes))
    return funcs, problems


HEADER_TMPL = '''\
"""AUTO-GENERATED by _gen_prototypes.py from PicoGK.h -- do not edit by hand.

Binds {n} native functions. Call ``apply(lib)`` on a loaded CDLL to set
``restype``/``argtypes`` on every exported symbol.
"""

from ctypes import (  # noqa: F401
    CDLL, POINTER, c_bool, c_char_p, c_int, c_int32, c_int64, c_uint64,
    c_float, c_double, c_void_p,
)
from typing import Any

from .ctypes_types import (  # noqa: F401
    PKVector2, PKVector3, PKVector4, PKCoord, PKTriangle, PKBBox3,
    PKMatrix4x4, PKColorFloat,
    PKPFnfSdf, PKFnTraverseActiveS, PKFnTraverseActiveV, PKFInfo,
    PKPFUpdateRequested, PKPFKeyPressed, PKPFMouseMoved, PKPFMouseButton,
    PKPFScrollWheel, PKPFWindowSize,
)

# name -> (restype, [argtypes])  as ctypes objects
SIGNATURES: dict[str, tuple[Any, list[Any]]] = {{
{rows}
}}

FUNCTIONS = tuple(SIGNATURES)


def apply(lib: CDLL) -> list[str]:
    """Set restype/argtypes on every function. Returns names not found in lib."""
    missing: list[str] = []
    for name, (restype, argtypes) in SIGNATURES.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        fn.restype = restype
        fn.argtypes = list(argtypes)
    return missing
'''


def render(funcs) -> str:
    rows = []
    for name, restype, argtypes in funcs:
        args = ", ".join(argtypes)
        rows.append(f'    "{name}": ({restype}, [{args}]),')
    return HEADER_TMPL.format(n=len(funcs), rows="\n".join(rows))


def main(argv: list[str]) -> int:
    text = HEADER.read_text(encoding="utf-8", errors="replace")
    funcs, problems = parse(text)
    print(f"parsed {len(funcs)} functions from {HEADER.name}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for reason, what in problems:
            print(f"  [{reason}] {what}")
    # category summary
    from collections import Counter
    cats = Counter(n.split("_", 1)[0] for n, _, _ in funcs)
    print("by category:", dict(sorted(cats.items(), key=lambda kv: -kv[1])))

    if "--check" in argv:
        return 1 if problems else 0
    OUT.write_text(render(funcs), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
