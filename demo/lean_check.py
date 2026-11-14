"""Lean proof checking for the demo's Proofs page (site/proofs.html).

The fingerprint says *who* wrote a proof; Lean says whether it's *right*. This
module adds two routes to the demo server:

    GET  /api/lean/info   {"lean": bool, "mathlib": bool}
    POST /api/lean        {"code": "..."} -> {"ok", "sorry", "blocked", "seconds", "messages"}

Setup (once):
    curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
    cd demo && lake +leanprover-community/mathlib4:lean-toolchain new proofs math
    cd proofs && lake exe cache get          # prebuilt Mathlib, a few GB

Plain proofs need only Lean; proofs that `import Mathlib` run inside the
demo/proofs Lake project. Lean code can run programs on this machine (#eval,
run_cmd, custom elaborators...), so anything that can reach IO or define new
syntax is refused before Lean runs. That is a guard for a local demo, not a
sandbox: a public deployment should also run the checker in an isolated container.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

HERE = Path(__file__).parent
LEAN_PROJECT = HERE / "proofs"  # Lake project with Mathlib as a dependency
TIMEOUT = 20
TIMEOUT_MATHLIB = 90  # loading Mathlib alone takes several seconds

BLOCKED = re.compile(
    r"#eval|#exit|\brun_cmd\b|\brun_elab\b|\brun_meta\b|\brun_tac\b|"
    r"\belab\b|\bmacro\b|\bmacro_rules\b|\bsyntax\b|\binitialize\b|\bbuiltin_initialize\b|"
    r"\bunsafe\b|\bextern\b|implemented_by|\bIO\b|\bopen\s+Lean\b|\bLean\.Elab\b|"
    r"\bLean\.Meta\b|\bSystem\b|\bnative_decide\b"
)
IMPORT_LINE = re.compile(r"^\s*import\b(.*)$", re.M)
MATHLIB_MODULE = re.compile(r"^Mathlib(\.[A-Za-z0-9_]+)*$")


class LeanReq(BaseModel):
    code: str = Field(min_length=1, max_length=20_000)


def lean_bin() -> str | None:
    for cand in (os.environ.get("LEAN_BIN"), shutil.which("lean"), str(Path.home() / ".elan/bin/lean")):
        if cand and Path(cand).exists():
            return cand
    return None


def mathlib_ready() -> bool:
    return (LEAN_PROJECT / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean").exists()


def _message(severity: str, text: str, line: int = 0, column: int = 0) -> dict:
    return {"severity": severity, "line": line, "column": column, "text": text}


def check(code: str) -> dict:
    lean = lean_bin()
    if lean is None:
        raise HTTPException(503, "Lean isn't installed on this server.")

    def blocked(pos: int, text: str) -> dict:
        return {"ok": False, "blocked": True, "sorry": False, "seconds": 0,
                "messages": [_message("error", text, code.count("\n", 0, pos) + 1)]}

    bad = BLOCKED.search(code)
    if bad:
        return blocked(bad.start(), f"`{bad.group(0).strip()}` isn't allowed here: the demo only checks plain "
                                    "proofs, not code that runs programs or defines new syntax.")
    imports = list(IMPORT_LINE.finditer(code))
    for m in imports:
        if not all(MATHLIB_MODULE.match(mod) for mod in m.group(1).split()):
            return blocked(m.start(), f"`import{m.group(1)}` isn't allowed here: only Mathlib can be imported.")
    uses_mathlib = bool(imports)
    if uses_mathlib and not mathlib_ready():
        return blocked(imports[0].start(), "This proof imports Mathlib, but Mathlib isn't installed on this server yet.")
    timeout = TIMEOUT_MATHLIB if uses_mathlib else TIMEOUT

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "Proof.lean"
        path.write_text(code)
        if uses_mathlib:  # run inside the Mathlib project so `import Mathlib` resolves
            cmd, cwd = [str(Path(lean).with_name("lake")), "env", "lean", "--json", str(path)], LEAN_PROJECT
        else:
            cmd, cwd = [lean, "--json", str(path)], Path(d)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return {"ok": False, "blocked": False, "sorry": False, "seconds": timeout,
                    "messages": [_message("error", f"Lean took longer than {timeout}s, so the check was stopped.")]}
        secs = round(time.time() - t0, 2)

    messages = []
    for line in p.stdout.splitlines():
        try:
            m = json.loads(line)
            pos = m.get("pos") or {}
            messages.append(_message(m.get("severity", "information"), str(m.get("data", "")).strip(),
                                     pos.get("line", 0), pos.get("column", 0)))
        except (json.JSONDecodeError, AttributeError):
            if line.strip():
                messages.append(_message("information", line.strip()))
    if p.stderr.strip():
        messages.append(_message("error", p.stderr.strip()[-2000:]))
    has_error = any(m["severity"] == "error" for m in messages)
    # `sorry` makes Lean accept an unfinished proof with only a warning; that isn't a proof.
    has_sorry = any("sorry" in m["text"] for m in messages if m["severity"] == "warning")
    return {"ok": p.returncode == 0 and not has_error and not has_sorry, "blocked": False,
            "sorry": has_sorry, "seconds": secs, "messages": messages}


def add_lean_routes(app: FastAPI) -> None:
    @app.get("/api/lean/info")
    def lean_info():
        return {"lean": lean_bin() is not None, "mathlib": mathlib_ready()}

    @app.post("/api/lean")
    def lean_check(req: LeanReq):
        return check(req.code)
