#!/usr/bin/env python3
"""Enforce the published-claims ledger (ai/results/claims.yaml).

Two invariants over the working docs (see the ledger header for scope):
  1. forbidden — superseded values must not appear anywhere in scope;
  2. require   — headline values must be present in their named files.

Run from ai/:  uv run python training/check_claims.py
Exit code 0 = ledger holds; 1 = violations (printed as file:line hits).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

AI_DIR = Path(__file__).resolve().parent.parent
REPO = AI_DIR.parent
LEDGER = AI_DIR / "results" / "claims.yaml"


def scan_files(cfg: dict) -> list[Path]:
    exts = set(cfg["extensions"])
    excludes = cfg.get("exclude", [])
    files: list[Path] = []
    for root in cfg["roots"]:
        p = REPO / root
        if p.is_file():
            files.append(p)
            continue
        if not p.is_dir():
            print(f"  WARN: scan root missing: {root}")
            continue
        for f in sorted(p.rglob("*")):
            if not f.is_file() or f.suffix not in exts:
                continue
            rel = str(f.relative_to(REPO))
            if any(x in rel for x in excludes):
                continue
            files.append(f)
    return files


def main() -> int:
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    files = scan_files(ledger["scan"])
    texts = {}
    for f in files:
        try:
            texts[f] = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

    failures = 0
    for rule in ledger.get("forbidden", []):
        pat = rule["pattern"]
        allow = {str(REPO / a) for a in rule.get("allow", [])}
        for f, text in texts.items():
            if str(f) in allow or pat not in text:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pat in line:
                    failures += 1
                    print(f"FORBIDDEN {f.relative_to(REPO)}:{i}: {pat!r}")
                    print(f"    ({rule['reason']})")

    for rule in ledger.get("require", []):
        f = REPO / rule["file"]
        text = texts.get(f)
        if text is None:
            failures += 1
            print(f"REQUIRE   {rule['file']}: file not in scan scope or unreadable")
            continue
        for pat in rule["patterns"]:
            if pat not in text:
                failures += 1
                print(f"REQUIRE   {rule['file']}: missing {pat!r}")

    n_forbidden = len(ledger.get("forbidden", []))
    n_require = sum(len(r["patterns"]) for r in ledger.get("require", []))
    if failures:
        print(f"\nFAILED: {failures} violation(s) over {len(texts)} files "
              f"({n_forbidden} forbidden patterns, {n_require} required values).")
        return 1
    print(f"OK: ledger holds over {len(texts)} files "
          f"({n_forbidden} forbidden patterns absent, {n_require} required values present).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
