"""Freeze the benchmark data and check that a cases file is the frozen one.

data.lock.json holds a SHA-256 for every suite in cases.jsonl and image_cases.jsonl: the hash of each case (state,
questions and gold, as canonical JSON) and of every image file it points to, in order. A model's results only compare
with the board when it ran on cases with these hashes, so run_decisions.py and score.py refuse cases that don't match.

Usage: python data_lock.py                                      check every file in the lock, exit 1 on any mismatch
       python data_lock.py --write cases.jsonl image_cases.jsonl  freeze these files (only for a new benchmark version)
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.lock.json")


def images(x):
    """Every image path a case points to, in order."""
    if isinstance(x, dict) and set(x) == {"image"}:
        yield x["image"]
    elif isinstance(x, dict):
        for v in x.values():
            yield from images(v)
    elif isinstance(x, list):
        for v in x:
            yield from images(v)


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint(path):
    """{suite: {"n", "sha256"}} for a cases file, plus one hash over all suites."""
    h, n = defaultdict(hashlib.sha256), defaultdict(int)
    for line in open(path, encoding="utf-8"):
        c = json.loads(line)
        h[c["suite"]].update(json.dumps(c, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
        for p in images(c):
            h[c["suite"]].update(f"{p} {file_sha(p) if os.path.exists(p) else 'missing'}\n".encode())
        n[c["suite"]] += 1
    suites = {s: {"n": n[s], "sha256": h[s].hexdigest()} for s in sorted(h)}
    return {"sha256": total(suites), "suites": suites}


def total(suites):
    return hashlib.sha256("".join(f"{s} {v['sha256']}\n" for s, v in sorted(suites.items())).encode()).hexdigest()


def check(path, suites=None):
    """Problems with a cases file against the lock, limited to `suites` when given; empty when it matches."""
    lock = json.load(open(LOCK)).get(os.path.basename(path))
    if lock is None:
        return [f"{os.path.basename(path)} is not in data.lock.json"]
    got = fingerprint(path)["suites"]
    problems = []
    for s in sorted(suites or lock["suites"]):
        want = lock["suites"].get(s)
        if want is None:
            problems.append(f"{s}: not a frozen suite")
        elif s not in got:
            problems.append(f"{s}: missing from {path}")
        elif got[s] != want:
            problems.append(f"{s}: {got[s]['n']} cases, hash {got[s]['sha256'][:12]}; frozen: {want['n']} cases, hash {want['sha256'][:12]}")
    return problems


def verified_sha(path):
    """The frozen hash of a cases file, or exit with what differs."""
    problems = check(path)
    if problems:
        sys.exit(f"{path} doesn't match data.lock.json, so results wouldn't compare with the board:\n  " + "\n  ".join(problems)
                 + "\nRebuild it with the build_*.py scripts, or pass --unverified for a run that isn't for the board.")
    return json.load(open(LOCK))[os.path.basename(path)]["sha256"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", nargs="+", metavar="CASES", help="freeze these cases files into data.lock.json")
    a = ap.parse_args()
    if a.write:
        lock = json.load(open(LOCK)) if os.path.exists(LOCK) else {}
        for path in a.write:
            lock[os.path.basename(path)] = fingerprint(path)
        with open(LOCK, "w") as f:
            f.write(json.dumps(lock, indent=1) + "\n")
        print("froze", ", ".join(a.write))
        sys.exit()
    bad = 0
    for name, entry in json.load(open(LOCK)).items():
        problems = check(name) if os.path.exists(name) else [f"{name} not found; build it first"]
        print(f"{name}: {'matches' if not problems else 'DOES NOT MATCH'} ({sum(v['n'] for v in entry['suites'].values())} cases, {entry['sha256'][:12]})")
        for p in problems:
            print("  " + p)
        bad += bool(problems)
    sys.exit(1 if bad else 0)
