"""Publish a GitHub release for every notes file in releases/, and keep published notes in sync with the files.

Each releases/vX.Y.Z.md starts with a front matter block:

    ---
    title: "v1.6.0: Searchable site"
    target: <full commit SHA the tag points at; defaults to the commit being built>
    date: 2026-10-15
    ---

Run by .github/workflows/release.yml with GH_TOKEN set. `--dry-run` prints what it would do.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DRY = "--dry-run" in sys.argv


def parse(path):
    text = path.read_text()
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        sys.exit(f"{path}: missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    body = text[m.end():].strip()
    if meta.get("date"):
        sha = meta.get("target") or os.environ.get("GITHUB_SHA", "")
        body = f"_Released {meta['date']}" + (f", commit `{sha[:7]}`" if sha else "") + "._\n\n" + body
    return meta, body + "\n"


def version(path):
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", path.stem)
    if not m:
        sys.exit(f"{path}: name it vMAJOR.MINOR.PATCH.md")
    return tuple(map(int, m.groups()))


def gh(*args, capture=False):
    if DRY and not capture:
        print("gh", *args)
        return ""
    return subprocess.run(["gh", *args], check=True, text=True, capture_output=capture).stdout


files = sorted(Path("releases").glob("v*.md"), key=version)
newest = files[-1].stem if files else None
published = {} if DRY else {r["tagName"]: r for r in json.loads(gh("release", "list", "--limit", "500", "--json", "tagName", capture=True))}

for path in files:
    tag, (meta, body) = path.stem, parse(path)
    title = meta.get("title") or tag
    target = meta.get("target") or os.environ.get("GITHUB_SHA", "")
    if target and subprocess.run(["git", "merge-base", "--is-ancestor", target, "HEAD"]).returncode != 0:
        sys.exit(f"{path}: target {target} is not a commit on this branch")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
    if tag not in published:
        print(f"{tag}: creating at {target[:7]}")
        gh("release", "create", tag, "--target", target, "--title", title, "--notes-file", f.name,
           f"--latest={'true' if tag == newest else 'false'}")
        continue
    live = json.loads(gh("release", "view", tag, "--json", "name,body", capture=True))
    if live["name"] != title or live["body"].strip() != body.strip():
        print(f"{tag}: updating notes")
        gh("release", "edit", tag, "--title", title, "--notes-file", f.name)
    else:
        print(f"{tag}: up to date")
