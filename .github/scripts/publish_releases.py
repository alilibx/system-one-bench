"""Publish a GitHub release for every notes file in releases/, and keep published notes in sync with the files.

Each releases/vX.Y.Z.md starts with a front matter block:

    ---
    title: "v1.6.0: Searchable site"
    target: <full commit SHA the tag points at; defaults to the commit being built>
    date: 2026-10-15
    ---

Run by .github/workflows/release.yml with GH_TOKEN set. `--dry-run` prints what it would do.

The workflow's token can't publish a release on a commit whose workflow files differ from main's (GitHub answers 403,
even when the tag already exists), which is the case for older commits. For those, run this script from a clone where
`gh` is logged in as a maintainer with the `workflow` scope (`gh auth refresh -h github.com -s workflow`): it skips
releases that already exist.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DRY = "--dry-run" in sys.argv


def tag_commit(tag):
    out = subprocess.run(["git", "rev-list", "-n", "1", tag], text=True, capture_output=True)
    return out.stdout.strip() if out.returncode == 0 else ""


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
    # An existing tag wins, so later runs (with a different GITHUB_SHA) don't rewrite the commit in the notes.
    meta["target"] = meta.get("target") or tag_commit(path.stem) or os.environ.get("GITHUB_SHA", "")
    if meta.get("date"):
        sha = meta["target"]
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


failed = []
files = sorted(Path("releases").glob("v*.md"), key=version)
newest = files[-1].stem if files else None
published = {} if DRY else {r["tagName"]: r for r in json.loads(gh("release", "list", "--limit", "500", "--json", "tagName", capture=True))}

for path in files:
    tag, (meta, body) = path.stem, parse(path)
    title = meta.get("title") or tag
    target = meta["target"]
    if target and subprocess.run(["git", "merge-base", "--is-ancestor", target, "HEAD"]).returncode != 0:
        sys.exit(f"{path}: target {target} is not a commit on this branch")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
    if tag not in published:
        print(f"{tag}: creating at {target[:7]}")
        try:
            gh("release", "create", tag, "--target", target, "--title", title, "--notes-file", f.name,
               f"--latest={'true' if tag == newest else 'false'}")
        except subprocess.CalledProcessError:
            failed.append(tag)
        continue
    live = json.loads(gh("release", "view", tag, "--json", "name,body", capture=True))
    if live["name"] != title or live["body"].strip() != body.strip():
        print(f"{tag}: updating notes")
        gh("release", "edit", tag, "--title", title, "--notes-file", f.name)
    else:
        print(f"{tag}: up to date")

if failed:
    print(f"\nThe workflow's token can't publish {', '.join(failed)}: their commits' workflow files differ from main's.")
    print("Publish them from a clone where gh is logged in as a maintainer with the workflow scope:")
    print("  gh auth refresh -h github.com -s workflow")
    print("  git pull && git fetch --tags && python3 .github/scripts/publish_releases.py")
    sys.exit(1)
