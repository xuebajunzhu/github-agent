"""Push unpushed commits to GitHub via the Git Data API.

Transport workaround for networks where github.com:443 (git smart HTTP) is
blocked but api.github.com is reachable. Walks origin/main..main in order
and replays each commit through blobs -> trees -> commits -> ref update,
preserving full history.

    .venv/Scripts/python.exe scripts/push_via_api.py
"""
from __future__ import annotations

import base64
import subprocess
import sys

import httpx

REPO = "xuebajunzhu/github-agent"


def git(*args: str) -> str:
    git_exe = r"C:\Program Files\Git\cmd\git.exe"
    result = subprocess.run([git_exe, *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main() -> None:
    diagnose = "--diagnose" in sys.argv
    token = subprocess.run(
        [r"C:\Program Files\GitHub CLI\gh.exe", "auth", "token"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    client = httpx.Client(base_url=f"https://api.github.com/repos/{REPO}", headers=headers, timeout=60)

    current = client.get("/git/refs/heads/main")
    current.raise_for_status()
    remote_sha = current.json()["object"]["sha"]
    print(f"remote main: {remote_sha[:12]}")

    pending = git("log", "--reverse", "--format=%H", f"{remote_sha}..main").split()
    if not pending:
        print("nothing to push")
        return
    print(f"pushing {len(pending)} commit(s) via API")

    api_parent = remote_sha
    for commit_sha in pending:
        local_parent = git("rev-parse", f"{commit_sha}^")
        if local_parent != api_parent:
            print(
                f"ABORT: local parent {local_parent[:12]} != API chain {api_parent[:12]} "
                "(refusing to create divergent history)"
            )
            sys.exit(1)
        message = git("log", "-1", "--format=%B", commit_sha)
        # exact identity + UTC-normalized dates -> the API-built commit hashes
        # byte-identically to the local one, so histories never diverge
        (an, ae, ad, cn, ce, cd) = git(
            "show", "-s", "--format=%an%n%ae%n%ad%n%cn%n%ce%n%cd", "--date=iso-strict", commit_sha
        ).splitlines()

        def to_utc_z(value: str) -> str:
            from datetime import datetime, timezone

            return datetime.fromisoformat(value).astimezone(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        identity = {
            "author": {"name": an, "email": ae, "date": to_utc_z(ad)},
            "committer": {"name": cn, "email": ce, "date": to_utc_z(cd)},
        }

        parent_tree = git("rev-parse", f"{local_parent}^{{tree}}")
        local_tree = git("ls-tree", "-r", commit_sha).splitlines()
        parent_files = {}
        for line in git("ls-tree", "-r", local_parent).splitlines():
            meta, path = line.split("\t", 1)
            mode, kind, sha = meta.split()
            parent_files[path] = (mode, kind, sha)
        local_files = {}
        for line in local_tree:
            meta, path = line.split("\t", 1)
            mode, kind, sha = meta.split()
            local_files[path] = (mode, kind, sha)

        entries = []
        changed = [
            path
            for path in set(parent_files) | set(local_files)
            if parent_files.get(path) != local_files.get(path)
        ]
        print(f"  {commit_sha[:12]}: {len(changed)} changed file(s)")
        for path in changed:
            if path not in local_files:
                entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
                continue
            mode, kind, local_sha = local_files[path]
            if kind != "blob":
                entries.append({"path": path, "mode": mode, "type": kind, "sha": local_sha})
                continue
            blob_resp = client.post(
                "/git/blobs",
                json={
                    "content": base64.b64encode(_blob_bytes(local_sha)).decode(),
                    "encoding": "base64",
                },
            )
            blob_resp.raise_for_status()
            entries.append({"path": path, "mode": mode, "type": "blob", "sha": blob_resp.json()["sha"]})

        tree_resp = client.post("/git/trees", json={"base_tree": parent_tree, "tree": entries})
        tree_resp.raise_for_status()
        commit_resp = client.post(
            "/git/commits",
            json={
                "message": message,
                "tree": tree_resp.json()["sha"],
                "parents": [api_parent],
                **identity,
            },
        )
        if commit_resp.status_code >= 400:
            print(f"    commit failed: {commit_resp.status_code} {commit_resp.text[:400]}")
            sys.exit(1)
        new_sha = commit_resp.json()["sha"]
        if new_sha != commit_sha:
            print(
                f"ABORT: API commit {new_sha[:12]} != local {commit_sha[:12]} "
                "(refusing divergent history)"
            )
            sys.exit(1)
        print(f"    -> {new_sha[:12]} (byte-identical to local)")
        api_parent = new_sha
        if diagnose:
            created = client.get(f"/git/commits/{new_sha}").json()
            local_raw = subprocess.run(
                [r"C:\Program Files\Git\cmd\git.exe", "cat-file", "commit", commit_sha],
                capture_output=True, text=True, check=True,
            ).stdout
            print("--- API object ---")
            print(json.dumps({k: created[k] for k in ("tree", "author", "committer", "message")}, indent=2)[:800])
            print("--- LOCAL object ---")
            print(local_raw[:800])
            sys.exit(2)

    update = client.patch("/git/refs/heads/main", json={"sha": api_parent, "force": False})
    update.raise_for_status()
    print(f"main updated to {parent[:12]}")
    if update.status_code == 200:
        git("update-ref", "refs/remotes/origin/main", parent)
        print("PUSH VIA API OK")


def _blob_bytes(sha: str) -> bytes:
    return subprocess.run(
        [r"C:\Program Files\Git\cmd\git.exe", "cat-file", "blob", sha],
        capture_output=True, check=True,
    ).stdout


if __name__ == "__main__":
    sys.exit(main())
