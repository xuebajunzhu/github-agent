"""Set up GitHub remote: extract stored credential, verify identity, create repo."""
import json
import subprocess
import sys

import httpx

GIT = r"C:\Program Files\Git\cmd\git.exe"

env = {"GCM_INTERACTIVE": "never", "GIT_TERMINAL_PROMPT": "0", "PATH": r"C:\Program Files\Git\cmd;C:\Windows\System32"}
fill = subprocess.run(
    [GIT, "credential", "fill"],
    input="protocol=https\nhost=github.com\n\n",
    capture_output=True,
    text=True,
    env=env,
    timeout=60,
)
if fill.returncode != 0:
    sys.exit(f"credential fill failed: {fill.stderr[:300]}")

token = None
username = None
for line in fill.stdout.splitlines():
    if line.startswith("password="):
        token = line.split("=", 1)[1]
    if line.startswith("username="):
        username = line.split("=", 1)[1]
if not token:
    sys.exit("no stored GitHub credential found")
print(f"credential found for user: {username} (token {token[:4]}...{token[-4:]}, len={len(token)})")

headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
with httpx.Client(headers=headers, timeout=30) as client:
    me = client.get("https://api.github.com/user")
    print("GET /user:", me.status_code, me.json().get("login"))
    owner = me.json().get("login", username)

    repo_resp = client.post(
        "https://api.github.com/user/repos",
        json={
            "name": "github-agent",
            "description": "Self-evolving GitHub open-source retrieval & analysis agent (trained local model, 24/7 patrol, voice in/out)",
            "private": False,
            "has_issues": True,
            "has_wiki": False,
        },
    )
    print("POST /user/repos:", repo_resp.status_code)
    if repo_resp.status_code == 201:
        print("repo created:", repo_resp.json().get("html_url"))
    elif repo_resp.status_code == 422:
        print("repo already exists")
    else:
        print(repo_resp.text[:300])

    scopes = client.get("https://api.github.com/user").headers.get("x-oauth-scopes", "?")
    print("token scopes:", scopes)
