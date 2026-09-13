#!/usr/bin/env python3
"""Resume draft releases safely after transient GitHub API failures."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote


class GitHubError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        match = re.search(r"HTTP (\d{3})", message)
        self.status = int(match[1]) if match else None


def gh(*args, payload=None):
    result = subprocess.run(
        ["gh", *args], input=json.dumps(payload) if payload is not None else None,
        text=True, capture_output=True,
    )
    if result.returncode:
        raise GitHubError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def retry(action):
    for attempt in range(5):
        try:
            return action()
        except GitHubError as error:
            # 422 can follow a successful create whose response was lost.
            if error.status not in (None, 408, 429, 422, 500, 502, 503, 504) or attempt == 4:
                raise
            delay = min(5 * 2 ** attempt, 40)
            print(f"GitHub request failed: {error}; retrying in {delay}s", file=sys.stderr)
            time.sleep(delay)


class Publisher:
    def __init__(self):
        self.repo = os.environ["GH_REPO"]
        self.tag = os.environ["RELEASE_TAG"]
        self.commit = os.environ["GITHUB_SHA"]
        self.prefix = f"repos/{self.repo}"

    def api(self, path, payload=None, method="GET"):
        args = ["api", f"{self.prefix}/{path}", "--method", method]
        if payload is not None:
            args += ["--input", "-"]
        return json.loads(gh(*args, payload=payload))

    def lookup(self, path):
        try:
            return retry(lambda: self.api(path))
        except GitHubError as error:
            if error.status == 404:
                return None
            raise

    def release(self):
        return self.lookup(f"releases/tags/{quote(self.tag, safe='')}")

    def ensure_tag(self):
        def ensure():
            if self.lookup(f"git/ref/tags/{quote(self.tag, safe='')}") is None:
                self.api("git/refs", {"ref": f"refs/tags/{self.tag}", "sha": self.commit}, "POST")
        retry(ensure)

    def ensure_draft(self):
        def ensure():
            release = self.release()
            if release is None:
                # Keep automatic notes generation out of the release-create request.
                release = self.api("releases", {
                    "tag_name": self.tag, "target_commitish": self.commit,
                    "name": self.tag, "draft": True,
                    "generate_release_notes": False,
                }, "POST")
            if not release["draft"]:
                raise RuntimeError(f"Release {self.tag} is already published; refusing to replace assets")
            return release
        return retry(ensure)

    def publish(self, directory):
        assets = sorted(p for p in directory.iterdir()
                        if p.name.endswith((".zip", ".tar.gz", ".sha256")))
        if not assets or not any(p.name.endswith(".sha256") for p in assets):
            raise RuntimeError("No release packages/checksums found")
        self.ensure_tag()
        release = self.ensure_draft()
        release_path = f"releases/{release['id']}"
        if not release.get("body"):
            try:
                notes = retry(lambda: self.api("releases/generate-notes", {
                    "tag_name": self.tag, "target_commitish": self.commit,
                }, "POST"))["body"]
            except GitHubError as error:
                print(f"::warning::Automatic release notes unavailable: {error}", file=sys.stderr)
                notes = f"Release {self.tag}. See the repository commit history for changes."
            retry(lambda: self.api(release_path, {"body": notes}, "PATCH"))
        for asset in assets:
            print(f"Uploading {asset.name}", flush=True)
            retry(lambda: gh("release", "upload", self.tag, str(asset),
                             "--repo", self.repo, "--clobber"))
        # PATCH is idempotent, including when publishing succeeded but its response was lost.
        retry(lambda: self.api(release_path, {"draft": False, "make_latest": "true"}, "PATCH"))
        print(f"Published {self.tag}")


if __name__ == "__main__":
    publisher = Publisher()
    if "--check" in sys.argv:
        release = publisher.release()
        print("true" if release is None or release["draft"] else "false")
    else:
        publisher.publish(Path("dist"))
