import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from publish_release import GitHubError, Publisher, retry


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, GH_REPO="owner/repo", RELEASE_TAG="v1.4.1", GITHUB_SHA="abc")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sleep = patch("publish_release.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.publisher = Publisher()

    def test_create_500_after_success_resumes_existing_draft(self):
        draft = {"id": 1, "draft": True}
        with patch.object(self.publisher, "release", side_effect=[None, draft]), patch.object(
            self.publisher, "api", side_effect=GitHubError("HTTP 500")
        ) as api:
            self.assertEqual(self.publisher.ensure_draft(), draft)
            self.assertEqual(api.call_count, 1)
            self.assertFalse(api.call_args.args[1]["generate_release_notes"])

    def test_tag_500_after_success_does_not_recreate(self):
        with patch.object(self.publisher, "lookup", side_effect=[None, {"ref": "tag"}]), patch.object(
            self.publisher, "api", side_effect=GitHubError("HTTP 500")
        ) as api:
            self.publisher.ensure_tag()
            self.assertEqual(api.call_count, 1)

    def test_published_release_is_never_overwritten(self):
        with patch.object(self.publisher, "release", return_value={"draft": False}):
            with self.assertRaisesRegex(RuntimeError, "already published"):
                self.publisher.ensure_draft()

    def test_only_404_means_missing(self):
        with patch.object(self.publisher, "api", side_effect=[GitHubError("HTTP 404"), []]):
            self.assertIsNone(self.publisher.release())
        for status in (401, 403, 500):
            with patch.object(self.publisher, "api", side_effect=GitHubError(f"HTTP {status}")):
                with self.assertRaises(GitHubError):
                    self.publisher.release()

    def test_permission_failure_is_not_retried(self):
        with patch("publish_release.gh", side_effect=GitHubError("HTTP 403")) as call:
            with self.assertRaises(GitHubError):
                retry(call)
            self.assertEqual(call.call_count, 1)

    def test_draft_found_via_list_when_tag_lookup_is_missing(self):
        draft = {"id": 42, "draft": True, "tag_name": "v1.4.1"}
        with patch.object(self.publisher, "api", side_effect=[GitHubError("HTTP 404"), [draft]]):
            self.assertEqual(self.publisher.release(), draft)

    def test_orphan_draft_found_by_name_when_tag_name_is_placeholder(self):
        draft = {"id": 43, "draft": True, "tag_name": "untagged-abc123", "name": "v1.4.1"}
        with patch.object(self.publisher, "api", side_effect=[GitHubError("HTTP 404"), [draft]]):
            self.assertEqual(self.publisher.release(), draft)

    def test_upload_uses_url_and_cleans_failed_asset_on_retry(self):
        release = {"id": 42, "upload_url": "https://uploads.github.com/repos/owner/repo/releases/42/assets{?name,label}"}
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / "package test.zip"
            asset.write_bytes(b"zip")
            with patch.object(self.publisher, "api", side_effect=[
                {"draft": True}, [],
                {"draft": True}, [{"id": 7, "name": asset.name}], None,
            ]) as api, patch("publish_release.gh", side_effect=[
                GitHubError("HTTP 502"), '{"state":"uploaded","size":3}',
            ]) as command:
                self.publisher.upload_asset(release, asset)
                self.assertEqual(command.call_count, 2)
                args = command.call_args.args
                self.assertEqual(args[0], "api")
                self.assertIn("/releases/42/assets?name=package%20test.zip", args[1])
                self.assertEqual(args[-2:], ("--input", str(asset)))
                api.assert_any_call("releases/assets/7", method="DELETE")

    def test_delete_accepts_empty_204_response(self):
        with patch("publish_release.gh", return_value=""):
            self.assertIsNone(self.publisher.api("releases/assets/7", method="DELETE"))

    def run_publish(self, upload_failure=False, draft_tag_name="v1.4.1", api=None):
        calls = []
        state = {"tag_name": draft_tag_name}
        if api is None:
            def api(path, payload=None, method="GET"):
                calls.append((path, payload, method))
                if path == "releases/generate-notes":
                    raise GitHubError("HTTP 500")
                if path == "releases/42":
                    if method == "GET":
                        return dict(state)
                    state.update(payload or {})
                return {}
        else:
            base_api = api
            def api(path, payload=None, method="GET"):
                calls.append((path, payload, method))
                return base_api(path, payload, method)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "package.zip").write_bytes(b"package")
            (directory / "package.zip.sha256").write_text("checksum")
            with patch.object(self.publisher, "ensure_tag"), patch.object(
                self.publisher, "ensure_draft",
                return_value={"id": 42, "draft": True, "tag_name": draft_tag_name},
            ), patch.object(self.publisher, "api", side_effect=api), patch(
                "publish_release.Publisher.upload_asset", side_effect=GitHubError("HTTP 500") if upload_failure else None
            ):
                if upload_failure:
                    with self.assertRaises(GitHubError):
                        self.publisher.publish(directory)
                else:
                    self.publisher.publish(directory)
        return calls

    def test_placeholder_tag_is_renamed_before_publish(self):
        calls = self.run_publish(draft_tag_name="untagged-abc123")
        rename = ("releases/42", {"tag_name": "v1.4.1"}, "PATCH")
        publish = ("releases/42", {"draft": False, "make_latest": "true"}, "PATCH")
        self.assertIn(rename, calls)
        self.assertLess(calls.index(rename), calls.index(publish))

    def test_correctly_named_draft_is_not_renamed(self):
        calls = self.run_publish()
        self.assertFalse(any(
            path == "releases/42" and method == "PATCH" and payload and "tag_name" in payload
            for path, payload, method in calls
        ))

    def test_notes_failure_does_not_prevent_publication(self):
        calls = self.run_publish()
        flip = ("releases/42", {"draft": False, "make_latest": "true"}, "PATCH")
        self.assertIn(flip, calls)
        self.assertTrue(any("body" in (payload or {}) for _, payload, _ in calls))

    def test_body_patch_placeholder_rewrite_is_pinned_before_flip(self):
        # GitHub rewrites a draft's tag_name to an "untagged-<hex>" placeholder
        # when a PATCH touches the draft while the tag ref already exists; the
        # pin after the notes update must restore the real tag before the flip.
        tag = {"name": "v1.4.1"}
        def api(path, payload=None, method="GET"):
            if path == "releases/generate-notes":
                raise GitHubError("HTTP 500")
            if path == "releases/42":
                if method == "GET":
                    return {"tag_name": tag["name"]}
                if payload and "tag_name" in payload:
                    tag["name"] = payload["tag_name"]
                elif payload and "body" in payload:
                    tag["name"] = "untagged-rewrite1"
            return {}
        calls = self.run_publish(api=api)
        index = lambda pred: next(i for i, c in enumerate(calls) if pred(*c))
        body = index(lambda p, pay, m: p == "releases/42" and m == "PATCH" and "body" in (pay or {}))
        pin = index(lambda p, pay, m: m == "PATCH" and pay and "tag_name" in pay)
        flip = index(lambda p, pay, m: m == "PATCH" and pay and pay.get("draft") is False)
        self.assertLess(body, pin)
        self.assertLess(pin, flip)
        self.assertEqual(calls[pin][2], "PATCH")
        self.assertEqual(calls[pin][1]["tag_name"], "v1.4.1")

    def test_placeholder_after_flip_is_repaired_after_publish(self):
        # The rewrite can also strike the publish flip itself; the post-publish
        # pin must re-associate the release with the real tag instead of
        # leaving it where the version check can never find it.
        rewritten = {"done": False}
        def api(path, payload=None, method="GET"):
            if path == "releases/generate-notes":
                raise GitHubError("HTTP 500")
            if path == "releases/42":
                if method == "GET":
                    return {"tag_name": "untagged-flip" if rewritten["done"] else "v1.4.1"}
                if payload and payload.get("draft") is False:
                    rewritten["done"] = True
                elif payload and "tag_name" in payload:
                    rewritten["done"] = False
            return {}
        calls = self.run_publish(api=api)
        index = lambda pred: next(i for i, c in enumerate(calls) if pred(*c))
        flip = index(lambda p, pay, m: m == "PATCH" and pay and pay.get("draft") is False)
        heal = index(lambda p, pay, m: m == "PATCH" and pay and "tag_name" in pay)
        self.assertLess(flip, heal)
        self.assertEqual(calls[heal][1]["tag_name"], "v1.4.1")

    def test_placeholder_that_cannot_be_pinned_fails_the_publish(self):
        # Refuse to print success when the release would stay on a placeholder
        # tag; that binding is what made every push republish the version.
        def api(path, payload=None, method="GET"):
            if path == "releases/generate-notes":
                raise GitHubError("HTTP 500")
            if path == "releases/42" and method == "GET":
                return {"tag_name": "untagged-stuck"}
            return {}
        with self.assertRaisesRegex(RuntimeError, "stayed on tag"):
            self.run_publish(api=api)

    def test_upload_failure_keeps_release_draft(self):
        calls = self.run_publish(upload_failure=True)
        self.assertFalse(any((payload or {}).get("draft") is False for _, payload, _ in calls))


if __name__ == "__main__":
    unittest.main()
