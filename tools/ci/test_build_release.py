import os
import unittest
from unittest.mock import patch

from build_release import BINARIES, PEFT_BINARY, peft_merge_enabled, release_binaries


class MergeBuildOptionTests(unittest.TestCase):
    def test_default_includes_merger_without_changing_other_binaries(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(peft_merge_enabled())
        self.assertEqual(release_binaries(True), BINARIES + (PEFT_BINARY,))

    def test_explicit_enable_and_disable(self):
        for value in ('true', 'ON', '1', ' on '):
            with self.subTest(value=value), patch.dict(os.environ, RWKV_BUILD_PEFT_MERGE=value):
                self.assertTrue(peft_merge_enabled())
        for value in ('false', 'OFF', '0', ' off '):
            with self.subTest(value=value), patch.dict(os.environ, RWKV_BUILD_PEFT_MERGE=value):
                self.assertFalse(peft_merge_enabled())
        self.assertEqual(release_binaries(False), BINARIES)
        self.assertNotIn(PEFT_BINARY, release_binaries(False))

    def test_invalid_option_fails_instead_of_silently_omitting_tool(self):
        for value in ('', 'yes', 'flase'):
            with self.subTest(value=value), patch.dict(os.environ, RWKV_BUILD_PEFT_MERGE=value):
                with self.assertRaises(ValueError):
                    peft_merge_enabled()


if __name__ == '__main__':
    unittest.main()
