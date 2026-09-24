"""Git integration checks for the sidebar updater."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


UPDATER_PATH = Path(__file__).resolve().parents[1] / "utils" / "updater.py"
spec = importlib.util.spec_from_file_location("o1key_updater_under_test", UPDATER_PATH)
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


class UpdaterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.remote = root / "remote.git"
        self.author = root / "author"
        self.install = root / "install"
        self.git(root, "init", "--bare", str(self.remote))
        self.git(root, "clone", str(self.remote), str(self.author))
        self.git(self.author, "config", "user.email", "test@example.com")
        self.git(self.author, "config", "user.name", "Updater Test")
        self.git(self.author, "switch", "-c", "main")
        (self.author / "requirements.txt").write_text("requests>=2\n", encoding="utf-8")
        (self.author / "version.txt").write_text("1\n", encoding="utf-8")
        self.commit_and_push()
        self.git(root, "clone", "--branch", "main", str(self.remote), str(self.install))
        updater.PLUGIN_DIR = self.install

    def git(self, cwd, *args):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    def commit_and_push(self):
        self.git(self.author, "add", ".")
        self.git(self.author, "commit", "-m", "test update")
        self.git(self.author, "push", "origin", "main")

    def test_fast_forward_and_requirements_change(self):
        self.assertFalse(updater.update_package()["updated"])
        (self.author / "version.txt").write_text("2\n", encoding="utf-8")
        (self.author / "requirements.txt").write_text("requests>=3\n", encoding="utf-8")
        self.commit_and_push()
        result = updater.update_package()
        self.assertTrue(result["updated"])
        self.assertTrue(result["requirements_changed"])
        self.assertEqual((self.install / "version.txt").read_text(encoding="utf-8"), "2\n")

    def test_local_changes_are_preserved(self):
        (self.install / "version.txt").write_text("local\n", encoding="utf-8")
        with self.assertRaisesRegex(updater.UpdateError, "本地修改"):
            updater.update_package()
        self.assertEqual((self.install / "version.txt").read_text(encoding="utf-8"), "local\n")

    def test_diverged_branch_is_rejected(self):
        self.git(self.install, "config", "user.email", "test@example.com")
        self.git(self.install, "config", "user.name", "Updater Test")
        (self.install / "version.txt").write_text("local commit\n", encoding="utf-8")
        self.git(self.install, "add", ".")
        self.git(self.install, "commit", "-m", "local")
        (self.author / "version.txt").write_text("remote commit\n", encoding="utf-8")
        self.commit_and_push()
        with self.assertRaisesRegex(updater.UpdateError, "已分叉"):
            updater.update_package()
        self.assertEqual((self.install / "version.txt").read_text(encoding="utf-8"), "local commit\n")


if __name__ == "__main__":
    unittest.main()
