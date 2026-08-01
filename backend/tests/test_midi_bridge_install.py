import importlib.util
from pathlib import Path
import plistlib
import shutil
import unittest
from unittest.mock import patch
import uuid


class BridgeInstallTest(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).resolve().parents[2] / "tools" / "midi-bridge"
        spec = importlib.util.spec_from_file_location("bridge_launcher", self.source / "run.py")
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)
        self.home = self.source / ".build" / ("install-test-" + uuid.uuid4().hex)
        self.home.mkdir(parents=True)
        self.helper = self.home / "helper"
        self.helper.write_bytes(b"native helper fixture")

    def tearDown(self):
        shutil.rmtree(self.home)

    def test_login_install_copies_runtime_and_preserves_exports_on_reinstall(self):
        install = self.home / "Library" / "Application Support" / "TheArtist" / "MIDI Bridge"
        exports = install / ".build" / "exports"
        exports.mkdir(parents=True)
        score = exports / "my-score.mid"
        score.write_bytes(b"existing score")
        with patch.object(Path, "home", return_value=self.home), \
                patch.object(self.launcher.subprocess, "run") as run, \
                patch.object(self.launcher, "bridge_running", side_effect=[False, True, False, True]):
            self.launcher.install_at_login(self.source, self.helper)
            self.launcher.install_at_login(self.source, self.helper)
        plist = self.home / "Library" / "LaunchAgents" / (self.launcher.LABEL + ".plist")
        config = plistlib.loads(plist.read_bytes())
        self.assertTrue(config["KeepAlive"])
        self.assertTrue(config["RunAtLoad"])
        self.assertEqual(config["ProgramArguments"], ["/usr/bin/python3", str(install / "bridge.py"),
                                                    "--helper", str(install / "CoreMidiInput")])
        self.assertEqual((install / "bridge.py").read_bytes(), (self.source / "bridge.py").read_bytes())
        self.assertEqual((install / "CoreMidiInput").read_bytes(), self.helper.read_bytes())
        self.assertEqual(score.read_bytes(), b"existing score")
        self.assertEqual(run.call_args.args[0][1], "bootstrap")

    def test_install_refuses_symbolic_link_without_stopping_any_service(self):
        library = self.home / "Library"
        library.mkdir()
        (library / "LaunchAgents").symlink_to(self.home, target_is_directory=True)
        with patch.object(Path, "home", return_value=self.home), \
                patch.object(self.launcher.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "symbolic link"):
                self.launcher.install_at_login(self.source, self.helper)
        run.assert_not_called()

    def test_foreground_bridge_conflict_does_not_write_installation(self):
        with patch.object(Path, "home", return_value=self.home), \
                patch.object(self.launcher.subprocess, "run") as run, \
                patch.object(self.launcher, "bridge_running", return_value=True), \
                patch.object(self.launcher.time, "sleep"):
            with self.assertRaisesRegex(SystemExit, "already running"):
                self.launcher.install_at_login(self.source, self.helper)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.home / "Library").exists())
