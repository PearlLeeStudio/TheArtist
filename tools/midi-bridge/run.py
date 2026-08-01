"""Build the Mac bridge, or install it to run automatically at login."""
import argparse
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

LABEL = "co.pearlleestudio.theartist-midi-bridge"


def agent_config(install):
    """Use the system Python and installed copies, independent of the checkout."""
    return {
        "Label": LABEL,
        "ProgramArguments": ["/usr/bin/python3", str(install / "bridge.py"),
                             "--helper", str(install / "CoreMidiInput")],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 5,
        "ProcessType": "Background",
        "EnvironmentVariables": {"PYTHONDONTWRITEBYTECODE": "1"},
        "StandardOutPath": str(install / "bridge.log"),
        "StandardErrorPath": str(install / "bridge-error.log"),
    }


def bridge_running():
    try:
        with urlopen("http://127.0.0.1:8768/health", timeout=1) as response:
            return json.load(response).get("service") == "theartist-midi-v1"
    except (OSError, URLError, ValueError):
        return False


def install_at_login(source, helper):
    library = Path.home() / "Library"
    install = library / "Application Support" / "TheArtist" / "MIDI Bridge"
    agents = library / "LaunchAgents"
    plist = agents / (LABEL + ".plist")
    # Installation touches only this helper's own files. Exported MIDI is retained.
    for path in [install, *install.parents, agents, plist,
                 install / "bridge.py", install / "CoreMidiInput"]:
        if path.is_symlink():
            raise SystemExit(f"Refusing to install through a symbolic link: {path}")
    if not Path("/usr/bin/python3").is_file():
        raise SystemExit("Install Apple's command-line developer tools first.")
    domain = f"gui/{os.getuid()}"
    subprocess.run(["/bin/launchctl", "bootout", f"{domain}/{LABEL}"],
                   capture_output=True, check=False)
    # A foreground copy must be stopped before launchd can own the port.
    for _ in range(20):
        if not bridge_running():
            break
        time.sleep(0.1)
    else:
        raise SystemExit("A bridge is already running. Stop its terminal process, then run --install again.")
    install.mkdir(parents=True, exist_ok=True, mode=0o700)
    agents.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "bridge.py", install / "bridge.py")
    shutil.copy2(helper, install / "CoreMidiInput")
    (install / "CoreMidiInput").chmod(0o700)
    with plist.open("wb") as output:
        plistlib.dump(agent_config(install), output)
    plist.chmod(0o600)
    subprocess.run(["/bin/launchctl", "enable", f"{domain}/{LABEL}"], check=True)
    subprocess.run(["/bin/launchctl", "bootstrap", domain, str(plist)], check=True)
    for _ in range(40):
        if bridge_running():
            print("TheArtist MIDI Bridge installed and responding at http://127.0.0.1:8768.")
            print("It starts at login and restarts if it stops. You can close this terminal.")
            return
        time.sleep(0.25)
    raise SystemExit(f"Installed, but the bridge did not respond. See {install / 'bridge-error.log'}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true",
                        help="install a per-user background service; starts at login and restarts on exit")
    parser.add_argument("--status", action="store_true", help="check whether the local bridge responds")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise SystemExit("The local bridge currently supports macOS. Chrome Web MIDI works without it.")
    if args.status:
        running = bridge_running()
        print("TheArtist MIDI Bridge is running." if running else "TheArtist MIDI Bridge is not running.")
        raise SystemExit(0 if running else 1)
    source = Path(__file__).resolve().parent
    build = source / ".build"
    build.mkdir(exist_ok=True)
    cache = build / "ModuleCache"
    cache.mkdir(exist_ok=True)
    helper = build / "CoreMidiInput"
    swift = source / "CoreMidiInput.swift"
    if not helper.exists() or helper.stat().st_mtime < swift.stat().st_mtime:
        subprocess.run(["/usr/bin/swiftc", "-module-cache-path", str(cache), str(swift), "-o", str(helper)], check=True)
    if args.install:
        install_at_login(source, helper)
        return
    os.execv(sys.executable, [sys.executable, str(source / "bridge.py"), "--helper", str(helper)])


if __name__ == "__main__":
    main()
