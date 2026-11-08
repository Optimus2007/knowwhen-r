import subprocess
import sys
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[1])


def run(env):
    code = "from knowwhen import paths; print(paths.ROOT, paths.ENV)"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.split()


def test_kw_root_alone_cannot_move_writes_elsewhere():
    env = dict(PYTHONPATH="src", KW_ROOT="/tmp/elsewhere", KW_ENV="/tmp/elsewhere2")
    assert run(env) == [REPO, "/Volumes/KWenv"]


def test_drive_of_finds_a_mount_point():
    sys.path.insert(0, str(Path(REPO) / "src"))
    from knowwhen.paths import drive_of
    import os
    assert os.path.ismount(drive_of(Path(REPO)))


def test_remote_mode_needs_the_explicit_switch():
    env = dict(PYTHONPATH="src", KW_REMOTE="1", KW_ROOT="/kaggle/working/kw", KW_ENV="/kaggle/working/kw_env")
    assert run(env) == ["/kaggle/working/kw", "/kaggle/working/kw_env"]
