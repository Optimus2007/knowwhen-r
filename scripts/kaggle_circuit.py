"""Entry point for the CircuitKIT study on Kaggle. Each stage saves its own outputs, so a later failure loses nothing."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT, ENV = Path("/kaggle/working/kw"), Path("/kaggle/working/kw_env")


def main() -> None:
    from kaggle_secrets import UserSecretsClient
    env = dict(os.environ, KW_REMOTE="1", KW_ROOT=str(ROOT), KW_ENV=str(ENV), PYTHONPATH=str(HERE / "src"), HF_TOKEN=UserSecretsClient().get_secret("HF_TOKEN"),
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "circuitkit==0.2.0", "transformers==5.22.1", "accelerate"], check=True)
    (ROOT / "experiments" / "circuit").mkdir(parents=True, exist_ok=True)
    ENV.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "grounding_pairs.csv", ROOT / "experiments" / "circuit" / "grounding_pairs.csv")
    base = [sys.executable, str(HERE / "scripts" / "circuit_discover.py"), "--device", "cuda", "--precision", "float16", "--batch", "1"]
    def stage(args, limit):
        print("+", args, flush=True)
        try:
            code = subprocess.run(base + args, env=env, cwd=ROOT, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        print("exit", code, flush=True)
        return code

    # Heads need a large per head tensor in TransformerLens. If that does not fit, fall back to MLPs.
    scope = "both" if stage(["--n", "16", "--tag", "smoke"], 2400) == 0 else "mlp"
    stage(["--n", "128", "--tag", f"tiny_{scope}", "--scope", scope, "--seed", "0", "--evaluate", "64"], 9000)
    stage(["--n", "128", "--tag", f"tiny_{scope}", "--scope", scope, "--seed", "1"], 7200)


if __name__ == "__main__":
    main()
