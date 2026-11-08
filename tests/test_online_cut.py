import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("online_cut", Path(__file__).resolve().parents[1] / "scripts" / "online_cut.py")
oc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oc)


def test_steady_stream_reaches_the_target():
    rng = np.random.default_rng(0)
    warm, stream = rng.random(500), rng.random(5000)
    keep = oc.stream_keep(warm, stream, 0.7, 200)
    assert abs(keep.mean() - 0.7) < 0.03


def test_decision_never_uses_the_current_or_later_items():
    rng = np.random.default_rng(1)
    warm, stream = rng.random(300), rng.random(400)
    base = oc.stream_keep(warm, stream, 0.7, 100)
    changed = stream.copy()
    changed[250:] = rng.random(150) * 10  # alter the item at 250 and everything after it
    again = oc.stream_keep(warm, changed, 0.7, 100)
    assert (base[:250] == again[:250]).all()


def test_window_adapts_after_a_shift():
    rng = np.random.default_rng(2)
    warm = rng.random(800)  # confidences on the old datasets
    stream = rng.random(1000) * 0.5  # a new dataset where the model is less confident
    planned = (stream >= np.quantile(warm, 0.3)).mean()
    online = oc.stream_keep(warm, stream, 0.7, 100).mean()
    assert planned < 0.5 and abs(online - 0.7) < 0.08
