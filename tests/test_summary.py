import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from summarize_final import holm  # noqa: E402


def test_holm_matches_hand_calculation():
    assert [round(float(x), 4) for x in holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]


def test_holm_caps_at_one():
    assert max(holm([0.9, 0.8])) == 1
