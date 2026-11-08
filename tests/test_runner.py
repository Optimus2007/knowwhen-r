"""Runs the real capture code on a tiny randomly initialized Cohere2 model."""
import copy

import numpy as np
import pytest
import torch
from transformers import Cohere2Config, Cohere2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers

from knowwhen.models.runner import build_prefix, generate_batch, select_layers


@pytest.fixture(scope="module")
def tiny():
    vocab = {"<pad>": 0, "<eos>": 1, **{f"w{i}": i + 2 for i in range(98)}, "[UNK]": 100}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="[UNK]")
    tok.padding_side = "left"
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    cfg = Cohere2Config(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0, eos_token_id=1, bos_token_id=None,
                        sliding_window=64, layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"])
    torch.manual_seed(0)
    return Cohere2ForCausalLM(cfg).eval(), tok


def test_select_layers_keeps_final():
    assert select_layers(37) == list(range(2, 37, 2))
    assert select_layers(36)[-1] == 35


def test_shapes_and_stats(tiny):
    model, tok = tiny
    b = generate_batch(model, tok, ["w1 w2 w3", "w4"], layers=[2, 4], max_new_tokens=5)
    assert b.pre.shape == (2, 2, 32) and b.last.shape == (2, 2, 32) and b.mean.shape == (2, 2, 32)
    assert np.isfinite(b.pre).all() and np.isfinite(b.mean).all()
    assert all(s["mean_entropy"] >= 0 and s["mean_logprob"] <= 0 for s in b.stats)


def test_left_padding_does_not_change_prompt_state(tiny):
    """A padded row must give the same pre generation state as the same row alone."""
    model, tok = tiny
    alone = generate_batch(model, tok, ["w4 w5"], layers=[4], max_new_tokens=3)
    padded = generate_batch(model, tok, ["w4 w5", "w1 w2 w3 w6 w7 w8"], layers=[4], max_new_tokens=3)
    assert np.allclose(alone.pre[0], padded.pre[0], atol=2e-2)
    assert alone.answers[0] == padded.answers[0]


def test_matches_generate_token_for_token(tiny):
    """The hand written loop must agree with the library on tokens, including padded rows."""
    from knowwhen.models.runner import encode
    model, tok = tiny
    prompts = ["w1 w2 w3", "w4", "w5 w6 w7 w8 w9 w10"]
    ours = generate_batch(model, tok, prompts, layers=[4], max_new_tokens=8)
    enc = encode(tok, prompts, "cpu")
    with torch.inference_mode():
        ref = model.generate(**enc, max_new_tokens=8, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=0)
    new = ref[:, enc["input_ids"].shape[1]:]
    for row, answer in zip(new.tolist(), ours.answers):
        cut = next((i for i, t in enumerate(row) if t in (0, 1)), len(row))
        assert tok.decode(row[:cut], skip_special_tokens=True).strip() == answer


def test_prefix_cache_matches_full_prompt(tiny):
    """Reusing the preamble cache must not change answers or captured states."""
    model, tok = tiny
    original = tok.chat_template
    tok.chat_template = "w90 w91 w92 w93 w94 {% for m in messages %}{{ m['content'] }}{% endfor %} w95 w96"
    try:
        prompts = ["w1 w2 w3", "w4", "w5 w6 w7 w8 w9 w10"]
        plain = generate_batch(model, tok, prompts, layers=[2, 4], max_new_tokens=6)
        prefix = build_prefix(model, tok)
        assert prefix.length == 5
        cached = generate_batch(model, tok, prompts, layers=[2, 4], max_new_tokens=6, prefix=prefix)
        again = generate_batch(model, tok, prompts, layers=[2, 4], max_new_tokens=6, prefix=prefix)
    finally:
        tok.chat_template = original
    assert plain.answers == cached.answers == again.answers  # again: the shared cache must stay unmodified
    assert np.allclose(plain.pre, cached.pre, atol=2e-3)
    assert np.allclose(plain.mean, cached.mean, atol=2e-3)


def test_prefix_matches_full_prompt_on_older_cohere_architecture(tiny):
    """Aya Expanse uses CohereForCausalLM, not Cohere2, so the preamble cache is checked on it too."""
    from transformers import CohereConfig, CohereForCausalLM
    _, tok = tiny
    tok = copy.copy(tok)
    tok.chat_template = "w90 w91 w92 w93 w94 {% for m in messages %}{{ m['content'] }} {% endfor %}w95"
    cfg = CohereConfig(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0, eos_token_id=1, bos_token_id=None)
    torch.manual_seed(1)
    model = CohereForCausalLM(cfg).float().eval()
    prompts = ["w1 w2 w3", "w4", "w5 w6 w7 w8 w9 w10"]
    plain = generate_batch(model, tok, prompts, layers=[2, 4], max_new_tokens=6)
    cached = generate_batch(model, tok, prompts, layers=[2, 4], max_new_tokens=6, prefix=build_prefix(model, tok))
    assert plain.answers == cached.answers
    assert np.allclose(plain.pre, cached.pre, atol=1e-4) and np.allclose(plain.mean, cached.mean, atol=1e-4)


def test_split_on_oom_keeps_every_row_once():
    from knowwhen.models.runner import split_on_oom

    done, notes = [], []

    def run(rows):
        if len(rows) > 2:
            raise torch.OutOfMemoryError("full")
        return rows

    split_on_oom(list(range(7)), run, lambda rows, out: done.extend(out), notes.append)
    assert done == list(range(7))
    assert len(notes) == 3


def test_split_on_oom_gives_up_on_a_single_row():
    from knowwhen.models.runner import split_on_oom

    def run(rows):
        raise torch.OutOfMemoryError("full")

    with pytest.raises(torch.OutOfMemoryError):
        split_on_oom([1, 2], run, lambda rows, out: None, lambda m: None)
