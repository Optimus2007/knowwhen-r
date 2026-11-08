"""Greedy generation with hidden states and token statistics captured in the same pass."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    answers: list[str]
    stats: list[dict]
    pre: np.ndarray    # (B, layers, hidden) last prompt token, known before generating
    last: np.ndarray   # (B, layers, hidden) last answer token
    mean: np.ndarray   # (B, layers, hidden) mean over answer tokens


def select_layers(n_hidden_states: int, step: int = 2) -> list[int]:
    """Every step-th layer, always keeping the final one. Index 0 is the embedding output."""
    layers = list(range(step, n_hidden_states, step))
    if layers[-1] != n_hidden_states - 1:
        layers.append(n_hidden_states - 1)
    return layers


def load_model(model_id: str, device: str = "mps", dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "auto":   # split across every visible GPU, for a model too large for one
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return model, tokenizer


def _eos_ids(model, tokenizer) -> set[int]:
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, (list, tuple)) else [ids]) if ids is not None else set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def render(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def encode(tokenizer, prompts: list[str], device: str):
    return tokenizer([render(tokenizer, p) for p in prompts], return_tensors="pt", padding=True,
                     add_special_tokens=False).to(device)


@dataclass
class Prefix:
    """KV cache of the chat template text that precedes every user message.

    Cohere's template adds a default preamble of about 364 tokens to every prompt, and prompt
    processing is the bottleneck on an M2. The preamble is kept, so model behaviour is unchanged,
    but it is computed once.
    """
    length: int
    cache: object
    tail: str


@torch.inference_mode()
def build_prefix(model, tokenizer) -> Prefix:
    from transformers import DynamicCache

    head, tail = render(tokenizer, "@@USER@@").split("@@USER@@")
    ids = tokenizer(head, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]
    cache = DynamicCache(config=model.config)
    model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return Prefix(ids.shape[1], cache, tail)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str], layers: list[int] | None, max_new_tokens: int = 32,
                   prefix: Prefix | None = None) -> Batch:
    """Greedy decoding with capture, written by hand.

    The loop exists to capture states during decoding and to reuse the cached chat preamble,
    which is the real speedup: that 361 token preamble, not `model.generate`, was the slow part
    (see E001 in docs/experiment_registry.md). `tests/test_runner.py` checks it reproduces
    `generate` token for token.
    """
    from transformers import DynamicCache

    device = model.device
    if prefix is None:
        enc = encode(tokenizer, prompts, str(device))
        ids, mask = enc["input_ids"], enc["attention_mask"]
        position = (mask.long().cumsum(-1) - 1).clamp(min=0)
        cache = DynamicCache(config=model.config)
    else:
        enc = tokenizer([p + prefix.tail for p in prompts], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        ids, suffix_mask = enc["input_ids"], enc["attention_mask"]
        # Left padding now sits between the preamble and the message. It is masked out, and
        # positions continue from the preamble, so the real tokens see what they would see unpadded.
        position = prefix.length + (suffix_mask.long().cumsum(-1) - 1).clamp(min=0)
        mask = torch.cat([suffix_mask.new_ones(ids.shape[0], prefix.length), suffix_mask], dim=1)
        cache = copy.deepcopy(prefix.cache)
        cache.batch_repeat_interleave(ids.shape[0])
    eos = sorted(_eos_ids(model, tokenizer))
    eos_t = torch.tensor(eos, device=device)
    batch = ids.shape[0]

    out = model(input_ids=ids, attention_mask=mask, position_ids=position, past_key_values=cache, use_cache=True,
                output_hidden_states=True, logits_to_keep=1)
    layer_idx = layers if layers is not None else list(range(1, len(out.hidden_states)))
    # Layers can live on different GPUs, so everything is gathered on the input device.
    pick = lambda hs: torch.stack([hs[l][:, -1, :].to(device) for l in layer_idx], dim=1)
    pre = pick(out.hidden_states).float()

    tokens, logprobs, entropies, states = [], [], [], []
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    next_pos = position[:, -1:] + 1
    for step in range(max_new_tokens):
        logp = torch.log_softmax(out.logits[:, -1, :].to(device).float(), dim=-1)
        token = logp.argmax(-1)
        entropies.append(-(logp.exp() * logp).sum(-1))
        logprobs.append(logp.gather(-1, token[:, None]).squeeze(-1))
        tokens.append(token)
        done = done | torch.isin(token, eos_t)
        if step == max_new_tokens - 1 or bool(done.all()):
            break
        mask = torch.cat([mask, mask.new_ones(batch, 1)], dim=1)
        out = model(input_ids=token[:, None], attention_mask=mask, position_ids=next_pos, past_key_values=cache,
                    use_cache=True, output_hidden_states=True, logits_to_keep=1)
        next_pos = next_pos + 1
        states.append(pick(out.hidden_states).float())   # state of the token chosen at this step

    gen = torch.stack(tokens, dim=1).tolist()
    chosen = torch.stack(logprobs, dim=1)
    entropy = torch.stack(entropies, dim=1)
    eos_set = set(eos)
    answers, stats, last_rows, mean_rows = [], [], [], []
    for b in range(batch):
        n = next((i for i, t in enumerate(gen[b]) if t in eos_set), len(gen[b]))
        n_eff = max(n, 1)                     # an immediate EOS still gets one scored position
        answers.append(tokenizer.decode(gen[b][:n], skip_special_tokens=True).strip())
        lp, en = chosen[b, :n_eff], entropy[b, :n_eff]
        stats.append(dict(n_tokens=n, truncated=n == max_new_tokens, mean_logprob=lp.mean().item(), min_logprob=lp.min().item(),
                          sum_logprob=lp.sum().item(), first_entropy=en[0].item(), mean_entropy=en.mean().item(),
                          max_entropy=en.max().item()))
        usable = min(n, len(states))          # the very last token of a truncated answer has no forward pass
        if usable == 0:
            last_rows.append(pre[b])
            mean_rows.append(pre[b])
        else:
            rows = torch.stack([states[t][b] for t in range(usable)], dim=0)
            last_rows.append(rows[-1])
            mean_rows.append(rows.mean(0))
    to_np = lambda x: x.to("cpu", torch.float32).numpy()
    return Batch(answers, stats, to_np(pre), to_np(torch.stack(last_rows)), to_np(torch.stack(mean_rows)))


def split_on_oom(rows: list, run, store, log=print) -> None:
    """Runs a batch, halving it if the GPU runs out of memory, and hands each finished piece to store.

    A few very long prompts can fill the GPU after hours of normal batches. Only the batch
    composition changes, so an item's answer can differ by no more than ordinary batching noise.
    """
    failed = False
    try:
        store(rows, run(rows))
    except torch.OutOfMemoryError:
        if len(rows) == 1:
            raise
        failed = True
    if failed:   # outside the except block, so the traceback no longer pins the tensors
        torch.cuda.empty_cache()
        log(f"out of memory at batch size {len(rows)}, splitting")
        half = len(rows) // 2
        split_on_oom(rows[:half], run, store, log)
        split_on_oom(rows[half:], run, store, log)
