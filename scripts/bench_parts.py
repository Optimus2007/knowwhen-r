"""Times model components on MPS to find what scales with batch size."""
import time
import torch
from transformers import DynamicCache
from knowwhen.models.runner import load_model

def timed(fn, n=3):
    fn(); torch.mps.synchronize(); t=time.time()
    for _ in range(n): fn()
    torch.mps.synchronize(); return (time.time()-t)/n

model, tok = load_model("CohereLabs/tiny-aya-global", dtype=torch.float16)
print('attn', model.config._attn_implementation, 'tied', model.config.tie_word_embeddings, type(model.lm_head.weight), model.lm_head.weight.dtype, model.lm_head.weight.device, flush=True)
print('devices', {str(p.device) for p in model.parameters()}, {str(p.dtype) for p in model.parameters()}, flush=True)
H=model.config.hidden_size
with torch.inference_mode():
    for B in (1,8,16):
        x=torch.randn(B,1,H,device='mps',dtype=torch.float16)
        print(f'B={B} lm_head {timed(lambda: model.lm_head(x)):.4f}', flush=True)
        ids=torch.randint(5,1000,(B,32),device='mps')
        print(f'B={B} body prefill32 {timed(lambda: model.model(input_ids=ids)):.4f}', flush=True)
        c=DynamicCache(config=model.config); o=model.model(input_ids=ids,past_key_values=c,use_cache=True)
        nxt=torch.randint(5,1000,(B,1),device='mps')
        def step():
            return model.model(input_ids=nxt,past_key_values=c,use_cache=True)
        print(f'B={B} body decode {timed(step,n=4):.4f}', flush=True)
        layer=model.model.layers[0]
        print(f'B={B} mlp only x36 {36*timed(lambda: layer.mlp(x)):.4f}', flush=True)
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
    B=8; ids=torch.randint(5,1000,(B,32),device='mps'); c=DynamicCache(config=model.config); model.model(input_ids=ids,past_key_values=c,use_cache=True)
    nxt=torch.randint(5,1000,(B,1),device='mps')
    model.model(input_ids=nxt,past_key_values=c,use_cache=True); torch.mps.synchronize()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=40))
