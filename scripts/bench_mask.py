"""Tests whether attention mask construction is what makes padded batches slow on MPS."""
import time, torch
from transformers import DynamicCache
from knowwhen.models.runner import load_model
model, tok = load_model("CohereLabs/tiny-aya-global")
B,T=16,32
ids=torch.randint(5,1000,(B,T),device='mps'); pad=torch.ones(B,T,dtype=torch.long,device='mps'); pad[:,:4]=0
pos=(pad.cumsum(-1)-1).clamp(min=0)
def bench(name, prefill_mask, step_mask):
    with torch.inference_mode():
        c=DynamicCache(config=model.config); torch.mps.synchronize(); t=time.time()
        model(input_ids=ids,attention_mask=prefill_mask(),position_ids=pos,past_key_values=c,use_cache=True,logits_to_keep=1); torch.mps.synchronize(); tp=time.time()-t
        nxt=ids[:,:1]; t=time.time()
        for k in range(4):
            model(input_ids=nxt,attention_mask=step_mask(k),position_ids=pos[:,-1:]+k+1,past_key_values=c,use_cache=True,logits_to_keep=1)
        torch.mps.synchronize(); print(f'{name}: prefill {tp:.2f}s step {(time.time()-t)/4:.3f}s',flush=True)
ones=lambda k: torch.cat([pad,pad.new_ones(B,k+1)],1)
bench('warm',lambda:None,lambda k:None)
bench('no mask',lambda:None,lambda k:None)
bench('2D padded mask',lambda:pad,ones)
def m4_pre():
    causal=torch.ones(T,T,dtype=torch.bool,device='mps').tril()
    m=causal[None,None]&pad.bool()[:,None,None,:]
    m=m|torch.eye(T,dtype=torch.bool,device='mps')[None,None]
    return {'full_attention':m,'sliding_attention':m}
def m4_step(k):
    m=ones(k).bool()[:,None,None,:]
    return {'full_attention':m,'sliding_attention':m}
bench('precomputed 4D dict',m4_pre,m4_step)
