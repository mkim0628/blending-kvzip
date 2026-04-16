"""Quick test for blend_generate_v2 after causal=False fix."""
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

torch.set_grad_enabled(False)

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
store = ChunkStore("./chunk_store_test")

context = """Albert Einstein was born on March 14, 1879 in Ulm, Germany. 
He developed the theory of relativity and won the Nobel Prize in Physics in 1921 
for his explanation of the photoelectric effect."""

# Step 1: Prefill + prune + save
kv = model.prefill(context, do_score=True)
kv.prune(ratio=0.3)
store.save_chunk("test_v2", kv)
print(f"Saved: {kv._seen_tokens} tokens")

# Step 2: Baseline - generate directly
kv_base = store.load_chunk("test_v2", device=model.device)
query = model.apply_template("When was Einstein born? Answer briefly.")
out_base = model.generate(query, kv=kv_base, update_cache=False)
print(f"Baseline: {out_base}")

# Step 3: blend_generate_v2
kv_v2 = store.load_chunk("test_v2", device=model.device)
out_v2 = model.blend_generate_v2(query, [kv_v2], recomp_ratio=0.3, check_layers=[1], method="iw_hkvd")
print(f"BlendV2:  {out_v2}")

# Step 4: blend_generate_v2 with r=1.0 (all tokens)
kv_v2b = store.load_chunk("test_v2", device=model.device)
out_v2b = model.blend_generate_v2(query, [kv_v2b], recomp_ratio=1.0, check_layers=[1], method="diff_only")
print(f"BlendV2 r=1.0: {out_v2b}")

print("\nDone!")
