"""KVzip 기본 기능 검증 — 짧은 입력으로 prefill → score → prune → generate."""
import torch
from model import ModelKVzip

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
print(f"Model loaded, KV type: {model.kv_type}")

# 짧은 context
context = "AI transforms healthcare by diagnosing diseases from medical images. " * 30
print(f"Context tokens: {model.encode(context).shape[1]}")

# prefill + scoring + prune
kv = model.prefill(context, do_score=True)
print(f"After prefill: {kv._seen_tokens} tokens, mem={kv._mem()} GB")

kv.prune(ratio=0.3)
print(f"After prune: mem={kv._mem()} GB, pruned={kv.pruned}")

# generate
queries = ["What does AI do in healthcare?", "How are diseases diagnosed?"]
for q in queries:
    query_ids = model.apply_template(q)
    output = model.generate(query_ids, kv=kv, update_cache=False)
    print(f"Q: {q}")
    print(f"A: {output}")
    print("---")

print("KVzip basic test PASSED!")
