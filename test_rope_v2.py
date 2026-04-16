"""Test RoPE re-rotation in blend_generate_v2"""
import torch
torch.set_grad_enabled(False)
from model import ModelKVzip
from attention.blend import ChunkStore

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
store = ChunkStore("./chunk_store_rope_v2")

docA = """Albert Einstein was born on March 14 1879 in Ulm in the Kingdom of Wuerttemberg in the German Empire. His father Hermann Einstein was a salesman and engineer. At age 12 Einstein taught himself algebra and Euclidean geometry. In 1905 Einstein published four groundbreaking papers including special relativity. He won the Nobel Prize in Physics in 1921."""

docB = """Alan Mathison Turing was born on 23 June 1912 in Maida Vale London England. In 1936 he published On Computable Numbers introducing the Turing machine concept. During World War Two he designed the Bombe machine to break Enigma codes. In 1950 he proposed the Turing test. Alan Turing died on 7 June 1954."""

queries = [
    ("What was Einstein father profession?", "salesman"),
    ("What year did Einstein publish four papers?", "1905"),
    ("When was Turing born?", "1912"),
    ("What machine did Turing design?", "Bombe"),
    ("When did Turing die?", "1954"),
    ("What test did Turing propose?", "Turing test"),
]

# Store [docA + docB], query with [docB + docA] (reordered)
stored = docA + " " + docB  # [Einstein + Turing]
reordered = docB + " " + docA  # [Turing + Einstein]

print("=" * 70)
print("Document Reorder: [docA+docB] stored → [docB+docA] queried")
print("=" * 70)

for ratio in [0.15, 0.10]:
    kv = model.prefill(stored, do_score=True)
    kv.prune(ratio=ratio)
    store.save_chunk(f"rope_{ratio}", kv)
    surviving = sum(kv.info["len_k"][0]).item()
    print(f"\n--- Retention={ratio}, surviving={surviving} ---")

    # 1. Full prefill oracle [docB+docA]
    kv_oracle = model.prefill(reordered, do_score=False)
    oracle = 0
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer briefly.")
        out = model.generate(qi, kv=kv_oracle, update_cache=False)
        if kw.lower() in out.lower(): oracle += 1
    print(f"Full prefill (oracle):     {oracle}/{len(queries)}")

    # 2. Blend WITHOUT rope_rerotation (current behavior)
    for method in ["iw_hkvd"]:
        correct = 0
        for q, kw in queries:
            kv_bl = store.load_chunk(f"rope_{ratio}", device=model.device)
            qi = model.apply_template(q + "\nAnswer briefly.")
            out = model.blend_generate_v2(
                qi, [kv_bl], recomp_ratio=0.10, check_layers=[1],
                method=method, context=reordered, rope_rerotation=False)
            if kw.lower() in out.lower(): correct += 1
        print(f"Blend (no re-rotation):    {correct}/{len(queries)}")

    # 3. Blend WITH rope_rerotation (new!)
    for method in ["iw_hkvd"]:
        correct = 0
        for q, kw in queries:
            kv_bl = store.load_chunk(f"rope_{ratio}", device=model.device)
            qi = model.apply_template(q + "\nAnswer briefly.")
            out = model.blend_generate_v2(
                qi, [kv_bl], recomp_ratio=0.10, check_layers=[1],
                method=method, context=reordered, rope_rerotation=True)
            if kw.lower() in out.lower(): correct += 1
        print(f"Blend (WITH re-rotation):  {correct}/{len(queries)}")

    # 4. No blend baseline
    correct = 0
    for q, kw in queries:
        kv_nb = store.load_chunk(f"rope_{ratio}", device=model.device)
        qi = model.apply_template(q + "\nAnswer briefly.")
        out = model.generate(qi, kv=kv_nb, update_cache=False)
        if kw.lower() in out.lower(): correct += 1
    print(f"No blend:                  {correct}/{len(queries)}")

print("\nDone!")
