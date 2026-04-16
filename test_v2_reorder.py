"""Document reorder test - IW-HKVD vs random with blend_generate_v2."""
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

torch.set_grad_enabled(False)

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
store = ChunkStore("./chunk_store_reorder_v2")

doc_A = "Albert Einstein was born on March 14 1879 in Ulm Germany. He developed the theory of relativity. He won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect. He worked at the patent office in Bern before becoming famous. He emigrated to the United States in 1933 and worked at the Institute for Advanced Study in Princeton."

doc_B = "Marie Curie was born on November 7 1867 in Warsaw Poland. She discovered two new elements polonium and radium. She won the Nobel Prize in Physics in 1903 and in Chemistry in 1911. Her husband Pierre Curie died in 1906 in a street accident. She founded the Curie Institute in Paris in 1920."

queries = [
    ("When was Einstein born?", "1879"),
    ("Where did Einstein work before becoming famous?", "Bern"),
    ("What year did Einstein emigrate to the US?", "1933"),
    ("When was Marie Curie born?", "1867"),
    ("What elements did Curie discover?", "polonium"),
    ("When did Pierre Curie die?", "1906"),
]

combined_AB = doc_A + " " + doc_B
combined_BA = doc_B + " " + doc_A

for ratio in [0.15, 0.30]:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Compression ratio: {ratio}")
    print(sep)

    kv = model.prefill(combined_AB, do_score=True)
    kv.prune(ratio=ratio)
    store.save_chunk(f"reorder_{ratio}", kv)
    surviving = sum(kv.info["len_k"][0]).item()
    print(f"Stored [A+B]: {kv._seen_tokens} tokens, surviving: {surviving}")

    # Baseline: full prefill [B+A]
    kv_full = model.prefill(combined_BA, do_score=False)
    bl_matches = 0
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer briefly.")
        out = model.generate(qi, kv=kv_full, update_cache=False)
        if kw.lower() in out.lower():
            bl_matches += 1
    print(f"Baseline [B+A]: {bl_matches}/{len(queries)}")

    # No blend
    kv_no = store.load_chunk(f"reorder_{ratio}", device=model.device)
    no_matches = 0
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer briefly.")
        out = model.generate(qi, kv=kv_no, update_cache=False)
        if kw.lower() in out.lower():
            no_matches += 1
    print(f"No blend (wrong order): {no_matches}/{len(queries)}")

    # Blend with different methods
    methods = ["iw_hkvd", "diff_only", "random"]
    recomp_ratios = [0.1, 0.3, 0.5]

    for method in methods:
        for r in recomp_ratios:
            matches = 0
            for q, kw in queries:
                kv_bl = store.load_chunk(f"reorder_{ratio}", device=model.device)
                qi = model.apply_template(q + "\nAnswer briefly.")
                out = model.blend_generate_v2(qi, [kv_bl], recomp_ratio=r, check_layers=[1], method=method)
                if kw.lower() in out.lower():
                    matches += 1
            print(f"  {method:>16s} r={r:.1f} match={matches}/{len(queries)}")

print("\nDone!")
