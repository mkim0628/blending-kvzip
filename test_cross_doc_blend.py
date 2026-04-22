"""Cross-document multi-hop blend test using blend_generate_multi.

Scenario
--------
  Offline : doc_A, doc_B 각각 KVzip prefill + prune → ChunkStore 저장
  Online  : context = [doc_B + doc_A] (역순)
            KV      = [pruned_KV_B, pruned_KV_A] concat
            → blend_generate_multi 로 쿼리 응답

Baselines
---------
  1) full prefill [doc_A + doc_B]  (A+B 원순)
  2) full prefill [doc_B + doc_A]  (B+A 역순)

Sweep
-----
  prune_ratio:  [0.3, 0.5]
  recomp_ratio: [0.15, 0.30, 0.50]
  method: iw_hkvd (fixed)
"""
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

torch.set_grad_enabled(False)

# ── Documents ─────────────────────────────────────────────────────────────────
doc_A = (
    "Albert Einstein was born on March 14, 1879, in Ulm, Germany, "
    "and died on April 18, 1955, in Princeton, USA, at the age of 76. "
    "He developed the theory of relativity and won the Nobel Prize in Physics "
    "in 1921 for his explanation of the photoelectric effect. "
    "Before becoming famous he worked at the patent office in Bern, Switzerland. "
    "He emigrated to the United States in 1933 and joined the "
    "Institute for Advanced Study in Princeton, where he remained until his death."
)

doc_B = (
    "Marie Curie was born on November 7, 1867, in Warsaw, Poland, "
    "and died on July 4, 1934, in France, at the age of 66. "
    "She discovered two new elements, polonium and radium. "
    "She won the Nobel Prize in Physics in 1903 and the Nobel Prize in "
    "Chemistry in 1911, making her the only scientist to win Nobel Prizes "
    "in two different sciences. Her husband Pierre Curie died in a street "
    "accident in 1906. She founded the Curie Institute in Paris in 1920."
)

# ── Multi-hop queries ────────────────────────────────────────────────────
queries = [
    (
        # doc_A: Bern → Einstein → Nobel Physics 1921
        # doc_B: radium → Curie → Nobel Physics 1903
        "The scientist who worked at the patent office in Bern won the Nobel Prize "
        "in Physics. The scientist who discovered radium also won the Nobel Prize in "
        "Physics. How many years separated their two Nobel Prize awards?",
        "18",
    ),
    (
        # doc_B: radium → Curie → born 1867
        # doc_A: theory of relativity → Einstein → born 1879
        "The scientist who discovered radium was born in 1867. "
        "How many years later was the scientist born who developed "
        "the theory of relativity?",
        "12",
    ),
    (
        # doc_A: Einstein died at 76
        # doc_B: scientist who died at 66 → Curie → discovered polonium
        "Einstein died at the age of 76 in Princeton. "
        "What element was discovered by the scientist who died at the age of 66?",
        "polonium",
    ),
    (
        # doc_B: Curie founded institute 1920
        # doc_A: theory of relativity → Einstein → emigrated 1933
        "Curie founded an institute in Paris in 1920. "
        "How many years after that did the scientist who developed "
        "the theory of relativity emigrate to the United States?",
        "13",
    ),
    (
        # doc_A: Einstein emigrated 1933
        # doc_B: polonium → Curie → died 1934
        "Einstein emigrated to the United States in 1933. "
        "Was the scientist who discovered polonium still alive at that time?",
        "yes",
    ),
    (
        # doc_B: two sciences Nobel → Curie, born 1867
        # doc_A: born 12 years later → Einstein → worked in Bern
        "The scientist who won Nobel Prizes in two different sciences was born "
        "in Warsaw in 1867. In which city did the scientist born exactly "
        "12 years after her work before becoming famous?",
        "bern",
    ),
    (
        # doc_B: Physics+Chemistry Nobel → Curie → died 1934
        # doc_A: Bern patent office → Einstein → died 1955
        "The scientist who won Nobel Prizes in both Physics and Chemistry died "
        "in France. The scientist who worked at the patent office in Bern died "
        "in Princeton. Who died first?",
        "curie",
    ),
]

PRUNE_RATIOS  = [0.3, 0.5]
RECOMP_RATIOS = [0.15, 0.30, 0.50]
METHOD        = "iw_hkvd"
CHECK_LAYERS  = [1]

# ── Init ──────────────────────────────────────────────────────────────────────
model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct", kv_type="evict")
store = ChunkStore("./chunk_store_cross_multihop")


def run_queries(model, kv, queries, tag):
    """full prefill kv로 모든 쿼리 실행."""
    matches = 0
    print(f"\n{'─'*60}")
    print(f"[{tag}]")
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer in one sentence.")
        out = model.generate(qi, kv=kv, update_cache=False)
        hit = kw.lower() in out.lower()
        matches += hit
        print(f"  {'O' if hit else 'X'}  Q: {q[:80]}...")
        print(f"       A: {out.strip()[:100]}")
        print(f"       expected: '{kw}'")
    print(f"  → {matches}/{len(queries)} correct")
    return matches


# ── Step 1: Prefill + prune + save each doc separately ────────────────────────
print("=" * 60)
print("Step 1: Building pruned KV per document")
print("=" * 60)

for pr in PRUNE_RATIOS:
    print(f"\n  prune_ratio = {pr}")

    kv_a = model.prefill(doc_A, do_score=True)
    kv_a.prune(ratio=pr)
    store.save_chunk(f"doc_a_r{pr}", kv_a)
    kept_a = sum(kv_a.info["len_k"][0]).item()
    print(f"    doc_A: {kv_a._seen_tokens} tokens, {kept_a} kept")

    kv_b = model.prefill(doc_B, do_score=True)
    kv_b.prune(ratio=pr)
    store.save_chunk(f"doc_b_r{pr}", kv_b)
    kept_b = sum(kv_b.info["len_k"][0]).item()
    print(f"    doc_B: {kv_b._seen_tokens} tokens, {kept_b} kept")

# ── Step 2: Baseline A+B — full prefill [doc_A + doc_B] ──────────────────────
print("\n" + "=" * 60)
print("Step 2: Baseline — full prefill [doc_A + doc_B]  (A+B order)")
print("=" * 60)

kv_ab = model.prefill(doc_A + " " + doc_B, do_score=False)
bl_ab = run_queries(model, kv_ab, queries, tag="Baseline A+B")

# ── Step 3: Baseline B+A — full prefill [doc_B + doc_A] ──────────────────────
print("\n" + "=" * 60)
print("Step 3: Baseline — full prefill [doc_B + doc_A]  (B+A order)")
print("=" * 60)

kv_ba = model.prefill(doc_B + " " + doc_A, do_score=False)
bl_ba = run_queries(model, kv_ba, queries, tag="Baseline B+A")

# ── Step 4: blend_generate_multi sweep ──────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 4: blend_generate_multi  chunk_kvs=[kv_B, kv_A]")
print(f"        method={METHOD}  check_layers={CHECK_LAYERS}")
print("=" * 60)

results = {}

for pr in PRUNE_RATIOS:
    for rr in RECOMP_RATIOS:
        tag = f"prune={pr} | recomp={rr} | method={METHOD}"
        print(f"\n{'─'*60}")
        print(f"[Blend] {tag}")

        blend_matches = 0
        for q, kw in queries:
            kv_b_l = store.load_chunk(f"doc_b_r{pr}", device=model.device)
            kv_a_l = store.load_chunk(f"doc_a_r{pr}", device=model.device)

            qi = model.apply_template(q + "\nAnswer in one sentence.")

            out = model.blend_generate_multi(
                qi,
                chunk_kvs=[kv_b_l, kv_a_l],
                recomp_ratio=rr,
                check_layers=CHECK_LAYERS,
                method=METHOD,
            )

            hit = kw.lower() in out.lower()
            blend_matches += hit
            print(f"  {'O' if hit else 'X'}  Q: {q[:80]}...")
            print(f"       A: {out.strip()[:100]}")
            print(f"       expected: '{kw}'")

        results[(pr, rr)] = blend_matches
        print(f"  → {blend_matches}/{len(queries)} correct")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  Baseline A+B (full prefill): {bl_ab}/{len(queries)}")
print(f"  Baseline B+A (full prefill): {bl_ba}/{len(queries)}")
print()
print(f"  {'prune':>8} | {'recomp':>8} | {'score':>8}")
print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}")
for pr in PRUNE_RATIOS:
    for rr in RECOMP_RATIOS:
        score = results[(pr, rr)]
        print(f"  {pr:>8} | {rr:>8} | {score:>4}/{len(queries)}")

print("\nDone!")
