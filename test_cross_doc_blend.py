"""Cross-document multi-hop blend test using blend_generate_multi.

Scenario
--------
  Offline : doc_A, doc_B 각각 KVzip prefill + prune(0.3) → ChunkStore 저장
  Online  : context = [doc_B + doc_A] (역순)
            KV      = [pruned_KV_B, pruned_KV_A] concat
            → blend_generate_multi 로 쿼리 응답

Queries
-------
  두 문서를 모두 거쳐야 답할 수 있는 멀티홈 추론 7개.
  단순히 "A 문서에서 찾기 + B 문서에서 찾기"가 아니라,
  한 문서의 사실을 브릿지로 삼아 다른 문서의 답을 끝어내는 구조.

Fixed params: prune_ratio=0.3, recomp_ratio=0.15, method="iw_hkvd"
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
# 각 쿼리는 두 문서를 브릿지즈럼 연결해야만 답할 수 있는 멀티홈 구조.
# (question, answer_keyword)
queries = [
    (
        # doc_A: Bern → Einstein → Nobel Physics 1921
        # doc_B: radium → Curie → Nobel Physics 1903
        # bridge: same field (Physics) → year difference
        "The scientist who worked at the patent office in Bern won the Nobel Prize "
        "in Physics. The scientist who discovered radium also won the Nobel Prize in "
        "Physics. How many years separated their two Nobel Prize awards?",
        "18",           # 1921 - 1903 = 18
    ),
    (
        # doc_B: radium → Curie → born 1867
        # doc_A: theory of relativity → Einstein → born 1879
        # bridge: birth years
        "The scientist who discovered radium was born in 1867. "
        "How many years later was the scientist born who developed "
        "the theory of relativity?",
        "12",           # 1879 - 1867 = 12
    ),
    (
        # doc_A: Einstein died at 76
        # doc_B: scientist who died at 66 → Curie → discovered polonium
        # bridge: age at death
        "Einstein died at the age of 76 in Princeton. "
        "What element was discovered by the scientist who died at the age of 66?",
        "polonium",     # Curie died at 66, discovered polonium
    ),
    (
        # doc_B: Curie founded institute 1920
        # doc_A: theory of relativity → Einstein → emigrated to US 1933
        # bridge: years between events
        "Curie founded an institute in Paris in 1920. "
        "How many years after that did the scientist who developed "
        "the theory of relativity emigrate to the United States?",
        "13",           # 1933 - 1920 = 13
    ),
    (
        # doc_A: Einstein emigrated to US in 1933
        # doc_B: polonium → Curie → died 1934
        # bridge: was Curie alive when Einstein emigrated?
        "Einstein emigrated to the United States in 1933. "
        "Was the scientist who discovered polonium still alive at that time?",
        "yes",          # Curie died 1934, so yes
    ),
    (
        # doc_B: two sciences Nobel → Curie, born Warsaw 1867
        # bridge: born 12 years after Curie (1867+12=1879) → Einstein
        # doc_A: Einstein worked in Bern before famous
        "The scientist who won Nobel Prizes in two different sciences was born "
        "in Warsaw in 1867. In which city did the scientist born exactly "
        "12 years after her work before becoming famous?",
        "bern",         # Einstein (born 1879 = 1867+12) worked in Bern
    ),
    (
        # doc_B: Physics+Chemistry Nobel → Curie → died 1934
        # doc_A: Bern patent office → Einstein → died 1955
        # bridge: who died first
        "The scientist who won Nobel Prizes in both Physics and Chemistry died "
        "in France. The scientist who worked at the patent office in Bern died "
        "in Princeton. Who died first?",
        "curie",        # Curie 1934 < Einstein 1955
    ),
]

PRUNE_RATIO  = 0.3
RECOMP_RATIO = 0.15
METHOD       = "iw_hkvd"
CHECK_LAYERS = [1]

# ── Init ──────────────────────────────────────────────────────────────────────
model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct", kv_type="evict")
store = ChunkStore("./chunk_store_cross_multihop")

# ── Step 1: Prefill + prune + save each doc separately ────────────────────────
print("=" * 60)
print("Step 1: Building pruned KV per document (prune_ratio=0.3)")
print("=" * 60)

kv_a = model.prefill(doc_A, do_score=True)
kv_a.prune(ratio=PRUNE_RATIO)
store.save_chunk("doc_a", kv_a)
kept_a = sum(kv_a.info["len_k"][0]).item()
print(f"  doc_A: {kv_a._seen_tokens} tokens total, {kept_a} kept")

kv_b = model.prefill(doc_B, do_score=True)
kv_b.prune(ratio=PRUNE_RATIO)
store.save_chunk("doc_b", kv_b)
kept_b = sum(kv_b.info["len_k"][0]).item()
print(f"  doc_B: {kv_b._seen_tokens} tokens total, {kept_b} kept")

# ── Step 2: Baseline — full prefill [B+A] ──────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Baseline — full prefill [doc_B + doc_A]")
print("=" * 60)

kv_full = model.prefill(doc_B + " " + doc_A, do_score=False)

bl_matches = 0
for q, kw in queries:
    qi = model.apply_template(q + "\nAnswer in one sentence.")
    out = model.generate(qi, kv=kv_full, update_cache=False)
    hit = kw.lower() in out.lower()
    bl_matches += hit
    print(f"  {'O' if hit else 'X'}  Q: {q[:80]}...")
    print(f"       A: {out.strip()[:100]}")
    print(f"       expected: '{kw}'")
print(f"\n  Baseline: {bl_matches}/{len(queries)}")

# ── Step 3: blend_generate_multi — KV=[B,A], context=[B+A] ───────────────────
print("\n" + "=" * 60)
print(f"Step 3: blend_generate_multi  chunk_kvs=[kv_B, kv_A]")
print(f"        method={METHOD}  recomp_ratio={RECOMP_RATIO}  check_layers={CHECK_LAYERS}")
print("=" * 60)

blend_matches = 0
for q, kw in queries:
    # 매 쿼리마다 새로 로드 (상태 오염 방지)
    kv_b_l = store.load_chunk("doc_b", device=model.device)
    kv_a_l = store.load_chunk("doc_a", device=model.device)

    qi = model.apply_template(q + "\nAnswer in one sentence.")

    # chunk_kvs = [kv_B, kv_A] → context 순서 = B + A
    out = model.blend_generate_multi(
        qi,
        chunk_kvs=[kv_b_l, kv_a_l],
        recomp_ratio=RECOMP_RATIO,
        check_layers=CHECK_LAYERS,
        method=METHOD,
    )

    hit = kw.lower() in out.lower()
    blend_matches += hit
    print(f"  {'O' if hit else 'X'}  Q: {q[:80]}...")
    print(f"       A: {out.strip()[:100]}")
    print(f"       expected: '{kw}'")

print(f"\n  Blend:    {blend_matches}/{len(queries)}")
print(f"  Baseline: {bl_matches}/{len(queries)}")
print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
