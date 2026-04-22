"""Cross-document blend test using blend_generate_multi.

Scenario
--------
  Offline : doc_A, doc_B 각각 KVzip prefill + prune → ChunkStore 저장
  Online  : context = [doc_B + doc_A] (역순),
            KV     = [pruned_KV_B, pruned_KV_A]
            → blend_generate_multi 로 쿼리 응답

Queries
-------
  두 문서를 모두 읽어야(cross-document attention) 답할 수 있는 7개 질문.
  각 쿼리마다 정답 키워드를 포함하는지 O/X 로 평가.
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

# ── Cross-document queries (both docs required) ────────────────────────────────
# (question, answer_keyword)
queries = [
    (
        "Who was born earlier, Einstein or Curie?",
        "curie",          # Curie: 1867 / Einstein: 1879
    ),
    (
        "How many years apart were Einstein and Curie born?",
        "12",             # 1879 - 1867 = 12
    ),
    (
        "Who won the Nobel Prize in Physics first, Einstein or Curie?",
        "curie",          # Curie: 1903 / Einstein: 1921
    ),
    (
        "How many Nobel Prizes did Curie win in total compared to Einstein's one?",
        "two",            # Curie: 2 (Physics + Chemistry) / Einstein: 1
    ),
    (
        "Who lived longer, Einstein or Curie?",
        "einstein",       # Einstein: 76 years / Curie: 66 years
    ),
    (
        "Who died first, Einstein or Curie?",
        "curie",          # Curie: 1934 / Einstein: 1955
    ),
    (
        "How many years before Einstein won the Nobel Prize did Curie win her first Nobel?",
        "18",             # 1921 - 1903 = 18
    ),
]


def evaluate(model, kv, queries, tag):
    """Run all queries against kv and print results."""
    matches = 0
    print(f"\n{'─'*60}")
    print(f"[{tag}]")
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer in one sentence.")
        out = model.generate(qi, kv=kv, update_cache=False)
        hit = kw.lower() in out.lower()
        matches += hit
        mark = "O" if hit else "X"
        print(f"  {mark}  Q: {q}")
        print(f"       A: {out.strip()}")
        print(f"       expected keyword: '{kw}'")
    print(f"  → {matches}/{len(queries)} correct")
    return matches


# ── Init ──────────────────────────────────────────────────────────────────────
model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct", kv_type="evict")
store = ChunkStore("./chunk_store_cross_doc")

# ── Step 1: Prefill + prune + save each doc separately ────────────────────────
print("=" * 60)
print("Step 1: Building pruned KV per document")
print("=" * 60)

for prune_ratio in [0.3, 0.5]:
    print(f"\n-- prune_ratio = {prune_ratio} --")

    kv_a = model.prefill(doc_A, do_score=True)
    kv_a.prune(ratio=prune_ratio)
    store.save_chunk(f"doc_a_r{prune_ratio}", kv_a)
    kept_a = sum(kv_a.info["len_k"][0]).item()
    print(f"  doc_A: {kv_a._seen_tokens} tokens total, {kept_a} kept after pruning")

    kv_b = model.prefill(doc_B, do_score=True)
    kv_b.prune(ratio=prune_ratio)
    store.save_chunk(f"doc_b_r{prune_ratio}", kv_b)
    kept_b = sum(kv_b.info["len_k"][0]).item()
    print(f"  doc_B: {kv_b._seen_tokens} tokens total, {kept_b} kept after pruning")

# ── Step 2: Baseline — full prefill of [B+A] ──────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Baseline — full prefill [doc_B + doc_A]")
print("=" * 60)

kv_full = model.prefill(doc_B + " " + doc_A, do_score=False)
bl_score = evaluate(model, kv_full, queries, tag="Baseline: full prefill [B+A]")

# ── Step 3: blend_generate_multi — KV=[B,A], context=[B+A] ───────────────────
print("\n" + "=" * 60)
print("Step 3: blend_generate_multi  chunk_kvs=[kv_B, kv_A]  context=[B+A]")
print("=" * 60)

for prune_ratio in [0.3, 0.5]:
    for method in ["iw_hkvd", "diff_only", "random"]:
        for recomp_ratio in [0.15, 0.30, 0.50]:

            tag = f"prune={prune_ratio} | method={method} | recomp={recomp_ratio}"
            print(f"\n{'─'*60}")
            print(f"[Blend] {tag}")

            matches = 0
            for q, kw in queries:
                # 매 쿼리마다 새로 로드 (상태 오염 방지)
                kv_b_l = store.load_chunk(f"doc_b_r{prune_ratio}", device=model.device)
                kv_a_l = store.load_chunk(f"doc_a_r{prune_ratio}", device=model.device)

                qi = model.apply_template(q + "\nAnswer in one sentence.")

                # chunk_kvs = [kv_B, kv_A]  →  context 순서 = B + A
                out = model.blend_generate_multi(
                    qi,
                    chunk_kvs=[kv_b_l, kv_a_l],
                    recomp_ratio=recomp_ratio,
                    check_layers=[1],
                    method=method,
                )

                hit = kw.lower() in out.lower()
                matches += hit
                mark = "O" if hit else "X"
                print(f"  {mark}  Q: {q}")
                print(f"       A: {out.strip()}")
                print(f"       expected keyword: '{kw}'")

            print(f"  → {matches}/{len(queries)} correct")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
