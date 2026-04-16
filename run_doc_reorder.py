"""문서 순서 변경 실험 — 실제 RAG 시나리오.

시나리오:
  오프라인: doc_A, doc_B 각각 따로 저장 (sys + doc_A), (sys + doc_B)
  온라인 1: [sys + doc_A + doc_B + query] — A가 먼저
  온라인 2: [sys + doc_B + doc_A + query] — B가 먼저 (순서 변경!)

핵심: doc_B가 [sys + doc_B]로 저장됐지만 [sys + doc_A + doc_B]로 사용되면
      doc_B의 K는 doc_A를 추가로 보게 되어 변함! (causal attention)

사용법:
    python run_doc_reorder.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import torch
from model import ModelKVzip
from attention.blend import ChunkStore


def compute_rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / m if m > 0 else 0
    r = lcs / n if n > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@torch.inference_mode()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    args.add_argument("--ratio", type=float, default=0.5)
    args = args.parse_args()

    print("=" * 70)
    print("문서 순서 변경 실험 — 실제 RAG 시나리오")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_reorder2")

    # 비반복 문서 2개 — 서로 다른 주제, 구체적 사실
    doc_A = (
        "Albert Einstein was born in Ulm Germany in 1879. "
        "He developed the theory of special relativity in 1905. "
        "His equation E equals mc squared describes mass energy equivalence. "
        "He received the Nobel Prize in Physics in 1921 for the photoelectric effect. "
        "Einstein moved to Princeton New Jersey in 1933 and worked at the Institute for Advanced Study. "
        "He became a US citizen in 1940. He died on April 18 1955 at the age of 76. "
        "Einstein published over 300 scientific papers during his career. "
        "His theory of general relativity was confirmed during a solar eclipse in 1919. "
    )

    doc_B = (
        "Marie Curie was born in Warsaw Poland in 1867. "
        "She won the Physics Nobel Prize in 1903 together with her husband Pierre Curie. "
        "She discovered the elements polonium and radium through her research on radioactivity. "
        "She won a second Nobel Prize in Chemistry in 1911 making her the first person to win Nobel Prizes in two different sciences. "
        "Curie founded the Curie Institutes in Paris and Warsaw which remain major research centers. "
        "She developed mobile X-ray units called petites Curies used in World War One. "
        "Marie Curie died on July 4 1934 from aplastic anemia caused by radiation exposure. "
        "Her notebooks are still radioactive and stored in lead-lined boxes at the Bibliotheque nationale de France. "
    )

    queries = [
        # doc_A 관련
        ("In what city was Einstein born?", "Ulm"),
        ("What year did Einstein get the Nobel Prize?", "1921"),
        ("When did Einstein become a US citizen?", "1940"),
        ("How many scientific papers did Einstein publish?", "300"),
        # doc_B 관련
        ("In what city was Marie Curie born?", "Warsaw"),
        ("What elements did Curie discover?", "polonium"),
        ("What year did Curie win the Chemistry Nobel?", "1911"),
        ("What are petites Curies?", "X-ray"),
        # 두 문서 모두 필요
        ("Compare the birth years of Einstein and Curie.", "1879"),
        ("Who won more Nobel Prizes, Einstein or Curie?", "Curie"),
    ]

    # ── 오프라인: 각각 저장 ──
    print("\n--- Offline: Store doc_A and doc_B separately ---")
    kv_A = model.prefill(doc_A, do_score=True)
    kv_A.prune(ratio=args.ratio)
    store.save_chunk("doc_A", kv_A)
    print(f"  doc_A: {kv_A._seen_tokens} tokens")

    kv_B = model.prefill(doc_B, do_score=True)
    kv_B.prune(ratio=args.ratio)
    store.save_chunk("doc_B", kv_B)
    print(f"  doc_B: {kv_B._seen_tokens} tokens")

    results = []

    # ── Baseline 1: full prefill [sys + doc_A + doc_B + query] ──
    print("\n--- Baseline: Full prefill [doc_A + doc_B] ---")
    kv_full_AB = model.prefill(doc_A + " " + doc_B, do_score=False)
    baseline_AB = {}
    for q, kw in queries:
        qids = model.apply_template(q + "\nAnswer briefly with the exact fact.")
        out = model.generate(qids, kv=kv_full_AB, update_cache=False)
        match = kw.lower() in out.lower()
        baseline_AB[q] = out
        print(f"  [A+B] {q[:50]:50s} → match={match}")
        results.append({"method": "full_AB", "order": "A+B", "query": q, "keyword": kw,
                        "output": out, "match": match})

    # ── Baseline 2: full prefill [sys + doc_B + doc_A + query] ──
    print("\n--- Baseline: Full prefill [doc_B + doc_A] ---")
    kv_full_BA = model.prefill(doc_B + " " + doc_A, do_score=False)
    baseline_BA = {}
    for q, kw in queries:
        qids = model.apply_template(q + "\nAnswer briefly with the exact fact.")
        out = model.generate(qids, kv=kv_full_BA, update_cache=False)
        match = kw.lower() in out.lower()
        baseline_BA[q] = out
        print(f"  [B+A] {q[:50]:50s} → match={match}")
        results.append({"method": "full_BA", "order": "B+A", "query": q, "keyword": kw,
                        "output": out, "match": match})

    # ── Blend 실험: method × r% × 순서 ──
    methods = ["iw_hkvd", "diff_only", "importance_only", "random"]
    recomp_ratios = [0.03, 0.05, 0.10, 0.15, 0.30]
    orders = [("A+B", ["doc_A", "doc_B"]), ("B+A", ["doc_B", "doc_A"])]

    for method in methods:
        for r in recomp_ratios:
            for order_name, doc_order in orders:
                matches = 0
                for q, kw in queries:
                    chunks = [store.load_chunk(d, device=model.device) for d in doc_order]
                    query_text = f"\n\n{q}\nAnswer briefly with the exact fact."
                    try:
                        out = model.blend_generate_multi(
                            query_text, chunks,
                            recomp_ratio=r,
                            check_layers=[1],
                            method=method,
                        )
                    except Exception as e:
                        out = f"[ERROR: {e}]"
                    bl = baseline_AB if order_name == "A+B" else baseline_BA
                    match = kw.lower() in out.lower()
                    rouge = compute_rouge_l(out, bl.get(q, ""))
                    matches += int(match)
                    results.append({
                        "method": method, "recomp": r, "order": order_name,
                        "query": q, "keyword": kw, "output": out,
                        "match": match, "rouge_l": round(rouge, 3),
                    })

                avg_match = matches / len(queries)
                print(f"  {method:>16s} r={r:<5.2f} [{order_name}] match={avg_match:.0%}")

    # ── 저장 + 요약 ──
    with open("doc_reorder_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("요약: Method × r% × Order → Match Rate")
    print("=" * 70)
    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        g[(r.get("method",""), r.get("recomp",""), r.get("order",""))].append(r)

    print(f"\n  {'Method':>16s}  {'r%':>5s}  {'Order':>5s}  {'Match':>6s}  {'ROUGE':>6s}")
    print("  " + "-" * 48)
    for key in sorted(g.keys()):
        items = g[key]
        mr = sum(1 for i in items if i.get("match")) / len(items) if items else 0
        rl = sum(i.get("rouge_l", 0) for i in items) / len(items) if items else 0
        m, r, o = key
        print(f"  {m:>16s}  {str(r):>5s}  {o:>5s}  {mr:>5.0%}  {rl:>6.2f}")

    print(f"\n결과: doc_reorder_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
