"""Multi-chunk 문서 순서 변경 실험.

오프라인: doc_A, doc_B, doc_C 각각 따로 압축 저장
온라인:   다양한 순서로 조합하여 blend + generate

비반복 텍스트 사용 → IW-HKVD vs random 차이 확인.

사용법:
    python run_multi_chunk.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
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
    args.add_argument("--ratio", type=float, default=0.3)
    args = args.parse_args()

    print("=" * 70)
    print("Multi-chunk 문서 순서 변경 실험")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_multi")

    # ── 비반복 문서 3개 ──
    documents = {
        "tech": (
            "Quantum computing uses qubits that can exist in superposition states. "
            "Unlike classical bits which are either 0 or 1, qubits can be both simultaneously. "
            "Google achieved quantum supremacy in 2019 with their Sycamore processor. "
            "IBM has developed a 1000-qubit quantum processor called Condor. "
            "Quantum error correction remains one of the biggest challenges in the field. "
            "Post-quantum cryptography is being developed to resist quantum attacks. "
            "Quantum annealing is used by D-Wave for optimization problems. "
            "Topological qubits are being researched by Microsoft for stability. "
        ),
        "history": (
            "The Roman Republic was established in 509 BC after overthrowing the monarchy. "
            "Julius Caesar crossed the Rubicon river in 49 BC starting a civil war. "
            "The Silk Road connected China to the Mediterranean for over 1500 years. "
            "The Black Death killed approximately one third of Europe's population in the 1340s. "
            "Johannes Gutenberg invented the printing press around 1440 in Mainz Germany. "
            "The Spanish Armada was defeated by England in 1588. "
            "The French Revolution began with the storming of the Bastille on July 14 1789. "
            "Napoleon Bonaparte crowned himself Emperor of France in 1804. "
        ),
        "science": (
            "DNA was first discovered by Friedrich Miescher in 1869. "
            "Watson and Crick determined the double helix structure of DNA in 1953. "
            "The speed of light in a vacuum is approximately 299792458 meters per second. "
            "Einstein published his theory of general relativity in 1915. "
            "The periodic table was first published by Dmitri Mendeleev in 1869. "
            "Penicillin was discovered by Alexander Fleming in 1928. "
            "The Higgs boson was confirmed at CERN in 2012. "
            "CRISPR gene editing technology was developed by Doudna and Charpentier. "
        ),
    }

    queries_by_docs = {
        ("tech",): [
            ("What did Google achieve in quantum computing?", "quantum supremacy"),
            ("How many qubits does IBM's Condor have?", "1000"),
        ],
        ("history",): [
            ("When did Caesar cross the Rubicon?", "49 BC"),
            ("Who invented the printing press?", "Gutenberg"),
        ],
        ("science",): [
            ("What is the speed of light?", "299792458"),
            ("Who discovered penicillin?", "Fleming"),
        ],
        ("tech", "history"): [
            ("What year did Google achieve quantum supremacy and when was the Bastille stormed?", "2019"),
            ("Compare quantum computing and the printing press in terms of revolutionary impact.", "quantum"),
        ],
        ("history", "science"): [
            ("When was DNA discovered and when did Caesar cross the Rubicon?", "1869"),
            ("What connects Mendeleev and Gutenberg?", "1869"),
        ],
        ("tech", "science"): [
            ("How do quantum computers relate to the Higgs boson discovery?", "quantum"),
        ],
    }

    # ── 오프라인: 각 문서 저장 ──
    print("\n--- Offline: Store documents ---")
    for doc_id, doc_text in documents.items():
        kv = model.prefill(doc_text, do_score=True)
        kv.prune(ratio=args.ratio)
        store.save_chunk(doc_id, kv)

    results = []

    # ── Baseline: full prefill ──
    print("\n--- Baseline: Full Prefill ---")
    baseline_outputs = {}
    for doc_combo, qas in queries_by_docs.items():
        combined_text = " ".join(documents[d] for d in doc_combo)
        kv_full = model.prefill(combined_text, do_score=False)
        for q, keyword in qas:
            qids = model.apply_template(q + "\nAnswer briefly.")
            output = model.generate(qids, kv=kv_full, update_cache=False)
            baseline_outputs[(doc_combo, q)] = output
            match = keyword.lower() in output.lower()
            print(f"  [{'+'.join(doc_combo)}] {q[:50]:50s} → match={match}")
            results.append({
                "method": "full_prefill", "docs": list(doc_combo),
                "query": q, "keyword": keyword, "output": output,
                "match": match,
            })

    # ── Multi-chunk blend 실험 ──
    for method in ["iw_hkvd", "diff_only", "random"]:
        for recomp in [0.05, 0.15, 0.30]:
            print(f"\n--- Blend: method={method}, recomp={recomp} ---")

            for doc_combo, qas in queries_by_docs.items():
                if len(doc_combo) == 1:
                    # 단일 chunk — 기존 blend_generate 사용
                    for q, keyword in qas:
                        kv_loaded = store.load_chunk(doc_combo[0], device=model.device)
                        query_text = f"\n\n{q}\nAnswer briefly."
                        output = model.blend_generate(
                            query_text, [kv_loaded],
                            recomp_ratio=recomp,
                            check_layers=[1],
                            position_offset=0,
                        )
                        bl_output = baseline_outputs.get((doc_combo, q), "")
                        rouge = compute_rouge_l(output, bl_output)
                        match = keyword.lower() in output.lower()
                        print(f"  [{'+'.join(doc_combo)}] r={recomp} {method}: {q[:40]:40s} → match={match} ROUGE={rouge:.2f}")
                        results.append({
                            "method": method, "recomp": recomp,
                            "docs": list(doc_combo),
                            "query": q, "keyword": keyword, "output": output,
                            "match": match, "rouge_l": round(rouge, 3),
                        })
                else:
                    # Multi-chunk
                    for q, keyword in qas:
                        loaded_chunks = [store.load_chunk(d, device=model.device) for d in doc_combo]
                        query_text = f"\n\n{q}\nAnswer briefly."
                        try:
                            output = model.blend_generate_multi(
                                query_text, loaded_chunks,
                                recomp_ratio=recomp,
                                check_layers=[1],
                                method=method,
                            )
                        except Exception as e:
                            output = f"[ERROR: {e}]"
                            print(f"  ERROR: {e}")

                        bl_output = baseline_outputs.get((doc_combo, q), "")
                        rouge = compute_rouge_l(output, bl_output)
                        match = keyword.lower() in output.lower()
                        print(f"  [{'+'.join(doc_combo)}] r={recomp} {method}: {q[:40]:40s} → match={match} ROUGE={rouge:.2f}")
                        results.append({
                            "method": method, "recomp": recomp,
                            "docs": list(doc_combo),
                            "query": q, "keyword": keyword, "output": output,
                            "match": match, "rouge_l": round(rouge, 3),
                        })

    # ── 순서 변경 실험 ──
    print("\n--- Document Reorder Experiment ---")
    reorder_combos = [
        (("tech", "history"), ("history", "tech")),
        (("history", "science"), ("science", "history")),
    ]
    for original, reordered in reorder_combos:
        for q, keyword in queries_by_docs.get(original, []):
            for order_name, combo in [("original", original), ("reordered", reordered)]:
                loaded = [store.load_chunk(d, device=model.device) for d in combo]
                query_text = f"\n\n{q}\nAnswer briefly."
                try:
                    output = model.blend_generate_multi(
                        query_text, loaded,
                        recomp_ratio=0.15,
                        check_layers=[1],
                        method="iw_hkvd",
                    )
                except Exception as e:
                    output = f"[ERROR: {e}]"

                match = keyword.lower() in output.lower()
                print(f"  [{'+'.join(combo)}] ({order_name}): {q[:40]:40s} → match={match}")
                results.append({
                    "method": "iw_hkvd", "recomp": 0.15,
                    "docs": list(combo), "order": order_name,
                    "query": q, "keyword": keyword, "output": output,
                    "match": match,
                })

    # ── 결과 저장 ──
    with open("multi_chunk_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 요약 ──
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        key = (r.get("method", ""), r.get("recomp", ""))
        g[key].append(r)

    for key in sorted(g.keys()):
        items = g[key]
        match_rate = sum(1 for i in items if i.get("match")) / len(items) if items else 0
        avg_rouge = sum(i.get("rouge_l", 0) for i in items) / len(items) if items else 0
        print(f"  {str(key):>35s}  match={match_rate:.0%}  rouge={avg_rouge:.2f}  n={len(items)}")

    print(f"\n결과 저장: multi_chunk_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
