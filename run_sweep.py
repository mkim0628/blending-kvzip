"""Sweep 실험: compression ratio × recompute ratio 조합.

다양한 (compression ratio, recomp ratio) 조합에서 품질과 시간을 측정.
논문 Pareto curve 데이터 생성.

사용법:
    python run_sweep.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
import json
import torch
from model import ModelKVzip
from attention.blend import ChunkStore
from transformers import DynamicCache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    print("=" * 70)
    print("Sweep 실험: Compression Ratio × Recompute Ratio")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_sweep")

    # 테스트 데이터
    doc_text = (
        "Artificial intelligence has transformed many industries including healthcare, "
        "finance, and transportation. Deep learning models can now diagnose diseases "
        "from medical images with accuracy surpassing human experts. In finance, "
        "AI-powered trading algorithms analyze market patterns at speeds impossible "
        "for human traders. Self-driving vehicles use computer vision and reinforcement "
        "learning to navigate complex traffic scenarios. Natural language processing "
        "enables machines to understand and generate human language, powering virtual "
        "assistants and translation services. "
    ) * 30

    queries = [
        ("What industries has AI transformed?",
         "healthcare, finance, and transportation"),
        ("How is AI used in healthcare?",
         "diagnosing diseases from medical images"),
        ("What role does AI play in finance?",
         "trading algorithms that analyze market patterns"),
    ]

    # Sweep 파라미터
    compression_ratios = [0.2, 0.3, 0.5, 0.7]  # pruned only (comp=1.0 dense는 별도 처리)
    recomp_ratios = [0.0, 0.05, 0.10, 0.15, 0.30, 0.50]

    results = []

    # ── Baseline: Full prefill (무압축, 무blend) ──
    print("\n--- Baseline: Full Prefill ---")
    kv_full = model.prefill(doc_text, do_score=False)
    baseline_outputs = {}
    for q, expected in queries:
        query_ids = model.apply_template(q + "\nAnswer in one sentence.")
        t0 = time.perf_counter()
        output = model.generate(query_ids, kv=kv_full, update_cache=False)
        t_gen = time.perf_counter() - t0
        baseline_outputs[q] = output
        print(f"  Q: {q} → {output[:80]}... ({t_gen*1000:.0f}ms)")
        results.append({
            "comp_ratio": 1.0, "recomp_ratio": "baseline",
            "query": q, "output": output, "time_ms": round(t_gen*1000),
            "match": any(kw in output.lower() for _, kw_list in [(q, expected)] for kw in [expected]),
        })

    # ── Sweep ──
    for comp_r in compression_ratios:
        print(f"\n{'='*70}")
        print(f"Compression ratio: {comp_r}")
        print(f"{'='*70}")

        # Prefill + prune (또는 무압축)
        kv = model.prefill(doc_text, do_score=True)
        if comp_r < 1.0:
            kv.prune(ratio=comp_r)

        # KVzip only (blend 없이)
        print(f"\n  --- KVzip only (ratio={comp_r}) ---")
        for q, expected in queries:
            query_ids = model.apply_template(q + "\nAnswer in one sentence.")
            t0 = time.perf_counter()
            output = model.generate(query_ids, kv=kv, update_cache=False)
            t_gen = time.perf_counter() - t0
            match = expected.lower() in output.lower()
            print(f"    Q: {q} → {output[:60]}... (match={match}, {t_gen*1000:.0f}ms)")
            results.append({
                "comp_ratio": comp_r, "recomp_ratio": "kvzip_only",
                "query": q, "output": output, "time_ms": round(t_gen*1000),
                "match": match,
            })

        # 저장 + 로드 + blend sweep
        chunk_id = f"doc_comp{int(comp_r*100)}"
        store.save_chunk(chunk_id, kv)

        for recomp_r in recomp_ratios:
            print(f"\n  --- Blend (comp={comp_r}, recomp={recomp_r}) ---")
            kv_loaded = store.load_chunk(chunk_id, device=model.device)

            for q, expected in queries:
                query_text = f"\n\n{q}\nAnswer in one sentence."
                t0 = time.perf_counter()
                output = model.blend_generate(
                    query_text,
                    chunk_kvs=[kv_loaded],
                    recomp_ratio=recomp_r,
                    check_layers=[1],
                )
                t_total = time.perf_counter() - t0
                match = expected.lower() in output.lower()
                print(f"    Q: {q} → {output[:60]}... (match={match}, {t_total*1000:.0f}ms)")
                results.append({
                    "comp_ratio": comp_r, "recomp_ratio": recomp_r,
                    "query": q, "output": output, "time_ms": round(t_total*1000),
                    "match": match,
                })

                # 매번 fresh load (이전 blend가 kv를 수정하므로)
                kv_loaded = store.load_chunk(chunk_id, device=model.device)

    # ── 결과 저장 (먼저!) ──
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: sweep_results.json ({len(results)} entries)")

    # ── 결과 요약 ──
    print("\n" + "=" * 70)
    print("결과 요약 — Match Rate")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (str(r["comp_ratio"]), str(r["recomp_ratio"]))
        grouped[key].append(r)

    print(f"\n  Comp     Recomp  Match   Avg ms")
    print("  " + "-" * 34)
    for key in sorted(grouped.keys()):
        items = grouped[key]
        match_rate = sum(1 for i in items if i["match"]) / len(items)
        avg_time = sum(i["time_ms"] for i in items) / len(items)
        print(f"  {key[0]:>5} {key[1]:>10} {match_rate:>5.0%} {avg_time:>8.0f}")


if __name__ == "__main__":
    main()
