"""종합 실험: KVzip only vs KVzip+CacheBlend vs Full prefill baseline.

실험 1: 원래 KVzip (prefill → prune → generate)
실험 2: KVzip + CacheBlend (오프라인 압축 → 온라인 blend → generate)
실험 3: Full prefill baseline (압축/blend 없이)

사용법:
    python run_experiments.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
import json
import torch
from model import ModelKVzip
from attention.blend import ChunkStore, iw_hkvd
from transformers import DynamicCache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--recomp", type=float, default=0.15)
    parser.add_argument("--check-layer", type=int, default=1)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    results = []

    print("=" * 70)
    print("종합 실험: KVzip vs KVzip+CacheBlend vs Full Prefill")
    print(f"Model: {args.model}, KVzip ratio: {args.ratio}, HKVD recomp: {args.recomp}")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_exp")

    # ── 테스트 데이터 ──
    doc_text = (
        "Artificial intelligence has transformed many industries including healthcare, "
        "finance, and transportation. Deep learning models can now diagnose diseases "
        "from medical images with accuracy surpassing human experts. In finance, "
        "AI-powered trading algorithms analyze market patterns at speeds impossible "
        "for human traders. Self-driving vehicles use computer vision and reinforcement "
        "learning to navigate complex traffic scenarios. Natural language processing "
        "enables machines to understand and generate human language, powering virtual "
        "assistants and translation services. "
    ) * 30  # ~30 repetitions for sufficient context

    queries = [
        "What industries has AI transformed?",
        "How is AI used in healthcare?",
        "What role does AI play in finance?",
    ]

    # ══════════════════════════════════════════════════════════════
    # 실험 1: 원래 KVzip (prefill → prune → generate)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("실험 1: 원래 KVzip (prefill → prune → generate)")
    print("=" * 70)

    t0 = time.perf_counter()
    kv_orig = model.prefill(doc_text, do_score=True)
    t_prefill = time.perf_counter() - t0

    t0 = time.perf_counter()
    kv_orig.prune(ratio=args.ratio)
    t_prune = time.perf_counter() - t0

    print(f"  Prefill+scoring: {t_prefill*1000:.0f}ms, Prune: {t_prune*1000:.0f}ms")
    print(f"  Tokens: {kv_orig._seen_tokens}, Memory: {kv_orig._mem()} GB")

    for q in queries:
        query_text = f"\n\n{q}\nAnswer in one sentence without explanation."
        query_ids = model.apply_template(q + "\nAnswer in one sentence without explanation.")

        t0 = time.perf_counter()
        output = model.generate(query_ids, kv=kv_orig, update_cache=False)
        t_gen = time.perf_counter() - t0

        print(f"\n  Q: {q}")
        print(f"  A: {output}")
        print(f"  Time: {t_gen*1000:.0f}ms")

        results.append({
            "experiment": "kvzip_only",
            "query": q,
            "output": output,
            "time_ms": round(t_gen * 1000),
        })

    # ══════════════════════════════════════════════════════════════
    # 실험 2: KVzip + CacheBlend (오프라인 저장 → 온라인 blend)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("실험 2: KVzip + CacheBlend (오프라인 저장 → 온라인 blend)")
    print("=" * 70)

    # 오프라인: prefill + prune + 저장 (EvictCache 통째로)
    kv_save = model.prefill(doc_text, do_score=True)
    kv_save.prune(ratio=args.ratio)
    store.save_chunk("doc_exp", kv_save)

    for q in queries:
        query_text = f"\n\n{q}\nAnswer in one sentence without explanation."

        # 온라인: 로드 + blend + generate
        kv_loaded = store.load_chunk("doc_exp", device=model.device)

        t0 = time.perf_counter()
        output = model.blend_generate(
            query_text,
            chunk_kvs=[kv_loaded],
            recomp_ratio=args.recomp,
            check_layers=[args.check_layer],
        )
        t_total = time.perf_counter() - t0

        print(f"\n  Q: {q}")
        print(f"  A: {output}")
        print(f"  Total time: {t_total*1000:.0f}ms")

        results.append({
            "experiment": "kvzip_cacheblend",
            "query": q,
            "output": output,
            "time_ms": round(t_total * 1000),
        })

    # ══════════════════════════════════════════════════════════════
    # 실험 3: Full prefill baseline (압축/blend 없이)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("실험 3: Full prefill baseline")
    print("=" * 70)

    t0 = time.perf_counter()
    kv_full = model.prefill(doc_text, do_score=False)
    t_full_prefill = time.perf_counter() - t0
    print(f"  Full prefill: {t_full_prefill*1000:.0f}ms, Tokens: {kv_full._seen_tokens}")

    for q in queries:
        query_ids = model.apply_template(q + "\nAnswer in one sentence without explanation.")

        t0 = time.perf_counter()
        output = model.generate(query_ids, kv=kv_full, update_cache=False)
        t_gen = time.perf_counter() - t0

        print(f"\n  Q: {q}")
        print(f"  A: {output}")
        print(f"  Time: {t_gen*1000:.0f}ms")

        results.append({
            "experiment": "full_prefill",
            "query": q,
            "output": output,
            "time_ms": round(t_gen * 1000),
        })

    # ══════════════════════════════════════════════════════════════
    # 결과 요약
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    for exp_name in ["kvzip_only", "kvzip_cacheblend", "full_prefill"]:
        exp_results = [r for r in results if r["experiment"] == exp_name]
        avg_time = sum(r["time_ms"] for r in exp_results) / len(exp_results)
        print(f"\n[{exp_name}] 평균 시간: {avg_time:.0f}ms")
        for r in exp_results:
            print(f"  Q: {r['query']}")
            print(f"  A: {r['output'][:120]}")

    # 결과 저장
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: experiment_results.json")


if __name__ == "__main__":
    main()
