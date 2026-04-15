"""CacheBlend + KVzip 통합 데모.

3단계로 동작:
  1. 문서 prefill + KVzip scoring + prune → chunk 저장
  2. 저장된 chunk 로드 → IW-HKVD blend → generate
  3. Full prefill baseline과 비교

사용법:
    python blend_demo.py
    python blend_demo.py -m Qwen/Qwen2.5-7B-Instruct-1M
    python blend_demo.py --ratio 0.3 --recomp 0.15
"""

import argparse
import time
import torch

from model import ModelKVzip
from attention.blend import ChunkStore
from utils.func import TimeStamp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    parser.add_argument("--ratio", type=float, default=0.3, help="KVzip compression ratio")
    parser.add_argument("--recomp", type=float, default=0.15, help="HKVD recomputation ratio")
    parser.add_argument("--check-layer", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    stamp = TimeStamp(verbose=True, unit="ms")

    print("=" * 70)
    print("CacheBlend + KVzip 통합 데모")
    print(f"  Model: {args.model}")
    print(f"  KVzip ratio: {args.ratio} (keep {args.ratio*100:.0f}%)")
    print(f"  HKVD recomp: {args.recomp} (recompute {args.recomp*100:.0f}%)")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store")

    # ── Phase 1: 문서 prefill + 압축 + 저장 ──
    print("\n>>> Phase 1: Prefill + Compress + Store")
    print("-" * 70)

    documents = {
        "doc_ai": "Artificial intelligence has transformed industries including healthcare, "
                  "finance, and transportation. Deep learning models can now diagnose diseases, "
                  "predict market trends, and drive autonomous vehicles. " * 50,
        "doc_climate": "Climate change poses significant challenges to global agriculture and "
                       "food security. Rising temperatures, changing precipitation patterns, and "
                       "extreme weather events threaten crop yields worldwide. " * 50,
    }

    for doc_id, doc_text in documents.items():
        stamp(f"Before prefill '{doc_id}'")

        kv = model.prefill(doc_text, do_score=True)
        stamp(f"After prefill+scoring '{doc_id}' ({kv._seen_tokens} tokens)")

        kv.prune(ratio=args.ratio)
        stamp(f"After prune '{doc_id}' (ratio={args.ratio})")

        store.save_chunk(doc_id, kv)
        stamp(f"After save '{doc_id}'")
        print()

    # ── Phase 2: Blend + Generate ──
    print("\n>>> Phase 2: Load + Blend + Generate")
    print("-" * 70)

    chunk_ai = store.load_chunk("doc_ai")
    chunk_climate = store.load_chunk("doc_climate")
    stamp("Chunks loaded")

    queries = [
        "What are the main applications of AI?",
        "How does climate change affect agriculture?",
        "Compare AI and climate change challenges.",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        query_text = f"\n\n{q.strip()}\nAnswer without explanation."

        output = model.blend_generate(
            query_text,
            chunk_kvs=[chunk_ai],  # 단일 chunk로 먼저 검증
            recomp_ratio=args.recomp,
            check_layers=[args.check_layer],
        )
        print(f"Blend output: {output}")
        stamp(f"After blend_generate")
        print("-" * 40)

    # ── Phase 3: Baseline (full prefill) ──
    print("\n>>> Phase 3: Baseline (full prefill, no blend)")
    print("-" * 70)

    full_context = documents["doc_ai"] + " " + documents["doc_climate"]
    kv_full = model.prefill(full_context, do_score=False)
    stamp("Full prefill done")

    for q in queries:
        query_text = f"\n\n{q.strip()}\nAnswer without explanation."
        query_ids = model.apply_template(q + "\nAnswer without explanation.")
        output_baseline = model.generate(query_ids, kv=kv_full, update_cache=False)
        print(f"Query: {q}")
        print(f"Baseline: {output_baseline}")
        print("-" * 40)

    print("\n" + "=" * 70)
    print("데모 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
