"""문서 순서 변경 실험 — RoPE 위치 보정 검증.

시나리오:
  1. doc_A를 오프라인에서 압축 저장
  2. 온라인에서 [padding + doc_A + query] 순서로 배치 (doc_A의 위치가 바뀜)
  3. RoPE 보정 없이 blend → 결과 확인
  4. RoPE 보정 있이 blend → 결과 확인
  5. 비교

사용법:
    python test_reorder.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
import torch
from model import ModelKVzip
from attention.blend import ChunkStore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--recomp", type=float, default=0.15)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    print("=" * 70)
    print("문서 순서 변경 실험 — RoPE 위치 보정 검증")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_reorder")

    # ── 테스트 데이터 ──
    doc_text = (
        "The capital of France is Paris. The Eiffel Tower is located in Paris. "
        "France is known for its cuisine, wine, and fashion. "
        "The French Revolution began in 1789. Napoleon Bonaparte was a French emperor. "
    ) * 20

    queries = [
        "What is the capital of France?",
        "Where is the Eiffel Tower?",
        "When did the French Revolution begin?",
    ]

    # ── Step 1: 오프라인 — prefill + prune + 저장 ──
    print("\n--- Step 1: Offline — Prefill + Prune + Save ---")
    kv_save = model.prefill(doc_text, do_score=True)
    print(f"  Prefill done: {kv_save._seen_tokens} tokens")
    kv_save.prune(ratio=args.ratio)
    store.save_chunk("doc_france", kv_save)

    # ── Step 2: Baseline — 같은 위치에서 blend (offset=0) ──
    print("\n--- Step 2: Blend at ORIGINAL position (offset=0) ---")
    for q in queries:
        kv_loaded = store.load_chunk("doc_france", device=model.device)
        query_text = f"\n\n{q}\nAnswer in one sentence."
        output = model.blend_generate(
            query_text, chunk_kvs=[kv_loaded],
            recomp_ratio=args.recomp, check_layers=[1],
            position_offset=0,
        )
        print(f"  Q: {q}")
        print(f"  A: {output}")

    # ── Step 3: 순서 변경 — RoPE 보정 없이 (offset=500, 보정 없음) ──
    print("\n--- Step 3: Blend at SHIFTED position (offset=500, NO RoPE fix) ---")
    for q in queries:
        kv_loaded = store.load_chunk("doc_france", device=model.device)
        query_text = f"\n\n{q}\nAnswer in one sentence."
        # position_offset=0으로 호출하지만 fresh forward에는 padding 포함
        # → RoPE 불일치 상태
        output = model.blend_generate(
            query_text, chunk_kvs=[kv_loaded],
            recomp_ratio=args.recomp, check_layers=[1],
            position_offset=0,  # 보정 안 함!
        )
        print(f"  Q: {q}")
        print(f"  A (no fix): {output}")

    # ── Step 4: 순서 변경 — RoPE 보정 있이 (offset=500) ──
    print("\n--- Step 4: Blend at SHIFTED position (offset=500, WITH RoPE fix) ---")
    for q in queries:
        kv_loaded = store.load_chunk("doc_france", device=model.device)
        query_text = f"\n\n{q}\nAnswer in one sentence."
        output = model.blend_generate(
            query_text, chunk_kvs=[kv_loaded],
            recomp_ratio=args.recomp, check_layers=[1],
            position_offset=500,  # RoPE 보정 적용!
        )
        print(f"  Q: {q}")
        print(f"  A (with fix): {output}")

    # ── Step 5: Full prefill baseline ──
    print("\n--- Step 5: Full prefill baseline ---")
    kv_full = model.prefill(doc_text, do_score=False)
    for q in queries:
        query_ids = model.apply_template(q + "\nAnswer in one sentence.")
        output = model.generate(query_ids, kv=kv_full, update_cache=False)
        print(f"  Q: {q}")
        print(f"  A (baseline): {output}")

    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
