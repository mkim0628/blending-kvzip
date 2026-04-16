"""TTFT 정밀 측정: Blend vs Full Prefill vs KVzip Only.

다양한 문서 길이에서 TTFT를 분리 측정:
  - T_load: chunk 로드 시간
  - T_rope: RoPE 보정 시간
  - T_forward: fresh forward 시간
  - T_blend: HKVD + overwrite 시간
  - T_first_token: 첫 토큰 생성 시간
  - TTFT = T_load + T_rope + T_forward + T_blend + T_first_token

사용법:
    python run_ttft.py -m Qwen/Qwen2.5-7B-Instruct
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
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--recomp", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()

    print("=" * 70)
    print("TTFT 정밀 측정")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_ttft")

    query = "\n\nSummarize the main topic in one sentence.\nAnswer:"

    # 다양한 길이의 문서
    base_text = (
        "Artificial intelligence has transformed many industries including healthcare, "
        "finance, and transportation. Deep learning models can now diagnose diseases "
        "from medical images with accuracy surpassing human experts. In finance, "
        "AI-powered trading algorithms analyze market patterns at speeds impossible "
        "for human traders. Self-driving vehicles use computer vision. "
    )

    doc_configs = [
        ("short_500", base_text * 5),
        ("medium_1000", base_text * 10),
        ("long_2000", base_text * 20),
        ("xlong_4000", base_text * 40),
    ]

    results = []

    for doc_name, doc_text in doc_configs:
        doc_ids = model.encode(doc_text)
        doc_len = doc_ids.shape[1]
        print(f"\n{'='*70}")
        print(f"Document: {doc_name} ({doc_len} tokens)")
        print(f"{'='*70}")

        # ── 오프라인: 저장 ──
        kv = model.prefill(doc_text, do_score=True)
        kv.prune(ratio=args.ratio)
        store.save_chunk(doc_name, kv)

        # ── 방법 1: Full Prefill ──
        print(f"\n  --- Full Prefill ---")
        ttft_full_list = []
        for trial in range(args.warmup + args.repeat):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            kv_fresh = model.prefill(doc_text, do_score=False)
            query_ids = model.apply_template("Summarize the main topic in one sentence.")

            torch.cuda.synchronize()
            t_prefill = time.perf_counter()

            # 첫 토큰만 생성
            input_ids = torch.cat([kv_fresh.prefill_ids, query_ids], dim=1)
            outputs = model.model(input_ids, past_key_values=kv_fresh, use_cache=True)
            first_logit = outputs.logits[0, -1, :]
            first_token = torch.argmax(first_logit)

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            ttft = (t_end - t_start) * 1000
            t_pf = (t_prefill - t_start) * 1000
            t_ft = (t_end - t_prefill) * 1000

            if trial >= args.warmup:
                ttft_full_list.append({"ttft": ttft, "prefill": t_pf, "first_token": t_ft})
                print(f"    Trial {trial-args.warmup}: TTFT={ttft:.0f}ms (prefill={t_pf:.0f}, first_tok={t_ft:.0f})")

        avg_full = sum(r["ttft"] for r in ttft_full_list) / len(ttft_full_list)

        # ── 방법 2: KVzip + CacheBlend ──
        print(f"\n  --- KVzip + CacheBlend ---")
        ttft_blend_list = []
        for trial in range(args.warmup + args.repeat):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            kv_loaded = store.load_chunk(doc_name, device=model.device)

            torch.cuda.synchronize()
            t_load = time.perf_counter()

            # Fresh forward
            prefill_ids = kv_loaded.prefill_ids
            query_ids_t = model.encode(query)
            all_ids = torch.cat([prefill_ids, query_ids_t], dim=1)
            fresh_cache = DynamicCache()
            model.model(all_ids, past_key_values=fresh_cache, use_cache=True)

            torch.cuda.synchronize()
            t_forward = time.perf_counter()

            # HKVD + overwrite (simplified — check layer 1 only)
            n_heads_kv = kv_loaded.n_heads_kv
            cl = 1
            cu_len_k = kv_loaded.info["cu_len_k"][cl]
            valid_pad = torch.ones(1, n_heads_kv, kv_loaded.start_idx, dtype=torch.bool)
            full_valid = torch.cat([valid_pad, kv_loaded.valid[cl].cpu()], dim=-1)

            for h in range(n_heads_kv):
                kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(model.device)
                k_old_h = kv_loaded.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
                k_new_h = fresh_cache.key_cache[cl][0, h, kept_pos, :]
                diff_h = ((k_new_h - k_old_h)**2).sum(-1)
                topk_h = max(int(len(diff_h) * args.recomp), 1)
                imp_h = torch.topk(diff_h, topk_h).indices
                # overwrite (all layers)
                for l in range(kv_loaded.n_layers):
                    cu_l = kv_loaded.info["cu_len_k"][l]
                    vp_l = torch.ones(1, n_heads_kv, kv_loaded.start_idx, dtype=torch.bool)
                    fv_l = torch.cat([vp_l, kv_loaded.valid[l].cpu()], dim=-1)
                    kp_l = fv_l[0, h].nonzero(as_tuple=True)[0]
                    abs_imp = set(kept_pos[imp_h].cpu().tolist())
                    local_imp = [i for i, p in enumerate(kp_l.tolist()) if p in abs_imp]
                    if local_imp:
                        lt = torch.tensor(local_imp, dtype=torch.long)
                        ap = kp_l[lt].to(model.device)
                        lt = lt.to(model.device)
                        kv_loaded.key_cache[l][cu_l[h] + lt] = fresh_cache.key_cache[l][0, h, ap, :]
                        kv_loaded.value_cache[l][cu_l[h] + lt] = fresh_cache.value_cache[l][0, h, ap, :]

            torch.cuda.synchronize()
            t_blend = time.perf_counter()

            # 첫 토큰 생성 (KVzip pruned decode)
            query_ids_gen = model.apply_template("Summarize the main topic in one sentence.")
            input_gen = torch.cat([kv_loaded.prefill_ids, query_ids_gen], dim=1)
            outputs = model.model(input_gen, past_key_values=kv_loaded, use_cache=True)
            first_logit = outputs.logits[0, -1, :]
            first_token = torch.argmax(first_logit)

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            ttft = (t_end - t_start) * 1000
            t_ld = (t_load - t_start) * 1000
            t_fw = (t_forward - t_load) * 1000
            t_bl = (t_blend - t_forward) * 1000
            t_ft = (t_end - t_blend) * 1000

            if trial >= args.warmup:
                ttft_blend_list.append({
                    "ttft": ttft, "load": t_ld, "forward": t_fw,
                    "blend": t_bl, "first_token": t_ft
                })
                print(f"    Trial {trial-args.warmup}: TTFT={ttft:.0f}ms "
                      f"(load={t_ld:.0f}, fwd={t_fw:.0f}, blend={t_bl:.0f}, ft={t_ft:.0f})")

        avg_blend = sum(r["ttft"] for r in ttft_blend_list) / len(ttft_blend_list)
        speedup = avg_full / avg_blend if avg_blend > 0 else 0

        print(f"\n  Summary for {doc_name} ({doc_len} tokens):")
        print(f"    Full prefill TTFT:  {avg_full:.0f}ms")
        print(f"    Blend TTFT:         {avg_blend:.0f}ms")
        print(f"    Speedup:            {speedup:.2f}x")

        results.append({
            "doc": doc_name, "tokens": doc_len,
            "full_prefill_ttft_ms": round(avg_full),
            "blend_ttft_ms": round(avg_blend),
            "speedup": round(speedup, 2),
            "blend_breakdown": {
                "load": round(sum(r["load"] for r in ttft_blend_list) / len(ttft_blend_list)),
                "forward": round(sum(r["forward"] for r in ttft_blend_list) / len(ttft_blend_list)),
                "blend": round(sum(r["blend"] for r in ttft_blend_list) / len(ttft_blend_list)),
                "first_token": round(sum(r["first_token"] for r in ttft_blend_list) / len(ttft_blend_list)),
            },
        })

    # 결과 저장
    with open("ttft_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TTFT 결과 요약")
    print("=" * 70)
    print(f"\n  {'Doc':<15} {'Tokens':>7} {'Full(ms)':>9} {'Blend(ms)':>10} {'Speedup':>8}")
    print("  " + "-" * 52)
    for r in results:
        print(f"  {r['doc']:<15} {r['tokens']:>7} {r['full_prefill_ttft_ms']:>9} "
              f"{r['blend_ttft_ms']:>10} {r['speedup']:>7.2f}x")

    print(f"\n결과 저장: ttft_results.json")


if __name__ == "__main__":
    main()
