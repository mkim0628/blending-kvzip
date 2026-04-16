"""IW-HKVD Ablation + ROUGE-L 평가 + TTFT 측정.

실험:
  A: topk(diff_k)              — CacheBlend 원본 (importance 없이)
  B: topk(diff_k × importance) — IW-HKVD (우리 방법)
  C: topk(importance)          — importance만
  D: random                    — random baseline
  E: KVzip only (blend 없이)   — 같은 세션 generate
  F: Full prefill baseline

다양한 문서 + ROUGE-L + TTFT 측정 포함.

사용법:
    python run_ablation.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
import json
import torch
import numpy as np
from model import ModelKVzip
from attention.blend import ChunkStore
from transformers import DynamicCache


def compute_rouge_l(prediction: str, reference: str) -> float:
    """간단한 ROUGE-L (토큰 기반 LCS)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    precision = lcs / m if m > 0 else 0
    recall = lcs / n if n > 0 else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--recomp", type=float, default=0.15)
    return parser.parse_args()


def custom_blend_generate(model, kv, query_text, recomp_ratio, method="iw_hkvd",
                           check_layers=None, position_offset=0):
    """blend_generate의 변형 — 다양한 토큰 선택 방법 지원.

    method: "iw_hkvd" | "diff_only" | "importance_only" | "random"
    """
    import time as _time
    check_layers = check_layers or [1]
    device = model.device
    n_layers = kv.n_layers
    n_heads_kv = kv.n_heads_kv
    is_pruned = getattr(kv, 'pruned', False)

    query_ids = model.encode(query_text) if isinstance(query_text, str) else query_text
    prefill_ids = kv.prefill_ids
    all_ids = torch.cat([prefill_ids, query_ids], dim=1)
    context_len = prefill_ids.shape[1]

    # RoPE re-rotation
    if position_offset > 0 and is_pruned:
        from attention.blend import reapply_rope
        rotary_emb = model.model.model.rotary_emb
        for l in range(n_layers):
            cu_len_k = kv.info["cu_len_k"][l]
            valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
            full_valid = torch.cat([valid_pad, kv.valid[l].cpu()], dim=-1)
            for h in range(n_heads_kv):
                kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0]
                old_positions = kept_pos.to(device)
                new_positions = (kept_pos + position_offset).to(device)
                k_h = kv.key_cache[l][cu_len_k[h]:cu_len_k[h+1]]
                kv.key_cache[l][cu_len_k[h]:cu_len_k[h+1]] = \
                    reapply_rope(k_h, old_positions, new_positions, rotary_emb)

    # Fresh forward
    fresh_cache = DynamicCache()
    t0 = _time.perf_counter()
    model.model(all_ids, past_key_values=fresh_cache, use_cache=True)
    t_forward = _time.perf_counter() - t0

    if not is_pruned:
        # Dense — 간단 처리
        t0 = _time.perf_counter()
        output = model.generate(query_ids, kv=kv, update_cache=False)
        t_gen = _time.perf_counter() - t0
        return output, t_forward, t_gen

    # IW-HKVD variants
    imp_per_head = {}
    for cl in check_layers:
        info = kv.info
        cu_len_k = info["cu_len_k"][cl]
        valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
        full_valid = torch.cat([valid_pad, kv.valid[cl].cpu()], dim=-1)

        for h in range(n_heads_kv):
            kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
            k_old_h = kv.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
            k_new_h = fresh_cache.key_cache[cl][0, h, kept_pos, :]
            len_k_h = k_old_h.shape[0]
            topk_h = max(int(len_k_h * recomp_ratio), 1)

            diff_h = ((k_new_h - k_old_h) ** 2).sum(-1)

            if method == "iw_hkvd":
                if kv.score is not None and cl < len(kv.score):
                    imp_score_h = kv.score[cl][0, h, :].to(device)
                    if imp_score_h.shape[0] < diff_h.shape[0]:
                        pad = torch.ones(diff_h.shape[0] - imp_score_h.shape[0], device=device)
                        imp_score_h = torch.cat([pad, imp_score_h])
                    elif imp_score_h.shape[0] > diff_h.shape[0]:
                        imp_score_h = imp_score_h[:diff_h.shape[0]]
                    metric = diff_h * imp_score_h
                else:
                    metric = diff_h
            elif method == "diff_only":
                metric = diff_h
            elif method == "importance_only":
                if kv.score is not None and cl < len(kv.score):
                    imp_score_h = kv.score[cl][0, h, :].to(device)
                    if imp_score_h.shape[0] != len_k_h:
                        if imp_score_h.shape[0] < len_k_h:
                            pad = torch.ones(len_k_h - imp_score_h.shape[0], device=device)
                            imp_score_h = torch.cat([pad, imp_score_h])
                        else:
                            imp_score_h = imp_score_h[:len_k_h]
                    metric = imp_score_h
                else:
                    metric = diff_h
            elif method == "random":
                metric = torch.rand(len_k_h, device=device)
            else:
                metric = diff_h

            imp_per_head[h] = torch.topk(metric, topk_h).indices

    # Overwrite
    t0 = _time.perf_counter()
    imp_abs_per_head = {}
    for cl in check_layers:
        valid_pad_cl = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
        full_valid_cl = torch.cat([valid_pad_cl, kv.valid[cl].cpu()], dim=-1)
        for h in range(n_heads_kv):
            kept_pos_cl = full_valid_cl[0, h].nonzero(as_tuple=True)[0]
            imp_abs_per_head[h] = set(kept_pos_cl[imp_per_head[h].cpu()].tolist())

    for l in range(n_layers):
        cu_len_k = kv.info["cu_len_k"][l]
        valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool)
        full_valid = torch.cat([valid_pad, kv.valid[l].cpu()], dim=-1)
        for h in range(n_heads_kv):
            if h not in imp_abs_per_head:
                continue
            kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
            abs_targets = imp_abs_per_head[h]
            local_imp = [i for i, p in enumerate(kept_pos.tolist()) if p in abs_targets]
            if not local_imp:
                continue
            local_imp_t = torch.tensor(local_imp, device=device, dtype=torch.long)
            abs_positions = kept_pos[local_imp_t]
            kv.key_cache[l][cu_len_k[h] + local_imp_t] = \
                fresh_cache.key_cache[l][0, h, abs_positions, :]
            kv.value_cache[l][cu_len_k[h] + local_imp_t] = \
                fresh_cache.value_cache[l][0, h, abs_positions, :]

    t_blend = _time.perf_counter() - t0

    # Generate
    t0 = _time.perf_counter()
    output = model.generate(query_ids, kv=kv, update_cache=False)
    t_gen = _time.perf_counter() - t0

    return output, t_forward + t_blend, t_gen


@torch.inference_mode()
def main():
    args = parse_args()

    print("=" * 70)
    print("IW-HKVD Ablation + ROUGE-L + TTFT")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_ablation")

    # 다양한 문서
    documents = {
        "ai": (
            "Artificial intelligence transforms healthcare by diagnosing diseases "
            "from medical images. In finance, AI trading algorithms analyze market "
            "patterns. Self-driving cars use computer vision. NLP powers virtual "
            "assistants and translation. "
        ) * 20,
        "climate": (
            "Climate change causes rising temperatures and extreme weather events. "
            "Glaciers are melting and sea levels are rising. Renewable energy sources "
            "like solar and wind power can help reduce carbon emissions. Forests act "
            "as carbon sinks absorbing CO2 from the atmosphere. "
        ) * 20,
        "history": (
            "The Roman Empire was one of the largest empires in history. Julius Caesar "
            "was assassinated in 44 BC. The Renaissance began in Italy in the 14th "
            "century. The Industrial Revolution started in Britain in the 18th century. "
        ) * 20,
    }

    qa_pairs = {
        "ai": [
            ("What does AI do in healthcare?", "AI diagnoses diseases from medical images"),
            ("How does AI help finance?", "AI trading algorithms analyze market patterns"),
            ("What powers virtual assistants?", "NLP powers virtual assistants"),
        ],
        "climate": [
            ("What causes rising temperatures?", "Climate change causes rising temperatures"),
            ("What can reduce carbon emissions?", "Renewable energy like solar and wind"),
            ("What absorbs CO2?", "Forests act as carbon sinks absorbing CO2"),
        ],
        "history": [
            ("When was Caesar assassinated?", "Julius Caesar was assassinated in 44 BC"),
            ("Where did the Renaissance begin?", "The Renaissance began in Italy"),
            ("Where did the Industrial Revolution start?", "Industrial Revolution started in Britain"),
        ],
    }

    methods = ["iw_hkvd", "diff_only", "importance_only", "random"]
    results = []

    # ── 오프라인: 문서 저장 ──
    print("\n--- Offline: Prefill + Prune + Save ---")
    for doc_id, doc_text in documents.items():
        kv = model.prefill(doc_text, do_score=True)
        kv.prune(ratio=args.ratio)
        store.save_chunk(doc_id, kv)
        print(f"  Saved '{doc_id}': {kv._seen_tokens} tokens")

    # ── Baseline: Full prefill ──
    print("\n--- Baseline: Full Prefill ---")
    baseline_outputs = {}
    for doc_id, doc_text in documents.items():
        kv_full = model.prefill(doc_text, do_score=False)
        for q, ref in qa_pairs[doc_id]:
            qids = model.apply_template(q + "\nAnswer in one sentence.")
            t0 = time.perf_counter()
            output = model.generate(qids, kv=kv_full, update_cache=False)
            t_gen = time.perf_counter() - t0
            baseline_outputs[(doc_id, q)] = output
            rouge = compute_rouge_l(output, ref)
            print(f"  [{doc_id}] Q: {q}")
            print(f"    A: {output[:80]}  ROUGE-L: {rouge:.2f}")
            results.append({
                "method": "full_prefill", "doc": doc_id, "query": q,
                "output": output, "reference": ref,
                "rouge_l": round(rouge, 3),
                "blend_ms": 0, "gen_ms": round(t_gen*1000),
            })

    # ── KVzip only ──
    print("\n--- KVzip Only ---")
    for doc_id in documents:
        kv_orig = model.prefill(documents[doc_id], do_score=True)
        kv_orig.prune(ratio=args.ratio)
        for q, ref in qa_pairs[doc_id]:
            qids = model.apply_template(q + "\nAnswer in one sentence.")
            t0 = time.perf_counter()
            output = model.generate(qids, kv=kv_orig, update_cache=False)
            t_gen = time.perf_counter() - t0
            rouge = compute_rouge_l(output, ref)
            print(f"  [{doc_id}] Q: {q}")
            print(f"    A: {output[:80]}  ROUGE-L: {rouge:.2f}")
            results.append({
                "method": "kvzip_only", "doc": doc_id, "query": q,
                "output": output, "reference": ref,
                "rouge_l": round(rouge, 3),
                "blend_ms": 0, "gen_ms": round(t_gen*1000),
            })

    # ── Ablation: 4 methods ──
    for method in methods:
        print(f"\n--- Method: {method} ---")
        for doc_id in documents:
            for q, ref in qa_pairs[doc_id]:
                kv_loaded = store.load_chunk(doc_id, device=model.device)
                query_text = f"\n\n{q}\nAnswer in one sentence."
                output, t_blend, t_gen = custom_blend_generate(
                    model, kv_loaded, query_text,
                    recomp_ratio=args.recomp, method=method,
                )
                rouge = compute_rouge_l(output, ref)
                bl_ref = baseline_outputs.get((doc_id, q), "")
                rouge_vs_bl = compute_rouge_l(output, bl_ref)
                print(f"  [{doc_id}] Q: {q}")
                print(f"    A: {output[:80]}  ROUGE-L(ref): {rouge:.2f}  ROUGE-L(baseline): {rouge_vs_bl:.2f}")
                results.append({
                    "method": method, "doc": doc_id, "query": q,
                    "output": output, "reference": ref,
                    "rouge_l": round(rouge, 3),
                    "rouge_l_vs_baseline": round(rouge_vs_bl, 3),
                    "blend_ms": round(t_blend*1000),
                    "gen_ms": round(t_gen*1000),
                })

    # ── 결과 저장 ──
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 요약 ──
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["method"]].append(r)

    print(f"\n  {'Method':<20} {'ROUGE-L':>8} {'Blend ms':>10} {'Gen ms':>8} {'Count':>6}")
    print("  " + "-" * 55)
    for method in ["full_prefill", "kvzip_only"] + methods:
        items = grouped[method]
        if not items:
            continue
        avg_rouge = np.mean([r["rouge_l"] for r in items])
        avg_blend = np.mean([r["blend_ms"] for r in items])
        avg_gen = np.mean([r["gen_ms"] for r in items])
        print(f"  {method:<20} {avg_rouge:>8.3f} {avg_blend:>10.0f} {avg_gen:>8.0f} {len(items):>6}")

    print(f"\n결과 저장: ablation_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
