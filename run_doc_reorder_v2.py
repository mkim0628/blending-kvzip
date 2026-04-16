"""문서 순서 변경 실험 v2 — 단일 chunk 기반.

접근: 두 문서를 하나의 prompt로 prefill → 하나의 chunk으로 저장.
순서 변경: 다른 순서의 prompt로 fresh forward → HKVD → blend.

이렇게 하면 multi-chunk merge 문제를 우회하면서
"문서 순서가 바뀌면 K가 변한다" 효과를 테스트.

사용법:
    python run_doc_reorder_v2.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import torch
from model import ModelKVzip
from attention.blend import ChunkStore
from transformers import DynamicCache


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


def blend_with_reorder(model, kv, reordered_text, query_text, recomp_ratio, method="iw_hkvd"):
    """저장된 KV를 다른 순서의 context에서 blend.

    kv: [sys + doc_A + doc_B]로 저장된 EvictCache
    reordered_text: [doc_B + doc_A] (순서 바뀜)
    → fresh forward는 [sys + doc_B + doc_A + query]로 실행
    → diff_k가 0이 아님! (순서 변경으로 K가 변함)
    """
    device = model.device
    n_layers = kv.n_layers
    n_heads_kv = kv.n_heads_kv

    query_ids = model.encode(query_text) if isinstance(query_text, str) else query_text

    # 새 순서로 전체 prompt 구성
    reordered_ids = model.encode(reordered_text)
    sys_ids = kv.prefill_ids[:, :kv.start_idx]  # sys_prompt
    all_ids = torch.cat([sys_ids, reordered_ids, query_ids], dim=1)

    # Fresh forward
    fresh_cache = DynamicCache()
    model.model(all_ids, past_key_values=fresh_cache, use_cache=True)

    # HKVD
    cl = 1
    cu_len_k = kv.info["cu_len_k"][cl]
    valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
    full_valid = torch.cat([valid_pad, kv.valid[cl]], dim=-1)

    imp_per_head = {}
    total_diff = 0
    for h in range(n_heads_kv):
        kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
        k_old_h = kv.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
        k_new_h = fresh_cache.key_cache[cl][0, h, kept_pos, :]
        diff_h = ((k_new_h - k_old_h) ** 2).sum(-1)
        total_diff += diff_h.sum().item()
        len_k_h = k_old_h.shape[0]
        topk_h = max(int(len_k_h * recomp_ratio), 1)

        if method == "iw_hkvd" and kv.score is not None and cl < len(kv.score):
            imp_score_h = kv.score[cl][0, h, :].to(device)
            if imp_score_h.shape[0] < len_k_h:
                pad = torch.ones(len_k_h - imp_score_h.shape[0], device=device)
                imp_score_h = torch.cat([pad, imp_score_h])
            elif imp_score_h.shape[0] > len_k_h:
                imp_score_h = imp_score_h[:len_k_h]
            metric = diff_h * imp_score_h
        elif method == "random":
            metric = torch.rand(len_k_h, device=device)
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
        else:
            metric = diff_h

        imp_per_head[h] = torch.topk(metric, topk_h).indices

    # Overwrite
    imp_abs = {}
    for h in range(n_heads_kv):
        kept_pos_cl = full_valid[0, h].nonzero(as_tuple=True)[0]
        imp_abs[h] = set(kept_pos_cl[imp_per_head[h].cpu()].tolist())

    for l in range(n_layers):
        cu_l = kv.info["cu_len_k"][l]
        vp_l = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
        fv_l = torch.cat([vp_l, kv.valid[l]], dim=-1)
        for h in range(n_heads_kv):
            kp_l = fv_l[0, h].nonzero(as_tuple=True)[0]
            local_imp = [i for i, p in enumerate(kp_l.tolist()) if p in imp_abs[h]]
            if not local_imp:
                continue
            lt = torch.tensor(local_imp, dtype=torch.long, device=device)
            ap = kp_l[lt].to(device)
            kv.key_cache[l][cu_l[h] + lt] = fresh_cache.key_cache[l][0, h, ap, :]
            kv.value_cache[l][cu_l[h] + lt] = fresh_cache.value_cache[l][0, h, ap, :]

    output = model.generate(query_ids, kv=kv, update_cache=False)
    return output, total_diff


@torch.inference_mode()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    args.add_argument("--ratio", type=float, default=0.5)
    args = args.parse_args()

    print("=" * 70)
    print("문서 순서 변경 실험 v2 — 단일 chunk 기반")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_reorder_v2")

    doc_A = (
        "Albert Einstein was born in Ulm Germany in 1879. "
        "He developed the theory of special relativity in 1905. "
        "His equation E equals mc squared describes mass energy equivalence. "
        "He received the Nobel Prize in Physics in 1921 for the photoelectric effect. "
        "Einstein moved to Princeton New Jersey in 1933. "
        "He became a US citizen in 1940. He died on April 18 1955. "
        "Einstein published over 300 scientific papers. "
        "His general relativity was confirmed during a solar eclipse in 1919. "
    )

    doc_B = (
        "Marie Curie was born in Warsaw Poland in 1867. "
        "She won the Physics Nobel in 1903 with her husband Pierre. "
        "She discovered polonium and radium. "
        "She won the Chemistry Nobel in 1911 becoming the first person with two Nobel Prizes in different sciences. "
        "Curie founded the Curie Institutes in Paris and Warsaw. "
        "She developed mobile X-ray units called petites Curies for World War One. "
        "Marie Curie died on July 4 1934 from aplastic anemia. "
        "Her notebooks are still radioactive stored in lead-lined boxes. "
    )

    queries = [
        ("In what city was Einstein born?", "Ulm"),
        ("What year did Einstein get the Nobel?", "1921"),
        ("How many papers did Einstein publish?", "300"),
        ("Where was Marie Curie born?", "Warsaw"),
        ("What elements did Curie discover?", "polonium"),
        ("What year did Curie win the Chemistry Nobel?", "1911"),
        ("Compare birth years of Einstein and Curie.", "1879"),
        ("Who won more Nobel Prizes?", "Curie"),
    ]

    # ── 저장: [sys + doc_A + doc_B] 순서로 ──
    print("\n--- Offline: Store [doc_A + doc_B] ---")
    combined_AB = doc_A + " " + doc_B
    kv_AB = model.prefill(combined_AB, do_score=True)
    kv_AB.prune(ratio=args.ratio)
    store.save_chunk("AB", kv_AB)
    print(f"  Stored: {kv_AB._seen_tokens} tokens")

    # ── Baseline: [sys + doc_A + doc_B + query] (same order) ──
    print("\n--- Baseline: Full prefill [A+B] ---")
    kv_full = model.prefill(combined_AB, do_score=False)
    baseline = {}
    for q, kw in queries:
        qids = model.apply_template(q + "\nAnswer briefly.")
        out = model.generate(qids, kv=kv_full, update_cache=False)
        baseline[q] = out
        print(f"  {q[:45]:45s} → {kw} in output: {kw.lower() in out.lower()}")

    # ── Baseline: [sys + doc_B + doc_A + query] (reordered) ──
    print("\n--- Baseline: Full prefill [B+A] ---")
    combined_BA = doc_B + " " + doc_A
    kv_full_BA = model.prefill(combined_BA, do_score=False)
    baseline_BA = {}
    for q, kw in queries:
        qids = model.apply_template(q + "\nAnswer briefly.")
        out = model.generate(qids, kv=kv_full_BA, update_cache=False)
        baseline_BA[q] = out
        print(f"  {q[:45]:45s} → {kw} in output: {kw.lower() in out.lower()}")

    results = []
    methods = ["iw_hkvd", "diff_only", "importance_only", "random"]
    recomp_ratios = [0.01, 0.03, 0.05, 0.10, 0.15, 0.30]

    # ── 같은 순서로 blend (diff_k ≈ 0 확인) ──
    print("\n--- Blend: Same order [A+B] (diff_k should be ~0) ---")
    for method in methods[:1]:
        for r in [0.15]:
            kv_loaded = store.load_chunk("AB", device=model.device)
            out, diff = blend_with_reorder(model, kv_loaded, doc_A + " " + doc_B,
                                           "\n\nIn what city was Einstein born?\nAnswer briefly.",
                                           r, method)
            print(f"  {method} r={r}: diff_k={diff:.0f}, output: {out[:60]}")

    # ── 순서 변경 blend (diff_k >> 0 기대) ──
    print("\n--- Blend: Reordered [B+A] (diff_k should be >> 0) ---")
    for method in methods:
        for r in recomp_ratios:
            matches = 0
            rouge_sum = 0
            total_diff_sum = 0
            for q, kw in queries:
                kv_loaded = store.load_chunk("AB", device=model.device)
                query_text = f"\n\n{q}\nAnswer briefly."
                out, diff = blend_with_reorder(
                    model, kv_loaded, combined_BA, query_text, r, method)
                match = kw.lower() in out.lower()
                rouge = compute_rouge_l(out, baseline_BA.get(q, ""))
                matches += int(match)
                rouge_sum += rouge
                total_diff_sum += diff
                results.append({
                    "method": method, "recomp": r, "order": "B+A",
                    "query": q, "keyword": kw, "output": out,
                    "match": match, "rouge_l": round(rouge, 3), "diff_k": round(diff),
                })

            avg_match = matches / len(queries)
            avg_rouge = rouge_sum / len(queries)
            avg_diff = total_diff_sum / len(queries)
            print(f"  {method:>16s} r={r:<5.2f} match={avg_match:.0%} ROUGE={avg_rouge:.2f} diff={avg_diff:.0f}")

    # 저장
    with open("doc_reorder_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n결과: doc_reorder_v2_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
