"""Context 변경 실험 — IW-HKVD vs Random 차이를 드러냄.

핵심: 저장 시와 재사용 시 시스템 프롬프트가 다르면
모든 doc 토큰의 K가 변함 → HKVD가 실제로 의미 있음.

사용법:
    python run_context_change.py -m Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import time
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


def blend_with_new_context(model, kv, new_sys_prompt, query_text, recomp_ratio, method="iw_hkvd"):
    """저장된 KV를 다른 시스템 프롬프트 context에서 blend.

    저장 시: [old_sys + doc]
    재사용 시: [new_sys + doc + query] — sys가 다름!
    → doc의 모든 K가 변함 → HKVD가 의미 있음
    """
    device = model.device
    n_layers = kv.n_layers
    n_heads_kv = kv.n_heads_kv

    query_ids = model.encode(query_text) if isinstance(query_text, str) else query_text

    # 새 시스템 프롬프트로 전체 prompt 구성
    new_sys_ids = model.encode(new_sys_prompt)

    # 원래 prefill_ids에서 doc 부분만 추출 (sys_prompt 제거)
    old_sys_len = kv.start_idx  # sys_prompt 길이
    original_doc_ids = kv.prefill_ids[:, old_sys_len:]  # doc 토큰만

    # 새 prompt: [new_sys + doc + query]
    all_ids = torch.cat([new_sys_ids, original_doc_ids, query_ids], dim=1)
    new_sys_len = new_sys_ids.shape[1]

    print(f"  [Context] old_sys={old_sys_len}, new_sys={new_sys_len}, "
          f"doc={original_doc_ids.shape[1]}, query={query_ids.shape[1]}, "
          f"total={all_ids.shape[1]}")

    # Fresh forward (새 context로)
    fresh_cache = DynamicCache()
    model.model(all_ids, past_key_values=fresh_cache, use_cache=True)

    # RoPE 보정: sys_prompt 길이가 다르면 doc의 위치가 다름
    # 원래: doc는 position old_sys_len부터
    # 새로: doc는 position new_sys_len부터
    # offset = new_sys_len - old_sys_len
    position_offset = new_sys_len - old_sys_len

    if position_offset != 0:
        from attention.blend import reapply_rope
        rotary_emb = model.model.model.rotary_emb

        for l in range(n_layers):
            cu_len_k = kv.info["cu_len_k"][l]
            valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
            full_valid = torch.cat([valid_pad, kv.valid[l]], dim=-1)

            for h in range(n_heads_kv):
                kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
                old_positions = kept_pos
                new_positions = kept_pos + position_offset
                k_h = kv.key_cache[l][cu_len_k[h]:cu_len_k[h+1]]
                kv.key_cache[l][cu_len_k[h]:cu_len_k[h+1]] = \
                    reapply_rope(k_h, old_positions, new_positions, rotary_emb)

        print(f"  [Context] RoPE offset={position_offset}")

    # HKVD — doc 토큰만 비교 (sys_prompt는 다르므로 비교 대상 아님)
    cl = 1
    cu_len_k = kv.info["cu_len_k"][cl]
    valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
    full_valid = torch.cat([valid_pad, kv.valid[cl]], dim=-1)

    imp_per_head = {}
    total_diff = 0
    for h in range(n_heads_kv):
        kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
        # fresh_cache에서 해당 위치의 K (새 context 기준)
        fresh_pos = kept_pos + position_offset  # 새 prompt에서의 위치
        k_old_h = kv.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
        k_new_h = fresh_cache.key_cache[cl][0, h, fresh_pos, :]

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

    print(f"  [Context] total diff_k sum={total_diff:.2f} (should be >> 0 if context changed!)")

    # Overwrite all layers
    imp_abs = {}
    valid_pad_cl = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
    full_valid_cl = torch.cat([valid_pad_cl, kv.valid[cl]], dim=-1)
    for h in range(n_heads_kv):
        kept_pos_cl = full_valid_cl[0, h].nonzero(as_tuple=True)[0]
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
            fresh_pos = (kp_l[lt] + position_offset).to(device)
            kv.key_cache[l][cu_l[h] + lt] = fresh_cache.key_cache[l][0, h, fresh_pos, :]
            kv.value_cache[l][cu_l[h] + lt] = fresh_cache.value_cache[l][0, h, fresh_pos, :]

    # Generate — 새 sys_prompt + doc를 prefill_ids로 설정
    kv.prefill_ids = torch.cat([new_sys_ids, original_doc_ids], dim=1)
    kv._seen_tokens = kv.prefill_ids.shape[1]

    output = model.generate(query_ids, kv=kv, update_cache=False)
    return output


@torch.inference_mode()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = args.parse_args()

    print("=" * 70)
    print("Context 변경 실험 — IW-HKVD vs Random 차이")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_ctx")

    # 긴 비반복 문서
    long_doc = """
Albert Einstein was born in Ulm Germany in 1879. He developed the theory of special relativity in 1905 and general relativity in 1915. His famous equation E equals mc squared describes mass energy equivalence. He received the Nobel Prize in Physics in 1921 for the photoelectric effect. Einstein moved to Princeton New Jersey in 1933.

Isaac Newton was born in Woolsthorpe England in 1643. He formulated three laws of motion and universal gravitation. Principia Mathematica was published in 1687. Newton discovered white light is composed of a spectrum of colors. He served as Master of the Royal Mint from 1696.

Marie Curie was born in Warsaw Poland in 1867. She won the Physics Nobel in 1903 and Chemistry Nobel in 1911. She discovered polonium and radium. Her research on radioactivity led to X-ray machines in World War One. She died in 1934 from radiation exposure.

Charles Darwin was born in Shrewsbury England in 1809. He published On the Origin of Species in 1859 describing evolution by natural selection. His voyage on HMS Beagle starting in 1831 provided key observations. He was buried in Westminster Abbey in 1882.

Nikola Tesla was born in Smiljan Croatia in 1856. He developed the alternating current AC electrical system. Tesla held over 300 patents. He conducted wireless experiments in Colorado Springs in 1899. Tesla died in room 3327 of the New Yorker Hotel in 1943.
""".strip()

    queries = [
        ("In what city was Einstein born?", "Ulm"),
        ("What year was Principia published?", "1687"),
        ("What elements did Curie discover?", "polonium"),
        ("What ship did Darwin sail on?", "Beagle"),
        ("How many patents did Tesla hold?", "300"),
        ("What year did Einstein get the Nobel?", "1921"),
        ("Where was Newton born?", "Woolsthorpe"),
        ("When did the French scientist Curie die?", "1934"),
        ("What year was Origin of Species published?", "1859"),
        ("In what room did Tesla die?", "3327"),
    ]

    # 저장: sys_prompt = "You are a helpful assistant."
    print("\n--- Offline: Store with default sys_prompt ---")
    kv_store = model.prefill(long_doc, do_score=True)
    kv_store.prune(ratio=0.3)
    store.save_chunk("doc_scientists", kv_store)
    print(f"  Stored: {kv_store._seen_tokens} tokens")

    # 새 시스템 프롬프트 (다른 context → K가 변함)
    new_sys = "You are a strict fact-checker. Provide only exact facts with numbers and dates. Be very precise and concise."

    # Baseline: 새 sys_prompt로 full prefill
    print("\n--- Baseline: Full prefill with new sys ---")
    model.set_chat_template()  # reset
    # 수동으로 새 sys_prompt 적용
    kv_full = model.prefill(long_doc, do_score=False)
    baseline = {}
    for q, kw in queries:
        qids = model.apply_template(q)
        out = model.generate(qids, kv=kv_full, update_cache=False)
        baseline[q] = out
        match = kw.lower() in out.lower()
        print(f"  Q: {q[:45]:45s} → {kw} in output: {match}")

    # 비교: method × r%
    methods = ["iw_hkvd", "diff_only", "importance_only", "random"]
    recomp_ratios = [0.01, 0.03, 0.05, 0.10, 0.15, 0.30]

    results = []
    for method in methods:
        for r in recomp_ratios:
            matches = 0
            rouge_sum = 0
            for qi, (q, kw) in enumerate(queries):
                kv_loaded = store.load_chunk("doc_scientists", device=model.device)
                query_text = f"\n\n{q}"
                out = blend_with_new_context(
                    model, kv_loaded, new_sys, query_text,
                    recomp_ratio=r, method=method,
                )
                match = kw.lower() in out.lower()
                rouge = compute_rouge_l(out, baseline.get(q, ""))
                matches += int(match)
                rouge_sum += rouge
                results.append({
                    "method": method, "recomp": r,
                    "query": q, "keyword": kw, "output": out,
                    "match": match, "rouge_l": round(rouge, 3),
                })

            avg_match = matches / len(queries)
            avg_rouge = rouge_sum / len(queries)
            print(f"  {method:>16s} r={r:<5.2f} match={avg_match:.0%} ROUGE={avg_rouge:.2f}")

    # 저장
    with open("context_change_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 요약
    print("\n" + "=" * 70)
    print("요약")
    print("=" * 70)
    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        g[(r["method"], r["recomp"])].append(r)

    print(f"\n  {'Method':>16s}  {'r%':>5s}  {'Match':>6s}  {'ROUGE':>6s}")
    print("  " + "-" * 40)
    for (m, r) in sorted(g.keys()):
        items = g[(m, r)]
        mr = sum(1 for i in items if i["match"]) / len(items)
        rl = sum(i["rouge_l"] for i in items) / len(items)
        print(f"  {m:>16s}  {r:>5.2f}  {mr:>5.0%}  {rl:>6.2f}")

    print(f"\n결과: context_change_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
