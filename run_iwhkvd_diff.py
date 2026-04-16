"""IW-HKVD vs Random 차이를 드러내는 실험.

조건: 긴 비반복 문서 + 극도로 낮은 r% (1~5%)
목표: "어떤 토큰을 재계산하느냐"가 결과에 직접 영향을 미치는 것을 보여줌.

사용법:
    python run_iwhkvd_diff.py -m Qwen/Qwen2.5-7B-Instruct
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


def custom_blend(model, kv, query_text, recomp_ratio, method="iw_hkvd"):
    """blend_generate의 변형 — method 파라미터 지원."""
    import time as _time
    from transformers import DynamicCache

    device = model.device
    n_layers = kv.n_layers
    n_heads_kv = kv.n_heads_kv

    query_ids = model.encode(query_text) if isinstance(query_text, str) else query_text
    prefill_ids = kv.prefill_ids
    all_ids = torch.cat([prefill_ids, query_ids], dim=1)

    fresh_cache = DynamicCache()
    model.model(all_ids, past_key_values=fresh_cache, use_cache=True)

    cl = 1  # check layer
    cu_len_k = kv.info["cu_len_k"][cl]
    valid_pad = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
    full_valid = torch.cat([valid_pad, kv.valid[cl]], dim=-1)

    imp_per_head = {}
    for h in range(n_heads_kv):
        kept_pos = full_valid[0, h].nonzero(as_tuple=True)[0].to(device)
        k_old_h = kv.key_cache[cl][cu_len_k[h]:cu_len_k[h+1]]
        k_new_h = fresh_cache.key_cache[cl][0, h, kept_pos, :]
        diff_h = ((k_new_h - k_old_h) ** 2).sum(-1)
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
        else:  # diff_only
            metric = diff_h

        imp_per_head[h] = torch.topk(metric, topk_h).indices

    # Overwrite
    imp_abs_per_head = {}
    valid_pad_cl = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
    full_valid_cl = torch.cat([valid_pad_cl, kv.valid[cl]], dim=-1)
    for h in range(n_heads_kv):
        kept_pos_cl = full_valid_cl[0, h].nonzero(as_tuple=True)[0]
        imp_abs_per_head[h] = set(kept_pos_cl[imp_per_head[h].cpu()].tolist())

    for l in range(n_layers):
        cu_len_k_l = kv.info["cu_len_k"][l]
        valid_pad_l = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
        full_valid_l = torch.cat([valid_pad_l, kv.valid[l]], dim=-1)
        for h in range(n_heads_kv):
            if h not in imp_abs_per_head:
                continue
            kept_pos_l = full_valid_l[0, h].nonzero(as_tuple=True)[0]
            abs_targets = imp_abs_per_head[h]
            local_imp = [i for i, p in enumerate(kept_pos_l.tolist()) if p in abs_targets]
            if not local_imp:
                continue
            lt = torch.tensor(local_imp, dtype=torch.long, device=device)
            ap = kept_pos_l[lt].to(device)
            kv.key_cache[l][cu_len_k_l[h] + lt] = fresh_cache.key_cache[l][0, h, ap, :]
            kv.value_cache[l][cu_len_k_l[h] + lt] = fresh_cache.value_cache[l][0, h, ap, :]

    output = model.generate(query_ids, kv=kv, update_cache=False)
    return output


@torch.inference_mode()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = args.parse_args()

    print("=" * 70)
    print("IW-HKVD vs Random 차이 실험")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_diff")

    # 긴 비반복 문서 — 각 문장이 서로 다른 구체적 사실
    long_doc = """
Albert Einstein was born in Ulm Germany in 1879. He developed the theory of special relativity in 1905 and general relativity in 1915. His famous equation E equals mc squared describes the relationship between mass and energy. He received the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect not for relativity. Einstein moved to the United States in 1933 and worked at the Institute for Advanced Study in Princeton New Jersey until his death in 1955.

Isaac Newton was born in Woolsthorpe England in 1643. He formulated the three laws of motion and the law of universal gravitation. His work Principia Mathematica published in 1687 is considered one of the most important scientific works ever written. Newton also made major contributions to optics discovering that white light is composed of a spectrum of colors. He served as the Warden and later Master of the Royal Mint from 1696 to 1727.

Marie Curie was born in Warsaw Poland in 1867. She was the first woman to win a Nobel Prize receiving the Physics prize in 1903 and the Chemistry prize in 1911. She discovered the elements polonium and radium. Her research on radioactivity a term she coined led to the development of X-ray machines used in World War One. She died in 1934 from aplastic anemia caused by radiation exposure.

Charles Darwin was born in Shrewsbury England in 1809. He is best known for his theory of evolution by natural selection published in On the Origin of Species in 1859. His five year voyage on HMS Beagle starting in 1831 provided observations that formed the basis of his theory. Darwin also studied barnacles extensively and wrote several books about plant biology. He died in 1882 and was buried in Westminster Abbey.

Nikola Tesla was born in Smiljan Croatia in 1856. He developed the alternating current AC electrical system that is widely used today. Tesla held over 300 patents including designs for the Tesla coil and the AC induction motor. He conducted experiments in wireless transmission at his laboratory in Colorado Springs in 1899. Tesla died in room 3327 of the New Yorker Hotel in Manhattan on January 7 1943 at the age of 86.
""".strip()

    # 질문 — 매우 구체적인 사실 (정확한 토큰 선택이 필요)
    queries = [
        ("In what city was Einstein born?", "Ulm"),
        ("What year did Einstein receive the Nobel Prize?", "1921"),
        ("Where did Newton work at the Royal Mint?", "Royal Mint"),
        ("What year was Principia Mathematica published?", "1687"),
        ("What elements did Marie Curie discover?", "polonium"),
        ("What year did Marie Curie win the Chemistry Nobel?", "1911"),
        ("What ship did Darwin travel on?", "Beagle"),
        ("What year was On the Origin of Species published?", "1859"),
        ("How many patents did Tesla hold?", "300"),
        ("In what room number did Tesla die?", "3327"),
    ]

    # 오프라인
    print("\n--- Offline: Prefill + Score + Prune ---")
    kv_store = model.prefill(long_doc, do_score=True)
    print(f"  Tokens: {kv_store._seen_tokens}")
    kv_store.prune(ratio=0.3)
    store.save_chunk("long_doc", kv_store)

    # Baseline
    print("\n--- Baseline ---")
    kv_full = model.prefill(long_doc, do_score=False)
    baseline = {}
    for q, kw in queries:
        qids = model.apply_template(q + "\nAnswer with just the fact, no explanation.")
        out = model.generate(qids, kv=kv_full, update_cache=False)
        match = kw.lower() in out.lower()
        baseline[q] = out
        print(f"  Q: {q[:45]:45s} → match={match} | {out[:60]}")

    # 비교 실험
    results = []
    methods = ["iw_hkvd", "diff_only", "importance_only", "random"]
    recomp_ratios = [0.01, 0.03, 0.05, 0.10, 0.15, 0.30]

    for method in methods:
        for r in recomp_ratios:
            matches = 0
            rouge_sum = 0
            for q, kw in queries:
                kv_loaded = store.load_chunk("long_doc", device=model.device)
                query_text = f"\n\n{q}\nAnswer with just the fact, no explanation."
                out = custom_blend(model, kv_loaded, query_text, recomp_ratio=r, method=method)
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
    with open("iwhkvd_diff_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 요약
    print("\n" + "=" * 70)
    print("요약: Method × Recomp → Match Rate")
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

    print(f"\n결과 저장: iwhkvd_diff_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
