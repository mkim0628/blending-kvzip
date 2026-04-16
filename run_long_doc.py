"""긴 문서 실험 — IW-HKVD 차이 확인.

긴 문서 2개를 [A+B]로 저장 → [B+A]로 재사용.
문서가 길수록 순서 변경 시 K 변화가 커져서 HKVD 차이가 나타날 수 있음.

사용법:
    python run_long_doc.py -m Qwen/Qwen2.5-7B-Instruct
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


def blend_reorder(model, kv, reordered_doc, query_text, recomp_ratio, method="iw_hkvd",
                   old_token_ids=None, new_token_ids=None):
    """저장된 [A+B] KV를 [B+A] context에서 blend.

    핵심: [B+A] 순서로 fresh forward 후, 토큰 ID 매핑으로 올바른 위치의 K를 비교.
    old_token_ids: [sys + A + B]의 토큰 ID (저장 시 순서)
    new_token_ids: [sys + B + A]의 토큰 ID (사용 시 순서)
    → 같은 토큰이 원래 위치 i에 있었다면, 새 순서에서 위치 j에 있음
    → fresh_cache[j]의 K를 kv_cache[i]와 비교해야 함
    """
    device = model.device
    n_layers = kv.n_layers
    n_heads_kv = kv.n_heads_kv

    query_ids = model.encode(query_text)
    sys_ids = kv.prefill_ids[:, :kv.start_idx]
    reorder_ids = model.encode(reordered_doc)
    all_ids = torch.cat([sys_ids, reorder_ids, query_ids], dim=1)

    # 위치 매핑 구축: old_pos → new_pos
    # 원래 토큰 순서 [sys + A + B]의 각 토큰이 새 순서 [sys + B + A]에서 어디에 있는지
    if old_token_ids is not None and new_token_ids is not None:
        old_ids = old_token_ids[0].tolist()
        new_ids = new_token_ids[0].tolist()
        # sys_prompt 부분은 동일 → 매핑 불필요 (위치 동일)
        # doc 부분만 매핑 필요
        sys_len = kv.start_idx
        old_doc = old_ids[sys_len:]
        new_doc = new_ids[sys_len:]

        # old_doc의 i번째 토큰이 new_doc에서 어디에 있는지 찾기
        # 토큰 ID가 중복될 수 있으므로, 순서 기반 매칭
        pos_map = {}  # old_abs_pos → new_abs_pos
        # sys_prompt: 위치 동일
        for i in range(sys_len):
            pos_map[i] = i
        # doc 부분: new_doc에서 찾기
        new_used = [False] * len(new_doc)
        for i, tok in enumerate(old_doc):
            for j, ntok in enumerate(new_doc):
                if ntok == tok and not new_used[j]:
                    pos_map[sys_len + i] = sys_len + j
                    new_used[j] = True
                    break
    else:
        # 매핑 없으면 1:1 (같은 순서)
        pos_map = {i: i for i in range(kv._seen_tokens)}

    fresh = DynamicCache()
    model.model(all_ids, past_key_values=fresh, use_cache=True)

    cl = 1
    cu = kv.info["cu_len_k"][cl]
    vp = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
    fv = torch.cat([vp, kv.valid[cl]], dim=-1)

    total_diff = 0
    for h in range(n_heads_kv):
        kept = fv[0, h].nonzero(as_tuple=True)[0].tolist()
        # 매핑된 새 위치에서 fresh K 가져오기
        new_positions = [pos_map.get(p, p) for p in kept]
        new_pos_t = torch.tensor(new_positions, device=device, dtype=torch.long)
        kept_t = torch.tensor(kept, device=device, dtype=torch.long)

        k_old = kv.key_cache[cl][cu[h]:cu[h+1]]
        k_new = fresh.key_cache[cl][0, h, new_pos_t, :]  # 매핑된 위치!
        diff = ((k_new - k_old)**2).sum(-1)
        total_diff += diff.sum().item()
        lk = k_old.shape[0]
        tk = max(int(lk * recomp_ratio), 1)

        if method == "iw_hkvd" and kv.score is not None and cl < len(kv.score):
            imp = kv.score[cl][0, h, :].to(device)
            if imp.shape[0] < lk:
                imp = torch.cat([torch.ones(lk - imp.shape[0], device=device), imp])
            elif imp.shape[0] > lk:
                imp = imp[:lk]
            metric = diff * imp
        elif method == "random":
            metric = torch.rand(lk, device=device)
        elif method == "importance_only":
            if kv.score is not None and cl < len(kv.score):
                imp = kv.score[cl][0, h, :].to(device)
                if imp.shape[0] != lk:
                    if imp.shape[0] < lk:
                        imp = torch.cat([torch.ones(lk - imp.shape[0], device=device), imp])
                    else:
                        imp = imp[:lk]
                metric = imp
            else:
                metric = diff
        else:
            metric = diff

        imp_h = torch.topk(metric, tk).indices
        imp_abs = set(fv[0, h].nonzero(as_tuple=True)[0][imp_h.cpu()].tolist())

        for l in range(n_layers):
            cu_l = kv.info["cu_len_k"][l]
            vp_l = torch.ones(1, n_heads_kv, kv.start_idx, dtype=torch.bool, device=device)
            fv_l = torch.cat([vp_l, kv.valid[l]], dim=-1)
            kp = fv_l[0, h].nonzero(as_tuple=True)[0]
            li = [i for i, p in enumerate(kp.tolist()) if p in imp_abs]
            if li:
                lt = torch.tensor(li, dtype=torch.long, device=device)
                # 매핑된 새 위치에서 fresh K/V 가져오기
                old_abs = kp[lt].tolist()
                new_abs = torch.tensor([pos_map.get(p, p) for p in old_abs], device=device, dtype=torch.long)
                kv.key_cache[l][cu_l[h] + lt] = fresh.key_cache[l][0, h, new_abs, :]
                kv.value_cache[l][cu_l[h] + lt] = fresh.value_cache[l][0, h, new_abs, :]

    output = model.generate(query_ids, kv=kv, update_cache=False)
    return output, total_diff


@torch.inference_mode()
def main():
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = args.parse_args()

    print("=" * 70)
    print("긴 문서 실험 — IW-HKVD 차이 확인")
    print("=" * 70)

    model = ModelKVzip(args.model)
    store = ChunkStore("./chunk_store_long")

    # 긴 비반복 문서 (~1000 tokens each)
    doc_A = """
Albert Einstein was born on March 14 1879 in Ulm in the Kingdom of Württemberg in the German Empire. His father Hermann Einstein was a salesman and engineer who founded Elektrotechnische Fabrik J Einstein and Cie. His mother Pauline Koch came from a wealthy family. In 1880 the family moved to Munich where Einstein's father and uncle Jakob founded a company manufacturing electrical equipment. Einstein attended the Luitpold Gymnasium where he received advanced primary and secondary education. He clashed with authorities and resented the school's teaching methods. At age 12 Einstein taught himself algebra and Euclidean geometry over a single summer. He also independently discovered his own original proof of the Pythagorean theorem. In 1894 Einstein's father's company failed and the family moved to Italy first to Milan and then to Pavia. Einstein stayed behind to finish his studies at the gymnasium. In 1895 at the age of sixteen Einstein sat the entrance examination for the Swiss Federal Polytechnic in Zurich. He failed to reach the required standard in the general part of the examination but obtained exceptional grades in physics and mathematics. Einstein finished his secondary schooling in Aarau Switzerland. In 1896 he enrolled at the Swiss Federal Polytechnic graduating in 1900 as a teacher of mathematics and physics. In 1902 Einstein was hired as an assistant examiner at the Federal Office for Intellectual Property the patent office in Bern. In 1905 Einstein published four groundbreaking papers. The first explained the photoelectric effect and earned him the Nobel Prize in Physics in 1921. The second paper explained Brownian motion. The third introduced special relativity. The fourth derived the equation E equals mc squared showing the equivalence of mass and energy. In 1915 Einstein completed his theory of general relativity which describes gravity as the warping of spacetime by mass and energy. This theory was confirmed during a total solar eclipse on May 29 1919 when Arthur Eddington observed the deflection of starlight by the Sun's gravity. In 1933 Einstein emigrated to the United States and took a position at the Institute for Advanced Study in Princeton New Jersey where he would spend the rest of his career. He became an American citizen in 1940. Einstein died on April 18 1955 at the age of 76 at Princeton Hospital. His brain was removed during autopsy and preserved for future study.
""".strip()

    doc_B = """
Marie Sklodowska Curie was born on November 7 1867 in Warsaw Poland which was then part of the Russian Empire. She was the youngest of five children of well known teachers Bronislawa and Wladyslaw Sklodowski. Her father was a mathematics and physics instructor. As a young woman she worked as a tutor and governess to earn money. In 1891 at the age of 24 she moved to Paris and enrolled at the University of Paris the Sorbonne where she studied physics and mathematics. In 1893 she was awarded a degree in physics and began working in an industrial laboratory. In 1894 she met Pierre Curie who was an instructor at the School of Physics and Chemistry. They married on July 26 1895. Marie became interested in the recent discoveries of Wilhelm Rontgen who discovered X-rays in 1895 and Henri Becquerel who discovered that uranium salts emitted rays that resembled X-rays in 1896. She began investigating uranium radiation using electrometer techniques. In 1898 she and Pierre discovered two new elements polonium named after Poland and radium named for its intense radioactivity. In 1903 Marie Pierre and Becquerel were awarded the Nobel Prize in Physics for their research on radiation phenomena. Marie was the first woman to win a Nobel Prize. Pierre died in a street accident on April 19 1906 when he was run over by a horse drawn cart. Marie took over his teaching post becoming the first female professor at the Sorbonne. In 1911 she received a second Nobel Prize this time in Chemistry for her discovery of radium and polonium making her the first person and only woman to win Nobel Prizes in two different sciences. During World War One she developed mobile radiography units nicknamed petites Curies to provide X-ray services to field hospitals. She trained over 150 women to use X-ray equipment. She founded the Curie Institute in Paris in 1920 and a second one in Warsaw in 1932. Marie Curie died on July 4 1934 at the Sancellemoz sanatorium in Passy from aplastic anemia caused by long term exposure to radiation. Her laboratory notebooks from the 1890s are still so radioactive that they are stored in lead lined boxes at the Bibliotheque nationale de France and researchers must wear protective clothing to access them.
""".strip()

    queries = [
        ("What was Einstein's father's profession?", "salesman"),
        ("At what age did Einstein teach himself algebra?", "12"),
        ("Where was the patent office where Einstein worked?", "Bern"),
        ("In what year was general relativity confirmed by eclipse?", "1919"),
        ("What happened to Einstein's brain after death?", "removed"),
        ("How old was Marie Curie when she moved to Paris?", "24"),
        ("What date did Pierre Curie die?", "April 19 1906"),
        ("How many women did Curie train to use X-ray?", "150"),
        ("In what year was the Warsaw Curie Institute founded?", "1932"),
        ("Where are Curie's notebooks stored?", "Bibliotheque"),
    ]

    combined_AB = doc_A + " " + doc_B
    combined_BA = doc_B + " " + doc_A

    for ratio in [0.1, 0.2, 0.3]:
        print(f"\n{'='*70}")
        print(f"Compression ratio: {ratio}")
        print(f"{'='*70}")

        kv = model.prefill(combined_AB, do_score=True)
        kv.prune(ratio=ratio)
        store.save_chunk(f"long_{ratio}", kv)
        surviving = sum(kv.info["len_k"][0]).item()
        print(f"Stored: {kv._seen_tokens} tokens, surviving: {surviving}")

        # Baseline: full prefill [B+A]
        kv_full = model.prefill(combined_BA, do_score=False)
        baseline = {}
        bl_matches = 0
        for q, kw in queries:
            qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
            out = model.generate(qi, kv=kv_full, update_cache=False)
            baseline[q] = out
            if kw.lower() in out.lower():
                bl_matches += 1
        print(f"Baseline [B+A]: {bl_matches}/{len(queries)} match")

        # No blend (just use stored [A+B] KV for [B+A] query)
        kv_no = store.load_chunk(f"long_{ratio}", device=model.device)
        no_matches = 0
        for q, kw in queries:
            qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
            out = model.generate(qi, kv=kv_no, update_cache=False)
            if kw.lower() in out.lower():
                no_matches += 1
        print(f"No blend (wrong order): {no_matches}/{len(queries)} match")

        # Blend with reorder
        methods = ["iw_hkvd", "diff_only", "importance_only", "random"]
        recomp_ratios = [0.05, 0.15, 0.30]

        for method in methods:
            for r in recomp_ratios:
                matches = 0
                rouge_sum = 0
                diff_sum = 0
                # 토큰 ID 매핑용
                old_tok = torch.cat([model.encode(model.tokenizer.decode(kv.prefill_ids[0][:kv.start_idx].tolist())),
                                     model.encode(combined_AB)], dim=1) if hasattr(kv, 'prefill_ids') else None
                new_tok = torch.cat([model.encode(model.tokenizer.decode(kv.prefill_ids[0][:kv.start_idx].tolist())),
                                     model.encode(combined_BA)], dim=1) if hasattr(kv, 'prefill_ids') else None

                for q, kw in queries:
                    kv_bl = store.load_chunk(f"long_{ratio}", device=model.device)
                    qt = f"\n\n{q}\nAnswer briefly with the exact fact."
                    out, diff = blend_reorder(model, kv_bl, combined_BA, qt, r, method,
                                              old_token_ids=old_tok, new_token_ids=new_tok)
                    if kw.lower() in out.lower():
                        matches += 1
                    rouge_sum += compute_rouge_l(out, baseline.get(q, ""))
                    diff_sum += diff

                avg_rouge = rouge_sum / len(queries)
                avg_diff = diff_sum / len(queries)
                print(f"  {method:>16s} r={r:<5.2f} match={matches}/{len(queries)} "
                      f"ROUGE={avg_rouge:.2f} diff={avg_diff:.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
