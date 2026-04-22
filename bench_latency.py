"""
Latency benchmark: full prefill vs zip+blend.

Measures
--------
1. End-to-end latency
     full prefill [B+A]:  t_prefill + t_generate
     zip+blend    [B+A]:  t_online  (load + blend_generate_multi)

2. zip+blend latency breakdown
     Offline (one-time):  prefill+score, prune, save  (per doc)
     Online  (per query): load, RoPE, fresh_forward,
                          HKVD+overwrite, merge, generate

Fixed: prune_ratio=0.3, recomp_ratio=0.15, method=iw_hkvd
"""
import io
import re
import sys
import time
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

# ── helpers ────────────────────────────────────────────────────────────────────

def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


class CaptureStdout:
    """blend_generate_multi 내부 [MultiBlend] 타이밍 출력을 캐폫."""
    def __enter__(self):
        self._buf = io.StringIO()
        self._orig = sys.stdout
        sys.stdout = self._buf
        return self._buf

    def __exit__(self, *_):
        sys.stdout = self._orig


def parse_blend_breakdown(captured: str) -> dict:
    """[MultiBlend] 출력에서 ms 단위 latency 파싱."""
    patterns = {
        "rope":          r"RoPE re-rotation:\s*([\d.]+)ms",
        "fresh_forward": r"Fresh forward:\s*([\d.]+)ms",
        "hkvd_overwrite":r"HKVD\+Overwrite:\s*([\d.]+)ms",
        "generate":      r"Generate:\s*([\d.]+)ms",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, captured)
        result[key] = float(m.group(1)) if m else 0.0
    return result


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, var ** 0.5


# ── docs & queries (same as test_cross_doc_blend.py) ──────────────────────────────
doc_A = (
    "Albert Einstein was born on March 14, 1879, in Ulm, Germany, "
    "and died on April 18, 1955, in Princeton, USA, at the age of 76. "
    "He developed the theory of relativity and won the Nobel Prize in Physics "
    "in 1921 for his explanation of the photoelectric effect. "
    "Before becoming famous he worked at the patent office in Bern, Switzerland. "
    "He emigrated to the United States in 1933 and joined the "
    "Institute for Advanced Study in Princeton, where he remained until his death."
)

doc_B = (
    "Marie Curie was born on November 7, 1867, in Warsaw, Poland, "
    "and died on July 4, 1934, in France, at the age of 66. "
    "She discovered two new elements, polonium and radium. "
    "She won the Nobel Prize in Physics in 1903 and the Nobel Prize in "
    "Chemistry in 1911, making her the only scientist to win Nobel Prizes "
    "in two different sciences. Her husband Pierre Curie died in a street "
    "accident in 1906. She founded the Curie Institute in Paris in 1920."
)

queries = [
    "The scientist who worked at the patent office in Bern won the Nobel Prize "
    "in Physics. The scientist who discovered radium also won the Nobel Prize in "
    "Physics. How many years separated their two Nobel Prize awards?",

    "The scientist who discovered radium was born in 1867. "
    "How many years later was the scientist born who developed "
    "the theory of relativity?",

    "Einstein died at the age of 76 in Princeton. "
    "What element was discovered by the scientist who died at the age of 66?",

    "Curie founded an institute in Paris in 1920. "
    "How many years after that did the scientist who developed "
    "the theory of relativity emigrate to the United States?",

    "Einstein emigrated to the United States in 1933. "
    "Was the scientist who discovered polonium still alive at that time?",

    "The scientist who won Nobel Prizes in two different sciences was born "
    "in Warsaw in 1867. In which city did the scientist born exactly "
    "12 years after her work before becoming famous?",

    "The scientist who won Nobel Prizes in both Physics and Chemistry died "
    "in France. The scientist who worked at the patent office in Bern died "
    "in Princeton. Who died first?",
]

PRUNE_RATIO  = 0.3
RECOMP_RATIO = 0.15
METHOD       = "iw_hkvd"
CHECK_LAYERS = [1]

torch.set_grad_enabled(False)
model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct", kv_type="evict")
store = ChunkStore("./chunk_store_latency")

# ────────────────────────────────────────────────────────────────────────────
# OFFLINE: prefill + score + prune + save  (one-time cost per doc)
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("OFFLINE: Building pruned KV (one-time)")
print("=" * 60)

offline = {}

for doc_name, doc_text in [("doc_A", doc_A), ("doc_B", doc_B)]:
    print(f"\n  [{doc_name}]")

    t0 = sync_time()
    kv = model.prefill(doc_text, do_score=True)
    t_prefill_score = (sync_time() - t0) * 1000

    t0 = sync_time()
    kv.prune(ratio=PRUNE_RATIO)
    t_prune = (sync_time() - t0) * 1000

    t0 = sync_time()
    store.save_chunk(doc_name, kv)
    t_save = (sync_time() - t0) * 1000

    offline[doc_name] = {
        "prefill_score_ms": t_prefill_score,
        "prune_ms":         t_prune,
        "save_ms":          t_save,
        "total_ms":         t_prefill_score + t_prune + t_save,
    }
    kept = sum(kv.info["len_k"][0]).item()
    print(f"    prefill+score: {t_prefill_score:7.1f} ms")
    print(f"    prune:         {t_prune:7.1f} ms   ({kept} tokens kept)")
    print(f"    save:          {t_save:7.1f} ms")
    print(f"    subtotal:      {offline[doc_name]['total_ms']:7.1f} ms")

t_offline_total = sum(v["total_ms"] for v in offline.values())
print(f"\n  Total offline: {t_offline_total:.1f} ms")

# ────────────────────────────────────────────────────────────────────────────
# ONLINE BASELINE: full prefill [B+A] + generate
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ONLINE BASELINE: full prefill [doc_B + doc_A]")
print("=" * 60)

t0 = sync_time()
kv_ba = model.prefill(doc_B + " " + doc_A, do_score=False)
t_prefill_ba = (sync_time() - t0) * 1000
print(f"\n  prefill [B+A]: {t_prefill_ba:.1f} ms")

bl_generate_times = []
for q in queries:
    qi = model.apply_template(q + "\nAnswer in one sentence.")
    t0 = sync_time()
    _ = model.generate(qi, kv=kv_ba, update_cache=False)
    bl_generate_times.append((sync_time() - t0) * 1000)

bl_gen_mean, bl_gen_std = mean_std(bl_generate_times)
print(f"  generate (avg over {len(queries)} queries): "
      f"{bl_gen_mean:.1f} ± {bl_gen_std:.1f} ms")
print(f"  E2E (prefill + generate): {t_prefill_ba + bl_gen_mean:.1f} ms")

# ────────────────────────────────────────────────────────────────────────────
# ONLINE ZIP+BLEND: load + blend_generate_multi
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"ONLINE ZIP+BLEND: load + blend_generate_multi")
print(f"  prune={PRUNE_RATIO}  recomp={RECOMP_RATIO}  method={METHOD}")
print("=" * 60)

load_times        = []
blend_total_times = []
breakdowns = {k: [] for k in ["rope", "fresh_forward", "hkvd_overwrite", "generate"]}

for q in queries:
    t0 = sync_time()
    kv_b_l = store.load_chunk("doc_B", device=model.device)
    kv_a_l = store.load_chunk("doc_A", device=model.device)
    load_times.append((sync_time() - t0) * 1000)

    qi = model.apply_template(q + "\nAnswer in one sentence.")

    with CaptureStdout() as buf:
        t0 = sync_time()
        _ = model.blend_generate_multi(
            qi,
            chunk_kvs=[kv_b_l, kv_a_l],
            recomp_ratio=RECOMP_RATIO,
            check_layers=CHECK_LAYERS,
            method=METHOD,
        )
        blend_total_times.append((sync_time() - t0) * 1000)

    bd = parse_blend_breakdown(buf.getvalue())
    for k, v in bd.items():
        breakdowns[k].append(v)

load_mean,  load_std  = mean_std(load_times)
blend_mean, blend_std = mean_std(blend_total_times)

print(f"\n  Per-query latency (avg over {len(queries)} queries):")
print(f"    load chunks:          {load_mean:7.1f} ± {load_std:.1f} ms")
print(f"    blend_generate_multi: {blend_mean:7.1f} ± {blend_std:.1f} ms")
print(f"      ├─ RoPE re-rotation:  {mean_std(breakdowns['rope'])[0]:6.1f} ms")
print(f"      ├─ fresh forward:     {mean_std(breakdowns['fresh_forward'])[0]:6.1f} ms")
print(f"      ├─ HKVD + overwrite:  {mean_std(breakdowns['hkvd_overwrite'])[0]:6.1f} ms")
print(f"      └─ generate (decode): {mean_std(breakdowns['generate'])[0]:6.1f} ms")
print(f"    Total online:         {load_mean + blend_mean:7.1f} ms")

# ────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

e2e_baseline     = t_prefill_ba + bl_gen_mean
e2e_blend_online = load_mean + blend_mean

print(f"\n  [Offline — one-time cost]")
for doc_name, v in offline.items():
    print(f"    {doc_name}: {v['total_ms']:.1f} ms  "
          f"(prefill+score {v['prefill_score_ms']:.0f} ms  "
          f"prune {v['prune_ms']:.0f} ms  "
          f"save {v['save_ms']:.0f} ms)")
print(f"    Total: {t_offline_total:.1f} ms")

print(f"\n  [Online — per query]")
print(f"    full prefill [B+A]: {e2e_baseline:8.1f} ms  "
      f"(prefill {t_prefill_ba:.0f} ms + generate {bl_gen_mean:.0f} ms)")
print(f"    zip+blend    [B+A]: {e2e_blend_online:8.1f} ms  "
      f"(load {load_mean:.0f} ms + blend {blend_mean:.0f} ms)")
print(f"\n    Online speedup:     {e2e_baseline / e2e_blend_online:.2f}x  "
      f"(full prefill E2E / zip+blend online E2E)")
print(f"    Prefill savings:    {t_prefill_ba:.0f} ms  "
      f"(amortized over queries when cache is reused)")

print("\nDone!")
