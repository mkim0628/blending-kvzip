"""Harder test: extreme compression + document reorder with blend_generate_v2."""
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

torch.set_grad_enabled(False)

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
store = ChunkStore("./chunk_store_hard_v2")

# Longer, denser documents (~500 tokens each)
doc_A = """Albert Einstein was born on March 14 1879 in Ulm in the Kingdom of Wuerttemberg in the German Empire. His father Hermann Einstein was a salesman and engineer who founded Elektrotechnische Fabrik J Einstein and Cie. His mother Pauline Koch came from a wealthy family. In 1880 the family moved to Munich where Einsteins father and uncle Jakob founded a company manufacturing electrical equipment. Einstein attended the Luitpold Gymnasium where he received advanced primary and secondary education. At age 12 Einstein taught himself algebra and Euclidean geometry over a single summer. He also independently discovered his own original proof of the Pythagorean theorem. In 1894 Einsteins fathers company failed and the family moved to Italy first to Milan and then to Pavia. In 1895 at the age of sixteen Einstein sat the entrance examination for the Swiss Federal Polytechnic in Zurich. He failed to reach the required standard in the general part but obtained exceptional grades in physics and mathematics. In 1896 he enrolled at the Swiss Federal Polytechnic graduating in 1900. In 1902 Einstein was hired as an assistant examiner at the Federal Office for Intellectual Property the patent office in Bern. In 1905 Einstein published four groundbreaking papers. The first explained the photoelectric effect. The second paper explained Brownian motion. The third introduced special relativity. The fourth derived the equation E equals mc squared showing mass energy equivalence. In 1915 Einstein completed general relativity describing gravity as spacetime warping."""

doc_B = """Marie Sklodowska Curie was born on November 7 1867 in Warsaw Poland which was then part of the Russian Empire. She was the youngest of five children of well known teachers Bronislawa and Wladyslaw Sklodowski. Her father was a mathematics and physics instructor. In 1891 at age 24 she moved to Paris and enrolled at the University of Paris the Sorbonne where she studied physics and mathematics. In 1893 she was awarded a degree in physics. In 1894 she met Pierre Curie who was an instructor at the School of Physics and Chemistry. They married on July 26 1895. She began investigating uranium radiation using electrometer techniques. In 1898 she and Pierre discovered two new elements polonium named after Poland and radium named for its intense radioactivity. In 1903 Marie Pierre and Becquerel were awarded the Nobel Prize in Physics for their research on radiation. Marie was the first woman to win a Nobel Prize. Pierre died in a street accident on April 19 1906. Marie took over his teaching post becoming the first female professor at the Sorbonne. In 1911 she received a second Nobel Prize in Chemistry for her discovery of radium and polonium. During World War One she developed mobile radiography units nicknamed petites Curies to provide X ray services to field hospitals. She trained over 150 women to use X ray equipment. She founded the Curie Institute in Paris in 1920. Marie Curie died on July 4 1934 at Sancellemoz sanatorium from aplastic anemia caused by long term radiation exposure."""

queries = [
    ("What was Einstein father profession?", "salesman"),
    ("At what age did Einstein teach himself algebra?", "12"),
    ("Where was the patent office where Einstein worked?", "Bern"),
    ("In what year did Einstein publish four papers?", "1905"),
    ("What equation did Einstein derive?", "mc squared"),
    ("How old was Marie Curie when she moved to Paris?", "24"),
    ("What date did Pierre Curie die?", "April 19"),
    ("How many women did Curie train?", "150"),
    ("What was the Curie Institute founded year?", "1920"),
    ("What caused Marie Curie death?", "aplastic anemia"),
]

combined_AB = doc_A + " " + doc_B
combined_BA = doc_B + " " + doc_A

# Test with extreme compression to force differentiation
for ratio in [0.05, 0.10, 0.15]:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Compression ratio: {ratio}")
    print(sep)

    kv = model.prefill(combined_AB, do_score=True)
    kv.prune(ratio=ratio)
    store.save_chunk(f"hard_{ratio}", kv)
    surviving = sum(kv.info["len_k"][0]).item()
    print(f"Stored [A+B]: {kv._seen_tokens} tokens, surviving: {surviving}")

    # Baseline [B+A]
    kv_full = model.prefill(combined_BA, do_score=False)
    bl_matches = 0
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
        out = model.generate(qi, kv=kv_full, update_cache=False)
        if kw.lower() in out.lower():
            bl_matches += 1
    print(f"Baseline [B+A]: {bl_matches}/{len(queries)}")

    # No blend
    kv_no = store.load_chunk(f"hard_{ratio}", device=model.device)
    no_matches = 0
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
        out = model.generate(qi, kv=kv_no, update_cache=False)
        if kw.lower() in out.lower():
            no_matches += 1
    print(f"No blend: {no_matches}/{len(queries)}")

    # Blend with reordered context
    methods = ["iw_hkvd", "diff_only", "random"]
    recomp_ratios = [0.1, 0.3, 0.5]

    for method in methods:
        for r in recomp_ratios:
            matches = 0
            for q, kw in queries:
                kv_bl = store.load_chunk(f"hard_{ratio}", device=model.device)
                qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
                out = model.blend_generate_v2(
                    qi, [kv_bl], recomp_ratio=r, check_layers=[1],
                    method=method, context=combined_BA)
                if kw.lower() in out.lower():
                    matches += 1
            print(f"  {method:>16s} r={r:.1f} match={matches}/{len(queries)}")

print("\nDone!")
