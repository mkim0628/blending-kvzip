"""Compression sweep: IW-HKVD should differentiate at high compression + small recomp budget."""
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

torch.set_grad_enabled(False)

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
store = ChunkStore("./chunk_store_sweep2")

doc1 = """Albert Einstein was born on March 14 1879 in Ulm in the Kingdom of Wuerttemberg in the German Empire. His father Hermann Einstein was a salesman and engineer who founded Elektrotechnische Fabrik J Einstein and Cie. His mother Pauline Koch came from a wealthy family. In 1880 the family moved to Munich where Einstein father and uncle Jakob founded a company manufacturing electrical equipment. Einstein attended the Luitpold Gymnasium where he received advanced primary and secondary education. At age 12 Einstein taught himself algebra and Euclidean geometry over a single summer. He also independently discovered his own original proof of the Pythagorean theorem. In 1894 Einstein father company failed and the family moved to Italy first to Milan and then to Pavia. In 1895 at the age of sixteen Einstein sat the entrance examination for the Swiss Federal Polytechnic in Zurich. He failed the general part but obtained exceptional grades in physics and mathematics. In 1896 he enrolled at the Swiss Federal Polytechnic graduating in 1900. In 1902 Einstein was hired as an assistant examiner at the Federal Office for Intellectual Property the patent office in Bern. In 1905 Einstein published four groundbreaking papers including special relativity and the photoelectric effect explanation which later earned him the Nobel Prize in Physics in 1921."""

doc2 = """Marie Sklodowska Curie was born on November 7 1867 in Warsaw Poland which was then part of the Russian Empire. She was the youngest of five children of well known teachers. Her father was a mathematics and physics instructor. In 1891 at age 24 she moved to Paris and enrolled at the University of Paris the Sorbonne. In 1893 she was awarded a degree in physics. In 1894 she met Pierre Curie who was an instructor at the School of Physics and Chemistry. They married on July 26 1895. In 1898 she and Pierre discovered two new elements polonium and radium. In 1903 Marie Pierre and Becquerel were awarded the Nobel Prize in Physics. Pierre died in a street accident on April 19 1906. In 1911 she received a second Nobel Prize in Chemistry. During World War One she developed mobile radiography units called petites Curies. She trained over 150 women to use X ray equipment. She founded the Curie Institute in Paris in 1920. Marie Curie died on July 4 1934 from aplastic anemia caused by radiation exposure."""

doc3 = """Alan Mathison Turing was born on 23 June 1912 in Maida Vale London England. His father Julius Mathison Turing was a civil servant in British India. Alan showed exceptional mathematical ability from an early age. He attended Sherborne School at age 13 where he became interested in science and mathematics. In 1931 Turing went to King College Cambridge to study mathematics graduating in 1934 with first class honours. In 1936 he published his seminal paper On Computable Numbers introducing the concept of a Turing machine which became foundational to theoretical computer science. In 1938 he obtained his PhD from Princeton University under the supervision of Alonzo Church. During World War Two Turing played a pivotal role at Bletchley Park in breaking German Enigma codes. He designed the Bombe machine which significantly reduced the work needed to decrypt intercepted messages. After the war Turing worked at the National Physical Laboratory where he designed the Automatic Computing Engine one of the first designs for a stored program computer. In 1948 he moved to the University of Manchester. In 1950 he published Computing Machinery and Intelligence proposing the Turing test as a measure of machine intelligence. Turing was prosecuted in 1952 for homosexual acts which were then criminal in Britain. He accepted chemical castration as an alternative to prison. Alan Turing died on 7 June 1954 from cyanide poisoning. In 2013 Queen Elizabeth II granted Turing a posthumous royal pardon."""

queries_doc1 = [
    ("What was Einstein father profession?", "salesman"),
    ("At what age did Einstein teach himself algebra?", "12"),
    ("Where was the patent office where Einstein worked?", "Bern"),
    ("What year did Einstein publish four papers?", "1905"),
    ("What prize did Einstein win?", "Nobel"),
]

queries_doc3 = [
    ("When was Alan Turing born?", "1912"),
    ("What did Turing publish in 1936?", "Computable Numbers"),
    ("What code breaking machine did Turing design?", "Bombe"),
    ("What test did Turing propose?", "Turing test"),
    ("When did Turing die?", "1954"),
    ("Who granted Turing a pardon?", "Elizabeth"),
]

queries = queries_doc1 + queries_doc3

stored_context = doc1 + " " + doc2
new_context = doc1 + " " + doc3

# Compression: 70%, 80%, 90% (retain 30%, 20%, 10%)
for ratio in [0.30, 0.20, 0.10, 0.05]:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Retention ratio: {ratio} (compression: {(1-ratio)*100:.0f}%)")
    print(sep)

    kv = model.prefill(stored_context, do_score=True)
    kv.prune(ratio=ratio)
    store.save_chunk(f"sweep_{ratio}", kv)
    surviving = sum(kv.info["len_k"][0]).item()
    total = kv._seen_tokens
    doc_surviving = surviving - 28 * 4  # subtract sys prompt (28) * 4 heads approx
    print(f"Stored [doc1+doc2]: {total} tokens, surviving: {surviving}, ~doc tokens: {max(0,doc_surviving)}")

    # Baseline
    kv_full = model.prefill(new_context, do_score=False)
    bl = sum(1 for q, kw in queries
             for out in [model.generate(model.apply_template(q + "\nAnswer briefly with the exact fact."), kv=kv_full, update_cache=False)]
             if kw.lower() in out.lower())
    print(f"Baseline [doc1+doc3]: {bl}/{len(queries)}")

    # No blend
    no_d1 = no_d3 = 0
    for q, kw in queries_doc1:
        kv_no = store.load_chunk(f"sweep_{ratio}", device=model.device)
        out = model.generate(model.apply_template(q + "\nAnswer briefly with the exact fact."), kv=kv_no, update_cache=False)
        if kw.lower() in out.lower(): no_d1 += 1
    for q, kw in queries_doc3:
        kv_no = store.load_chunk(f"sweep_{ratio}", device=model.device)
        out = model.generate(model.apply_template(q + "\nAnswer briefly with the exact fact."), kv=kv_no, update_cache=False)
        if kw.lower() in out.lower(): no_d3 += 1
    print(f"No blend: doc1={no_d1}/5 doc3={no_d3}/6 total={no_d1+no_d3}/{len(queries)}")

    # Blend with very small recomp ratios
    methods = ["iw_hkvd", "diff_only", "random"]
    recomp_ratios = [0.01, 0.03, 0.05, 0.10, 0.20]

    for r in recomp_ratios:
        row = f"  r_blend={r:.2f} |"
        for method in methods:
            m_d1 = m_d3 = 0
            for q, kw in queries_doc1:
                kv_bl = store.load_chunk(f"sweep_{ratio}", device=model.device)
                qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
                out = model.blend_generate_v2(qi, [kv_bl], recomp_ratio=r, check_layers=[1], method=method, context=new_context)
                if kw.lower() in out.lower(): m_d1 += 1
            for q, kw in queries_doc3:
                kv_bl = store.load_chunk(f"sweep_{ratio}", device=model.device)
                qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
                out = model.blend_generate_v2(qi, [kv_bl], recomp_ratio=r, check_layers=[1], method=method, context=new_context)
                if kw.lower() in out.lower(): m_d3 += 1
            row += f" {method}={m_d1+m_d3}/{len(queries)} |"
        print(row)

print("\nDone!")
