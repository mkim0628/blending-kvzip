"""Long document + high retention: maximize surviving doc tokens for IW-HKVD differentiation."""
import torch
from model import ModelKVzip
from attention.blend import ChunkStore

torch.set_grad_enabled(False)

model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct")
store = ChunkStore("./chunk_store_long_v2")

# ~800 tokens each — dense factual content
doc_A = """Albert Einstein was born on March 14 1879 in Ulm in the Kingdom of Wuerttemberg in the German Empire. His father Hermann Einstein was a salesman and engineer who founded Elektrotechnische Fabrik J Einstein and Cie a company that manufactured electrical equipment based on direct current. His mother was Pauline Koch. In 1880 the family moved to Munich where Einstein father and his uncle Jakob founded Elektrotechnische Fabrik J Einstein and Cie which manufactured electrical equipment based on direct current. Albert attended a Catholic elementary school in Munich from the age of five for three years. At the age of eight he was transferred to the Luitpold Gymnasium now known as the Albert Einstein Gymnasium where he received advanced primary and secondary school education until he left the German Empire seven years later. In 1894 Hermann Einstein company failed to get an important contract to electrify the city of Munich and was forced to move to Italy. Albert was left at a boarding house in Munich and expected to finish his education. Alone miserable and repelled by the looming prospect of military duty when he turned sixteen Albert ran away six months later and landed on the doorstep of his surprised parents. His parents realized the enormous problems that he faced as a school dropout and draft dodger with no employable skills. His prospects did not look promising. In 1895 Einstein sat the entrance examination for the Swiss Federal Polytechnic in Zurich. He failed the general part of the examination but obtained exceptional grades in physics and mathematics. On the advice of the principal of the Polytechnic he attended the Argovian cantonal school in Aarau Switzerland in 1895 and 1896 to complete his secondary schooling. In September 1896 he passed the Swiss Matura with mostly good grades and though only seventeen enrolled in the four year mathematics and physics teaching diploma program at the Zurich Polytechnic. Marie Winteler who was a year older moved to Olsberg Switzerland for a teaching post. Einstein future wife a twenty year old Serbian named Mileva Maric also enrolled at the Polytechnic that year. She was the only woman among the six students in the mathematics and physics section. Over the next few years Einstein and Maric friendship developed into romance. In 1900 Einstein passed the exams in Maths and Physics and was awarded a Federal teaching diploma. In 1901 he gained Swiss citizenship and as he had failed to find a teaching post he accepted a position as technical assistant in the Swiss Patent Office. In 1905 he received his doctorate from the University of Zurich."""

doc_B = """Marie Sklodowska Curie was born on 7 November 1867 in Warsaw Poland which was then part of Vistula Land in the Russian Empire. She was the youngest of five children of well known teachers Bronislawa nee Boguska and Wladyslaw Sklodowski. Both sides of the family had lost their property and fortunes through patriotic involvements in Polish national uprisings aimed at restoring Poland independence. This left the family members struggling to get ahead in life. Marias paternal grandfather Jozef Sklodowski had been a respected teacher in Lublin. Her father Wladyslaw Sklodowski taught mathematics and physics and was also director of two Warsaw gymnasia for boys. Her mother Bronislawa operated a prestigious Warsaw boarding school for girls. She resigned from the position after Maria was born. When Maria was ten years old her mother Bronislawa died of tuberculosis in May 1878. Less than three years earlier Marias oldest sibling Zofia had died of typhus contracted from a boarder. Marias father was an atheist and her mother a devout Catholic. The deaths of Marias mother and sister caused her to give up Catholicism and become agnostic. When she was ten years old Maria began attending the boarding school of J Sikorska and subsequently attended a gymnasium for girls from which she graduated on 12 June 1883 with a gold medal. After a collapse attributed to depression she spent the following year in the countryside with relatives of her father and the next year with her father in Warsaw where she did some tutoring. Unable to enroll in a regular institution of higher education because she was a woman she and her sister Bronislawa became involved with the clandestine Flying University. Maria made an agreement with her sister Bronislawa that she would give her financial assistance during Bronislawa medical studies in Paris in exchange for similar assistance two years later. In connection with this Maria took a position as governess first with a family in Szczuki then for two years with a family in Ciechanow. While working for the latter family she studied in her own time. In early 1889 she returned home to her father in Warsaw. She continued to work as a governess and continued studying. She began her practical scientific training in a chemical laboratory at the Museum of Industry and Agriculture at Krakowskie Przedmiescie 66 run by her cousin Jozef Boguski who had been an assistant in the Saint Petersburg laboratory of Dmitri Mendeleev. In 1891 she left Poland for France. In Paris Maria briefly found shelter with her sister and brother in law before renting a garret closer to the university in the Latin Quarter. She studied physics chemistry and mathematics at the University of Paris. She survived on her meager resources supplementing her income with some evening tutoring. In 1893 she was awarded a degree in physics and began work in an industrial laboratory of Professor Gabriel Lippmann. Meanwhile she continued studying at the University of Paris and in 1894 earned a second degree. She met Pierre Curie in the spring of 1894. Pierre was an instructor at the School of Physics and Chemistry. They were introduced by the Polish physicist Professor Jozef Wierusz Kowalski who had learned that she was looking for a larger laboratory space. Pierre had no sympathy for feminists but he found Marie to be an exceptional woman. She was his equal and tried hard to be worthy of him. They married on 26 July 1895."""

queries = [
    ("What was the name of Einstein father company?", "Elektrotechnische"),
    ("What school did Einstein attend in Munich?", "Luitpold"),
    ("Why did Einstein family move to Italy?", "failed"),
    ("What year did Einstein get Swiss citizenship?", "1901"),
    ("Where did Einstein work after failing to find teaching post?", "Patent Office"),
    ("What year did Einstein receive his doctorate?", "1905"),
    ("What was Marie Curie mother name?", "Bronislawa"),
    ("What disease killed Marie mother?", "tuberculosis"),
    ("What was the Flying University?", "clandestine"),
    ("Who introduced Marie and Pierre Curie?", "Kowalski"),
    ("When did Marie Curie arrive in France?", "1891"),
    ("When did Marie and Pierre marry?", "1895"),
]

combined_AB = doc_A + " " + doc_B
combined_BA = doc_B + " " + doc_A

# High retention ratios — more surviving doc tokens
for ratio in [0.30, 0.40, 0.50]:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Compression ratio: {ratio}")
    print(sep)

    kv = model.prefill(combined_AB, do_score=True)
    kv.prune(ratio=ratio)
    store.save_chunk(f"long_{ratio}", kv)
    surviving = sum(kv.info["len_k"][0]).item()
    total = kv._seen_tokens
    print(f"Stored [A+B]: {total} tokens, surviving: {surviving} ({surviving/total*100:.0f}% of total)")

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
    kv_no = store.load_chunk(f"long_{ratio}", device=model.device)
    no_matches = 0
    for q, kw in queries:
        qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
        out = model.generate(qi, kv=kv_no, update_cache=False)
        if kw.lower() in out.lower():
            no_matches += 1
    print(f"No blend: {no_matches}/{len(queries)}")

    # Blend with reordered context
    methods = ["iw_hkvd", "diff_only", "random"]
    recomp_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]

    for method in methods:
        for r in recomp_ratios:
            matches = 0
            for q, kw in queries:
                kv_bl = store.load_chunk(f"long_{ratio}", device=model.device)
                qi = model.apply_template(q + "\nAnswer briefly with the exact fact.")
                out = model.blend_generate_v2(
                    qi, [kv_bl], recomp_ratio=r, check_layers=[1],
                    method=method, context=combined_BA)
                if kw.lower() in out.lower():
                    matches += 1
            print(f"  {method:>16s} r={r:.2f} match={matches}/{len(queries)}")

print("\nDone!")
