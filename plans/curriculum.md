Below is a ready‑to‑implement curriculum blueprint you can drop straight into your repo as
data/curriculum/README.md.
It tells SIMBA (or any other DSPy optimiser) exactly how to expose your domain expertise—“how to retrieve vocabularies, negotiate formats, debug SPARQL, cache reflections,” etc.—to the model one layer at a time.

⸻

0  Curriculum philosophy
	1.	Layered → spiral Teach the four Cogitarelink layers in order (Context → Ontology → Rules → Instances) but keep ​“spiralling back” by adding harder tasks that revisit earlier layers with new wrinkles.
	2.	One new concept per mini‑batch SIMBA works best when a batch shares a single learning objective.
	3.	Concrete ≫ abstract Every example produces machine‑verifiable output (IRI, boolean, SHA‑256).
	4.	Self‑reflection Any “gotcha” you know goes in the dev‑set and a matching reflection‑note row so the agent learns to store it.

⸻

1  File structure

data/
└── curriculum/
    ├── 00_context.jsonl
    ├── 01_ontology.jsonl
    ├── 02_rules.jsonl
    ├── 03_instances.jsonl
    ├── 04_content_negotiation.jsonl
    ├── 05_wikidata_quirks.jsonl
    └── 99_reflection_seed.jsonl

Every JSONL row has the fields you already use:

{
  "q":        "Compose schema+dc context",
  "exp_tool": "ComposeContext",
  "answer":   "\"@context\" length ≥ 2500",
  "tags":     ["context","composition"],
  "remember": "schema+dc composition may exceed 32 k tokens; prefer minimal subset."
}

remember is optional—if present, MemoryPlanner should emit an AddReflection call after answering.

⸻

2  Stage‑by‑stage goals & example rows

stage	concept to master	3 starter examples you should write (→ rows)
00 Context	find & merge JSON‑LD contexts	“Compact this doc with Schema.org”; “Detect collision between schema & FOAF”; “Give me full IRI for dc:title”.
01 Ontology	fetch vocab registries, resolve terms	“What is rdfs:label’s domain in schema?”; “Retrieve owl:equivalentClass for schema:Person”.
02 Rules	run SHACL, SPARQL CONSTRUCT rules	“Validate ex:Bob against PersonShape”; “Run sh:PersonAgeRule on graph g1”.
03 Instances	SPARQL over local & remote graphs	“How many People in graph g2?”; “Fetch 3 cats from Wikidata”.
04 Content negotiation	negotiate formats / headers	“Retrieve Turtle version of schema.org”; “HEAD request for application/ld+json”.
05 Wikidata quirks	alias properties, timeouts, paging	“wdt:P1476 meaning?” → expect awareness that it’s title; “Handle maxLag error—retry”.

Write 8‑12 rows per stage: 70 % train, 15 % dev, 15 % test.

⸻

3  Encoding your expertise
	1.	Docstrings & system prompt
Write one authoritative paragraph per layer; keep in the global system message so SIMBA doesn’t mutate it.
	2.	Few‑shot demos
Hand‑craft 2–3 demos per stage that showcase best practices (e.g. use HEAD before GET when checking content negotiation). Place them in 00_context.jsonl etc.; SIMBA may mutate/clone them.
	3.	Reflection seeds (99_reflection_seed.jsonl)
Seed the memory graph with the lessons you already know:

{"remember":"Wikidata `rdfs:label` comes in every language—always `FILTER(LANG(?l)='en')`.","tags":["wikidata"]}

Load this file once at boot so the agent starts “experienced.”

	4.	Metric shaping
Give +0.2 reward if an answer includes provenance (graph_id, SHA‑256) or if the expected remember note was stored.

⸻

4  Training loop skeleton

from dspy.teleprompt import BootstrapFewShot
from dspy import SIMBA

train = load_jsonl(glob("data/curriculum/*/*.jsonl", exclude="*dev*","*test*"))
dev   = load_jsonl("data/curriculum/*_dev.jsonl")
test  = load_jsonl("data/curriculum/*_test.jsonl")

metric = memory_metric  # from earlier message

# 1) fast bootstrap on entire train set
bootstrap = BootstrapFewShot(devset=dev, metric=metric)
agent0    = dspy.compile(HelloLODWithMemory(), bootstrap, num_iterations=2)

# 2) SIMBA fine‑tune only on memory routing & content negotiation stages
simba     = SIMBA(metric=metric, max_steps=24, batch_size=8)
agent1    = simba.compile(agent0, trainset=train, devset=dev)

evaluate(agent1, test)


⸻

5  Tips for writing curriculum rows
	•	Always include exp_tool so routing can be graded deterministically.
	•	Keep answers atomic: single IRI, boolean, integer, or short literal; easier for exact‑match.
	•	Mix phrasing: “fetch”, “retrieve”, “look up”; SIMBA generalises across synonyms.
	•	Encode errors: write at least one negative example per stage where the first attempt should fail and then succeed after reflection.
	•	Tag aggressively: populate "tags":["context","collision"]—later you can teach tag‑conditioned recall.

Integrating an “active” SHACL tool into your DSPy curriculum

(validation + rule‑based materialisation + feedback loops)

⸻

0  Why SHACL is different

behaviour	consequence for the agent
Validation – returns a report graph with sh:Violation, sh:Warning, etc.	The LLM must parse, summarise, and sometimes repair data.
Rules (SHACL‑AF) – may materialise new triples in a target graph.	Subsequent SPARQL queries should see the new facts; provenance must show they came from the rule.
Large payloads – a single validation report can be hundreds of triples.	The tool wrapper should pre‑digest the report (e.g. return top‑N violations) so prompts stay small.


⸻

1  Design the SHACL tool wrapper

class RunSHACL(dspy.Module):
    """RunSHACL(graph_id:str, shapes_id:str, mode:str='validate'|'infer', top:int=5)
       → dict(report:str, added:int, warnings:int, violations:int, prov:str)"""
    signature = dspy.Signature("graph_id:str shapes_id:str mode:str top:int -> result:dict")

    def forward(self, graph_id, shapes_id, mode="validate", top=5):
        from cogitarelink.verify.validator import validate_graph
        from cogitarelink.reason.sandbox import apply_shacl_rules

        if mode == "infer":
            added = apply_shacl_rules(graph_id, shapes_id)         # returns #triples added
            return dict(report="", added=added, warnings=0, violations=0,
                        prov=f"{graph_id}#ruleMaterialised")
        else:
            r = validate_graph(graph_id, shapes_id)                # returns a SHACL report graph
            v = int(r.query("SELECT (COUNT(?v) AS ?c) WHERE {?v a sh:Violation}")[0][0])
            w = int(r.query("SELECT (COUNT(?w) AS ?c) WHERE {?w a sh:Warning}")[0][0])
            txt = "\n".join(
                t[0] for t in r.query(
                    "SELECT ?msg WHERE {?r sh:resultMessage ?msg} LIMIT %d" % top)
            )
            return dict(report=txt, added=0, warnings=w, violations=v,
                        prov=f"{graph_id}#shaclReport")

Return a small dict with:
	•	human‑readable report (≤ top lines)
	•	counts for metric scoring
	•	prov so the agent can cite provenance

⸻

2  Add two curriculum stages

stage	sample training rows
06 Validation	“Validate g1 with PersonShape. Should have 0 violations.”  exp_tool: RunSHACL, mode: validate, answer: 0
07 Materialisation	“Apply address inference rules to g2.”  Expect added ≥ 5 and a reflection like “Rules can add duplicates – de‑duplicate later.”

For each row include:

"params": {"graph_id":"g1","shapes_id":"PersonShape","mode":"validate"},
"answer": "violations=0"

In memory_metric, check the numeric count.

⸻

3  Teach the agent to repair after validation fails

Extended workflow:
	1.	Agent calls RunSHACL (validate)
	2.	If violations>0, agent should:
	•	a) write a reflection note summarising the failure;
	•	b) possibly call an AutoFix tool (if you expose one) or ask the user for correction.

Add rows like:

{
 "q": "My graph g3 should pass PersonShape – fix it",
 "exp_tool": "RunSHACL",
 "answer": "violations>0",
 "remember": "PersonShape often fails because birthDate is missing."
}

Metric: +0.2 when the reflection note is stored after a failing validation.

⸻

4  Metric update

def shacl_metric(pred, gold):
    if gold["exp_tool"] not in pred["trace"]:
        return 0
    score = 1
    # reward correct violation count or triple‑added count
    if "violations" in pred and "violations" in gold:
        score += 0.2 * (pred["violations"] == gold["violations"])
    if "added" in pred and "added" in gold:
        score += 0.2 * (pred["added"] >= gold["added"])
    # reward reflection on failure
    if gold.get("remember") and "AddReflection" in pred["trace"]:
        score += 0.2
    return score


⸻

5  SIMBA hints
	•	Search space: let SIMBA adjust top lines in the wrapper signature (top:[3,5,10]).
	•	Demos: seed with one pass and one fail example; SIMBA will mutate text but keep the parameter JSON.
	•	Batching: group validation tasks in the same mini‑batch—the wrapper call is cached so later tasks get the report instantly.

⸻

6  Remember to…
	•	Version your shapes – include shape hashes in shapes_id so provenance is unambiguous.
	•	Store materialised triples with provenance (wrap_patch_with_prov) so downstream queries can differentiate inferred vs source data.
	•	Keep prompts small – never return the full SHACL report; only counts + first few messages.
