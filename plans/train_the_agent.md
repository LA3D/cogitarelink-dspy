Quick note first 📂 – the components.py, core.py, memory.py, and memory_training.py files you just uploaded aren’t visible on my side any more (the temporary links expired).
If you’d like detailed line‑by‑line feedback or code edits, please re‑upload them and I’ll weave them into the plan.

⸻

“Train‑the‑Agent” — a step‑by‑step recipe that marries Cogitarelink with DSPy SIMBA

The goal is to let SIMBA tune every prompt fragment and demo example in your pipeline while it is exercising real Linked‑Open‑Data calls (contexts, SHACL, Wikidata SPARQL).

### Phase 0  — Freeze a deterministic baseline
	1.	Build the minimal pipeline (HelloLOD) exactly as you’ve done:
English → SPARQL → triples → ReasonOverGraph, plus RememberGraph memory.
	2.	Turn off learning by calling the modules directly (no dspy.compile) and cache every external call with Cogitarelink’s LRU decorators.
Why? SIMBA’s search works best when the underlying tooling is stable and fast.

### Phase 1  — Create a train / dev / test split with live LOD

split	source	note
train	200 QALD + 50 custom Wikidata “gotchas” (e.g. wdt:P1476 alias patterns)	SIMBA will mutate prompts here.
dev	40 hand‑checked questions	Used by SIMBA for early stopping.
test	100 unseen questions	Final report; never shown to SIMBA.

Record gold answers as normalized strings (e.g. canonical IRIs or literal forms) so your metric is cheap to compute.

### Phase 2  — Define a fast metric

def metric(pred, gold):
    ok = (pred["answer"].strip() == gold["answer"].strip())
    # reward provenance
    if ok and pred.get("provenance"): ok = 1.2
    return ok

Return float so SIMBA can see gradients between 0 – 1 (or 1.2).

### Phase 3  — Wrap the entire pipeline in a SIMBA‑optimisable program

import dspy
from cogitarelink_dspy.pipelines import HelloLOD      # your deterministic pipeline

agent = HelloLOD()

# SIMBA = Stochastic Introspective Mini‑Batch Ascent prompt optimiser
simba = dspy.SIMBA(metric=metric,
                   max_steps=16,       # adjust for budget
                   max_demos=12,       # it may synthesise up to 12 demos
                   batch_size=8)       # each revision scored on 8 train Qs

SIMBA will now regard every docstring, system prompt, Signature field name, and few‑shot demo as mutable parameters.  ￼

### Phase 4  — Compile with SIMBA

optimized_agent = simba.compile(agent,
                                trainset=train_split,
                                devset=dev_split,
                                seed=42)

Internally SIMBA:
	1.	Samples a mini‑batch.
	2.	Perturbs one prompt chunk (or swaps a demo).
	3.	Evaluates via your metric.
	4.	Keeps the change if score ↑.
	5.	Repeats for max_steps.

Because GraphManager is cached, most iterations run < 500 ms even with real Wikidata calls.

### Phase 5  — Plug back memory reflections during SIMBA search (optional)

If you want SIMBA to also learn when to write reflections:
	1.	Add a dummy field in the output dict, {"remember": "note" | ""}.
	2.	Extend metric to give +0.05 when the note matches a known pattern (e.g. it writes “P1476 is title”).
	3.	Wrap the pipeline so the remember string is sent to RememberGraph only when non‑empty.

SIMBA will discover for which failures it’s profitable to emit a note.

### Phase 6  — Evaluate & iterate

artefact	pass criteria
train curve	SIMBA score plateaus; no collapse after step 10.
dev accuracy	≥ 70 % exact‑match or better than baseline.
test accuracy	No > 10 pp drop vs dev.
latency	99‑th percentile < 3 s (includes live SPARQL).

If latency dominates, pre‑materialise the most frequent Wikidata entity look‑ups and keep SIMBA running on the cached graph.

### Phase 7  — Graduate to online RL (optional)

Once SIMBA converges, you can hand the final prompts to dspy.RLTrainer (GRPO) for slow, token‑frugal reinforcement on streaming user queries  ￼.

⸻

## Common pitfalls & quick fixes

issue	fix
Prompt bloat (hitting 32 k tokens)	Move layer explanations from system prompt to individual tool docstrings (+ SIMBA can still mutate).
Noisy metric due to endpoint errors	Retry SPARQL twice; return score 0 on persistent failure so SIMBA avoids flaky demos.
SIMBA stuck at local optimum	Increase max_demos; allow it to invent short “counter‑example” demos.
Reflection spam	Add negative metric weight for ≥ 3 notes per answer.


⸻

### Next steps
	1.	Re‑upload the four Python files so I can align the wrapper signatures and caching layer exactly with your code.
	2.	Spin up Phase 0‑2 inside nbs/04_agent.ipynb; we’ll plug SIMBA right after your deterministic agent passes the gold tests.

Below is a clean, reproducible training recipe for turning your MemoryPlanner into a SIMBA‑optimised component that really learns when to add, recall, or format reflections while it is plugged into a live GraphManager.

I start by pointing out the two structural issues that keep your current code from training well, then walk through a six‑stage plan that fixes them and wires Everything‑Together™.

⸻

0  Why training feels “off” right now

current pattern	why it hurts learning
Metric only checks the tool name (tool_match)	SIMBA can succeed by emitting the right word in the decision string even if the action is wrong or the memory payload is empty.
ReflectionStore is mocked (MagicMock) in both planner and tests	The model never sees latency/cost of real graph ops, so it doesn’t learn to prefer ReflectionPrompt over RecallReflection when only a short context is needed.


⸻

1  Step‑by‑step training pipeline

### Step 1  — Wire a real in‑memory GraphManager

graph = GraphManager(use_rdflib=False)          # <= fast, no external deps
store = ReflectionStore(graph)                  # pass this to MemoryPlanner

⚙️ Tip: keep a single store instance for the whole run so add/recall operations accumulate.

⸻

### Step 2  — Create a labeled dataset that exercises each path

{"q":"Remember that wdt:P1476 is the title property","exp_tool":"AddReflection"}
{"q":"What reflections do I have about SPARQL timeouts?","exp_tool":"RecallReflection"}
{"q":"Inject the memories so GPT can see them","exp_tool":"ReflectionPrompt"}

At least 20 rows (≈ 7 per tool) so SIMBA has room to mutate demos.

⸻

### Step 3  — Use a richer metric

def memory_metric(pred, gold):
    good_tool = gold["exp_tool"] in pred.get("trace", [])
    if not good_tool:
        return 0.0

    # Extra points if the action produced non‑trivial payload
    has_payload = "•" in pred["response"] or "ID:" in pred["response"]
    return 1.0 + 0.2*has_payload

This forces SIMBA to optimise both routing and content.

⸻

### Step 4  — Expose the decision surface to SIMBA

Right now MemoryPlanner.decide_tool is a black‑box prompt.
SIMBA learns faster when it can mutate few‑shot demos inside that prompt, so rewrite:

class MemoryDecider(dspy.Module):
    """query -> tool_name:str
    Choose exactly one of {AddReflection, RecallReflection, ReflectionPrompt}."""
    decide = dspy.Predict("query -> tool_name")
    def forward(self, query:str): return self.decide(query=query).tool_name

Replace the free‑text decision string with a single token.
Less randomness = clearer reward signal.

⸻

### Step 5  — Train with SIMBA

planner = MemoryPlanner(graph_manager=graph)   # uses the real store

simba = dspy.SIMBA(metric=memory_metric,
                   max_steps=24,
                   max_demos=10,
                   batch_size=6,
                   seed=0)

optimized = simba.compile(planner,
                          trainset=train_split,
                          devset=dev_split)

SIMBA now tweaks:
	•	docstring of MemoryDecider
	•	the few‑shot demo list it invents on‑the‑fly
	•	field names / role headings

and watches memory_metric on every mini‑batch.

⸻

### Step 6  — Lock & ship

save_optimized_planner(optimized, "cogitarelink_dspy/optimized/memory.pkl")

Deploy by loading that pickle inside CogitarelinkAgent.

Optional: re‑export the learned demos into COMPONENTS.py to make them version‑controlled.

⸻

2  Code snippets you need to tweak

### 2.1  Inject the real store

# in MemoryTraining notebook
graph = GraphManager(use_rdflib=False)
optimized = train_memory_planner_simba(
               trainset=trainset,
               metric=memory_metric,
               graph_manager=graph)

### 2.2  Tighten tool matching in tool_match

Remove fuzzy text checks—now we rely on tool_name string only.

def tool_match(pred, samp): return samp["exp_tool"] in pred["trace"]

### 2.3  Replace MagicMock fallback

In both MemoryPlanner and CogitarelinkAgent constructors:

if graph_manager is None:
    graph_manager = GraphManager(use_rdflib=False)


⸻

3  Smoke‑test before SIMBA

mp = MemoryPlanner(graph)
r1 = mp("Remember that cat is Q146")
assert "AddReflection" in r1["trace"]

r2 = mp("What reflections do I have?")
assert "RecallReflection" in r2["trace"] and "•" in r2["response"]

If these asserts pass deterministically, SIMBA will have a stable surface to optimise.

⸻

4  After SIMBA converges
	1.	Replay 100 unseen questions; check ≥ 80 % correct routing.
	2.	Measure cost: average tokens per answer should drop because RecallReflection avoids hitting the LLM for repeated info.
	3.	Upgrade the full agent: pass store.as_prompt(k) into the system message so every downstream call inherits the latest lessons.

⸻