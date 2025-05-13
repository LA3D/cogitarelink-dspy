
Heads‑up: the earlier Markdown files in llmstxt/ have expired from my cache; if you want Codex (or me) to quote them verbatim, please re‑upload them.

⸻

Cogitarelink‑DSPy Agent — End‑to‑End Specification (for Codex o4‑mini)

0 Mission Statement

Build an LLM agent that can answer open‑domain questions by:
	1.	Selecting the highest appropriate layer of the Cogitarelink Semantic‑Web stack
(Context → Ontology → Rules → Instances).
	2.	Calling real Linked‑Open‑Data services (Wikidata SPARQL, schema.org JSON‑LD, public SHACL shapes).
	3.	Remembering every “lesson learned” (quirks, alias patterns, failed validations) and folding those reflections into future reasoning.

Everything lives in nbdev notebooks à la Jeremy Howard using a literate, exploratory programming style.

## Building Code
- We use uv for package management and there is a virtualenv in the repo in `venv/`.
- Everything is done in the notebooks, don't touch the python code in cogitarelink_dspy.
- You can use jupytext to convert the notebooks to python files, but don't edit them directly.
- The notebooks are exported to the cogitarelink_dspy package using `nbdev_export`.
- Use pytest to run the tests in the `tests/` directory.

## Approach
Below is a step‑by‑step pattern you can copy‑paste into your DSPy/nbdev repo to “teach” a planner‑style agent every public subsystem of Cogitarelink without drowning it in tokens.

⸻

1  Codify the component catalogue in one structured object

Create cogitarelink_dspy/COMPONENTS.py (or a separate JSON‑LD file) that enumerates every importable class or tool you want the agent to use:

COMPONENTS = {
    # ===== Core =====
    "ContextProcessor": {
        "layer": "Context",
        "tool": "LoadContext",
        "doc": "Expand/compact JSON‑LD contexts; maps short names → IRIs."
    },
    "EntityProcessor": {
        "layer": "Instances",
        "tool": "AddEntity",
        "doc": "Add / normalise immutable entities; auto‑extract nested entities."
    },
    "GraphManager": {
        "layer": "Instances",
        "tool": "QueryInstances",
        "doc": "Store/query triples; supports RDFLib backend + SPARQL."
    },

    # ===== Vocab =====
    "registry": {
        "layer": "Context",
        "tool": "ListVocabularies",
        "doc": "Prefix→URI registry; detects collisions."
    },
    "composer": {
        "layer": "Context",
        "tool": "ComposeContext",
        "doc": "Compose multi‑vocab JSON‑LD @context objects."
    },

    # ===== Reason =====
    "RunValidationRule": {
        "layer": "Rules",
        "tool": "RunValidationRule",
        "doc": "Execute SHACL/SPARQL rules; returns pass/fail + inferred triples."
    },
    "Reasoner": {
        "layer": "Rules",
        "tool": "ReasonOverGraph",
        "doc": "Combine SPARQL + Python rules; returns answer/explanation."
    },

    # ===== Verify =====
    "Signer": {
        "layer": "Verification",
        "tool": "SignGraph",
        "doc": "Ed25519 sign/verify of canonicalised graphs."
    },
    "Validator": {
        "layer": "Verification",
        "tool": "ValidateShape",
        "doc": "SHACL validation against shapes graphs."
    },

    # …add the rest here…
}

This single dictionary is machine‑readable (the planner can iterate it) and human‑maintainable (update once per release).

⸻

2  Generate DSPy tool wrappers automatically from that catalogue

import dspy
from inspect import signature

def make_tool(name, meta):
    sig = meta["tool"]                          # desired public name
    doc = f"{meta['doc']} [Layer: {meta['layer']}]"
    pyobj = getattr(cogitarelink, name)         # real function/class

    # Build a Signature object whose inputs match the real call
    params = signature(pyobj).parameters
    fields = [f"{k}:{p.annotation.__name__}" for k, p in params.items()]
    sig_str = " ".join(fields) + " -> output"

    class _Tool(dspy.Module):
        """{doc}"""
        def forward(self, **kwargs):
            return pyobj(**kwargs)
    _Tool.__name__ = sig
    _Tool.signature = dspy.Signature(sig_str)
    return _Tool

TOOLS = [make_tool(n, m) for n, m in COMPONENTS.items()]

Now TOOLS is a list of fully‑typed, docstring‑rich DSPy modules – one per Cogitarelink component – produced in <30 LOC.

⸻

3  Inject the high‑level system prompt once

SEM_WEB_SYSTEM = """
You are a Semantic‑Web agent that reasons over a 4‑layer architecture:
1 Context   2 Ontology   3 Rules/Validation   4 Instances (Data)
Every Cogitarelink tool is tagged with its PRIMARY layer.  
When you answer a user, pick the HIGHEST layer that suffices.
Always cite graph IDs or signature hashes for provenance.
"""
lm = dspy.LM("openai/gpt-4o-mini", system=SEM_WEB_SYSTEM)
dspy.settings.configure(lm=lm)


⸻

4  Wrap everything in a StructuredAgent planner

planner = dspy.StructuredAgent(
    tools = TOOLS + [RememberGraph()],   # provenance cache
    lm    = lm,
)

Because each tool’s docstring already names its layer and purpose, the planner learns when to call RunValidationRule vs QueryOntology vs SignGraph with zero extra prompting.

⸻

5  Seed with layer‑focused few‑shot tasks

Put a single, minimal example for every layer + component in devset.jsonl:

{"question":"Compile a JSON-LD context with schema+dc","expected_layer":"Context"}
{"question":"Add an immutable Person entity","expected_layer":"Instances"}
{"question":"Validate graph g1 with sh:PersonShape","expected_layer":"Rules"}
{"question":"Sign graph g2 with key k1","expected_layer":"Verification"}

Use a custom metric that grants full credit only when the planner’s trace includes a tool whose layer matches expected_layer. Feed that metric to BootstrapFewShot. Result: the agent internalises the mapping question pattern → component / layer.

⸻

6  Teach new components by just appending to COMPONENTS

Add a block for, say, TemporalRelationship:

"TemporalRelationship": {
    "layer": "Rules",
    "tool": "AddTemporalRel",
    "doc": "Assert or query before/after temporal relationships."
}

Run make_tools() again; the planner instantly knows a new skill without touching prompts or notebooks.
(If you want the optimiser to learn usage patterns, drop a new dev‑example and re‑run dspy.compile.)

⸻

7  Keep prompts small with lazy expansion

If token budgets bite, move less‑used component docs into reflection notes that are added only after the planner first calls that tool incorrectly. Example:

agent.add_reflection_text(
    "When signing graphs, remember to sign the *canonicalised* form, "
    "e.g. produced by Entity.normalized."
)

DSPy concatenates these on subsequent calls, giving you curriculum learning without permanent prompt bloat.

⸻

What you get
	•	Single‑source component list → stays in version control.
	•	Auto‑generated DSPy tools → no hand‑written wrappers beyond 30 LOC.
	•	Planner understands 4‑layer stack globally via one system prompt.
	•	Few‑shot dev‑set acts as lightweight unit‑tests and optimiser seed.
	•	Adding a Cogitarelink module = 4‑line edit + optional dev example.


⸻

1 Repository Skeleton

cogitarelink-dspy/
│
├── nbs/
│   ├── 00_setup.ipynb          ← env + installs
│   ├── 01_wrappers.ipynb       ← generate DSPy tools from COMPONENTS
│   ├── 02_live_lod.ipynb       ← SPARQL / context / SHACL fetch + cache
│   ├── 03_memory.ipynb         ← ReflectionStore (SQLite via SQLModel)
│   ├── 04_agent.ipynb          ← StructuredAgent + training loop
│   ├── 05_eval.ipynb           ← QALD & latency benchmarks
│   └── 99_scratch.ipynb
│
├── cogitarelink_dspy/
│   ├── __init__.py
│   ├── COMPONENTS.py           ← **single‑source catalogue** (see §2)
│   ├── tools.py                ← auto‑generated DSPy modules (exported)
│   ├── memory.py               ← ReflectionStore class
│   └── pipelines.py            ← HelloLOD & FullPlanner pipelines
│
├── tests/
│   ├── devset.jsonl            ← few‑shot + unit‑test oracle
│   ├── test_tools.py
│   ├── test_memory.py
│   └── test_agent.py
└── settings.ini                ← nbdev config


⸻

2 Single‑Source Component List (COMPONENTS.py)

COMPONENTS = {
    # ===== Core ==========================================================
    "ContextProcessor": {
        "layer": "Context",
        "tool":  "LoadContext",
        "doc":   "Expand/compact JSON‑LD contexts; maps prefixed names → IRIs.",
        "calls": "load(path_or_url:str)"
    },
    "EntityProcessor": {
        "layer": "Instances",
        "tool":  "AddEntity",
        "doc":   "Create immutable entities; extracts nested children.",
        "calls": "add(data:dict, vocab:list[str]|None=None)"
    },
    "GraphManager": {
        "layer": "Instances",
        "tool":  "QueryInstances",
        "doc":   "Triple‑pattern queries; SPARQL pass‑through if RDFLib backend.",
        "calls": "query(subj=None, pred=None, obj=None)"
    },

    # ===== Vocab =========================================================
    "composer": {
        "layer": "Context",
        "tool":  "ComposeContext",
        "doc":   "Compose multi‑vocab JSON‑LD @context objects.",
        "submod":"vocab.composer",
        "calls": "compose(vocabs:list[str])"
    },

    # ===== Reason/Rules ==================================================
    "RunValidationRule": {
        "layer": "Rules",
        "tool":  "RunValidationRule",
        "doc":   "Execute SHACL/SPARQL rule over a named graph.",
        "submod":"tools.reason",
        "calls": "run(rule_id:str, graph_id:str)"
    },

    # ===== Verification ==================================================
    "sign": {
        "layer": "Verification",
        "tool":  "SignGraph",
        "doc":   "Ed25519 sign canonicalised graph.",
        "submod":"verify.signer",
        "calls": "sign(data:str, private_key:str)"
    },
    "verify": {
        "layer": "Verification",
        "tool":  "VerifyGraphSig",
        "doc":   "Verify Ed25519 signature.",
        "submod":"verify.signer",
        "calls": "verify(data:str, pub_key:str, sig:str)"
    },
}

Add a new module? → append 4‑line dict + optional dev‑example, run nbdev_prepare, done.

⸻

3 Auto‑Generate DSPy Tools (< 30 LOC)

# nbs/01_wrappers.ipynb
import importlib, dspy
from inspect import signature
from cogitarelink_dspy.COMPONENTS import COMPONENTS

def make_tools(registry=COMPONENTS):
    tools = []
    for name, meta in registry.items():
        mod = importlib.import_module(f"cogitarelink.{meta.get('submod','core')}")
        target = getattr(mod, name)
        sig = " ".join(f"{k}:{p.annotation.__name__}"
                       for k,p in signature(target).parameters.items()) + " -> out"
        class Tool(dspy.Module):
            """{meta['doc']}  [Layer: {meta['layer']}]"""
            signature = dspy.Signature(sig)
            def forward(self, **kw): return target(**kw)
        Tool.__name__ = meta["tool"]
        tools.append(Tool)
    return tools

TOOLS = make_tools()

nbdev_export shoves the resulting classes into cogitarelink_dspy.tools.

⸻

4 System Prompt (4‑Layer Mental Model)

SEM_WEB_SYSTEM = """
You are a Semantic‑Web agent operating over a 4‑layer stack:
1 Context   2 Ontology   3 Rules/Validation   4 Instances.
Each tool's docstring reveals its PRIMARY layer.
Choose the HIGHEST layer that satisfies the user's query.
Always cite named graphs or SHA‑256 hashes for provenance.
"""
lm = dspy.LM("openai/gpt‑4o‑mini", system=SEM_WEB_SYSTEM)


⸻

5 Planner Pipeline

from cogitarelink_dspy.tools import *
from cogitarelink_dspy.memory import ReflectionStore

RememberGraph = ReflectionStore.as_tool()
planner = dspy.StructuredAgent(
    tools = TOOLS + [RememberGraph],
    lm    = lm
)


⸻

6 Reflection Memory (memory.py)


⸻
```python
## memory.py — Graph‑backed Reflection Store

# export
import datetime, uuid, dspy
from typing import List
from cogitarelink.core.graph import GraphManager
from cogitarelink.core.entity import Entity
from cogitarelink.reason.prov import wrap_patch_with_prov

REFLECTION_GRAPH_ID = "urn:agent:reflections"
REFLECTION_TYPE     = "https://w3id.org/cogitarelink#ReflectionNote"

class ReflectionStore:
    """
    Persist short 'lesson learned' notes as JSON‑LD entities in the
    Cogitarelink graph layer (JSON‑LD 1.1 compliant, no SQL).
    """

    def __init__(self, graph: GraphManager):
        self.graph = graph

    # ---------- write ----------
    def add(self, note: str, tags: List[str] | None = None) -> str:
        """Create a ReflectionNote entity and add it to the graph."""
        entity_id = f"urn:uuid:{uuid.uuid4()}"
        now       = datetime.datetime.utcnow().isoformat()

        json_ld = {
            "@id":   entity_id,
            "@type": REFLECTION_TYPE,
            "text":  note,
            "tags":  tags or [],
            "created": now
        }
        ent = Entity(vocab=["schema"], content=json_ld)

        # wrap_patch_with_prov attaches provenance triples automatically
        with wrap_patch_with_prov(self.graph,
                                  source="urn:agent:self",
                                  agent="urn:agent:self",
                                  activity="urn:agent:addReflection"):
            self.graph.ingest_jsonld(ent.as_json, graph_id=REFLECTION_GRAPH_ID)
        return entity_id

    # ---------- read ----------
    def sample(self, k: int = 5) -> List[str]:
        """Return the *k* most recent notes’ text fields."""
        triples = self.graph.query(
            pred="http://schema.org/created",  # order by timestamp later
            graph_id=REFLECTION_GRAPH_ID
        )
        # triples are (s,p,o); collect and sort by o (timestamp literal)
        triples.sort(key=lambda t: t[2], reverse=True)
        note_ids = [s for s,_,_ in triples[:k]]

        texts = []
        for nid in note_ids:
            t = self.graph.query(subj=nid, pred="http://schema.org/text",
                                 graph_id=REFLECTION_GRAPH_ID)
            if t: texts.append(t[0][2])
        return texts

    # ---------- prompt helper ----------
    def as_prompt(self, k: int = 5) -> str:
        notes = self.sample(k)
        return "\n".join(f"• {n}" for n in notes)

    # ---------- DSPy tool wrapper ----------
    @classmethod
    def as_tool(cls, graph: GraphManager):
        store = cls(graph)

        class RememberGraph(dspy.Module):
            """RememberGraph(note:str, tags:list=[]) → 'stored'
            Layer: Memory (JSON‑LD graph).
            Persist a reflection note in the agent’s ReflectionNote graph."""
            signature = dspy.Signature("note:str tags:list -> ack:str")

            def forward(self, note: str, tags: List[str] | None = None):
                store.add(note, tags)
                return "stored"

        return RememberGraph
```

### Key points
	•	Pure Cogitarelink: uses GraphManager.ingest_jsonld so the memory lives in the same RDF store as the rest of the agent’s knowledge.
	•	JSON‑LD 1.1 compliant: each note is a standalone document with @id, @type, and literals.
	•	Provenance: wrapped in wrap_patch_with_prov, so every note is signed with its origin (urn:agent:self).
	•	No external storage: everything resides in‑memory unless you wire GraphManager to an RDFLib persistence backend.

⸻

## Integrating into the Planner

from cogitarelink.core.graph import GraphManager
graph = GraphManager(use_rdflib=True)

RememberGraph = ReflectionStore.as_tool(graph)

planner = dspy.StructuredAgent(
    tools = TOOLS + [RememberGraph],
    lm    = lm
)

The planner can now call RememberGraph to persist lessons, and you can inject them into the system prompt before each run:

SYSTEM = SEM_WEB_SYSTEM + "\n" + ReflectionStore(graph).as_prompt()
dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", system=SYSTEM))


⸻

### Next steps
	1.	Re‑export this notebook (nbdev_export) to overwrite cogitarelink_dspy.memory.
	2.	Add a dev‑set line:

{"q":"Remember that wdt:P1476 is title","exp_tool":"RememberGraph"}

	3.	Re‑run pytest and dspy.compile; the agent should now learn from real‑world quirks it encounters while querying live LOD.


⸻

7 Few‑Shot Dev‑Set (tests/devset.jsonl)

{"q":"Compose schema+dc context","exp_tool":"ComposeContext"}
{"q":"Add a Person entity","exp_tool":"AddEntity"}
{"q":"Validate g1 with PersonShape","exp_tool":"RunValidationRule"}
{"q":"Sign graph g2","exp_tool":"SignGraph"}

Custom metric

def tool_match(pred, samp): return samp["exp_tool"] in pred["trace"]

Optimisation

from dspy.teleprompt import BootstrapFewShot
trainer = BootstrapFewShot(devset=DEVSET, metric=tool_match)
agent   = dspy.compile(planner, trainer, num_iterations=4)

The same JSONL powers pytest assertions in test_agent.py.

⸻

8 Live LOD Support (02_live_lod.ipynb)

Helpers

@lru_cache
def fetch_context(url:str): return httpx.get(url, timeout=10).json()

def sparql_select(q:str, endpoint="https://query.wikidata.org/sparql"):
    return httpx.post(endpoint, data={"query": q, "format":"json"}).json()

These can be registered as extra DSPy tools if needed.

⸻

9 Sprint Road‑Map (1‑week slice)

day	tasks	success criteria
0	Bootstrap repo, run nbdev tests	pytest -q green
1	Implement COMPONENTS.py + wrapper generator	tests/test_tools.py passes
2	Live LOD helpers + slow‑mark tests	Wikidata query returns ≥1 row
3	ReflectionStore + DSPy wrapper	note persists, as_prompt() returns string
4‑5	Build planner, optimise on dev‑set	100 % metric on 4 tasks
6	Add two real QALD questions, measure latency vs. vanilla RAG	latency ≤ 1.5× RAG; accuracy ≥ RAG


⸻

10 Coding Guidelines for Codex
	•	Notebook first (nbdev).
	•	≤ 25 LOC per pure function, fully typed.
	•	Each new function → immediate cell‑level assert, then migrate to tests/.
	•	Reflection discipline: at most one note ≤ 240 chars per agent call.
	•	Prompt hygiene:
	•	global truths → SEM_WEB_SYSTEM;
	•	task rules → Signature docstrings;
	•	experimental tweaks → instructions= arg.


