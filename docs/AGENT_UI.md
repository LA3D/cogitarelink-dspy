Below is a drop‑in notebook scaffold that gives you a working, dialog‑style instructor UI in one Jupyter cell.
Everything is pure Python/ipywidgets, so you stay inside the nbdev/fast‑ai workflow; nothing to deploy.

⸻

0  Prereqs – one cell to run first

# Cell 0  – imports & agent bootstrap
from ipywidgets import Textarea, Button, VBox, HBox, Output, Layout
from IPython.display import display, Markdown
from cogitarelink.core.graph import GraphManager
from cogitarelink_dspy.memory import ReflectionStore
from cogitarelink_dspy.memory_training import load_optimized_planner  # if you trained it
from cogitarelink_dspy.wrappers import get_tool_by_name, get_tools
import dspy, datetime, json, uuid, os

# --- create the semantic memory + agent ----
try:
    graph = GraphManager(use_rdflib=True)          # 🔴 full feature‑set
except ImportError as e:
    # RDFLib isn’t installed or rdflib‑sparql extras missing
    print("⚠️  RDFLib not available – falling back to in‑memory graph "
          "(SPARQL/SHACL disabled).  Install with: pip install cogitarelink[rdf]")
    graph = GraphManager(use_rdflib=False)         # still works for reflections

mem   = ReflectionStore(graph)

try:  # load tuned MemoryPlanner if it exists
    MemoryPlanner = load_optimized_planner(graph_manager=graph)
except FileNotFoundError:
    from cogitarelink_dspy.memory_training import MemoryPlanner
    MemoryPlanner = MemoryPlanner(graph)

# assemble all DSPy tools including memory wrappers
tools  = get_tools() + [MemoryPlanner.add_reflection,   # expose as callable fns
                        MemoryPlanner.recall_reflection,
                        MemoryPlanner.reflection_prompt]

lm     = dspy.LM("openai/o3",
                 system = "You are a helpful Knowledge‑Graph instructor. "
                          "Use tools judiciously, think step‑by‑step, "
                          "but reveal only the final answer.")

chat_agent = dspy.StructuredAgent(lm=lm, tools=tools)   # planner/playground


⸻

1  Interactive widget cell

# Cell 1  – UI loop

# widgets
inp  = Textarea(placeholder="Ask about ontologies, JSON‑LD, SHACL…",
                layout=Layout(width='100%', height='80px'))
send = Button(description="Send", button_style='success')
out  = Output(layout=Layout(border='1px solid gray', height='400px', overflow='auto'))
box  = VBox([inp, send, out])
display(box)

conversation = []               # keeps (role,content) tuples for display

def render():
    out.clear_output()
    with out:
        for role, msg in conversation:
            display(Markdown(f"**{role}:** {msg}"))

def handle_click(_):
    q = inp.value.strip()
    if not q: return
    inp.value = ""
    conversation.append(("Instructor", q))
    render()

    # === call the agent ===
    result = chat_agent(q)                 # dict with "answer", "trace", etc.
    reply  = result.get("answer", str(result))

    # store a reflection if user explicitly says: remember: ...
    if q.lower().startswith("remember:"):
        note = q.split("remember:", 1)[1].strip()
        mem.add(text=note, tags=["manual"])
        reply = f"Got it – stored: {note}"

    # attach provenance hash & timestamp
    ts   = datetime.datetime.utcnow().isoformat()
    hash = uuid.uuid4().hex[:8]
    conversation.append(("Agent", f"{reply}\n\n<sub>{ts} • {hash}</sub>"))
    render()

send.on_click(handle_click)

Run the first two cells and you have a simple scrollable chat window inside the notebook.
All tool calls, SHACL validations, new triples, etc. happen under the hood; the agent’s hidden chain‑of‑thought is stored as hashed telemetry (see earlier answer).

⸻

### How to teach the agent on‑the‑fly
	•	Manual note – type
remember: in Wikidata, wdt:P1476 is the title property
→ reflection stored with tags=["manual"].
	•	Ask it to ingest docs –
Load https://www.w3.org/TR/json-ld11/ and store it
→ agent should call FetchDoc + IngestDoc, then you can ask
What does @container "@set @index" mean?.

⸻

2  Persist & reload memory between notebook sessions

# Cell 2  – save / load helpers
def save_graph(path="memory.nq"):
    graph.serialize(path, format="nquads")
def load_graph(path="memory.nq"):
    if os.path.exists(path):
        graph.parse(path, format="nquads")

Call save_graph() at the end of a teaching session; call load_graph() at the top next time.

⸻

3  Optional polish

want	add
Pretty markdown	wrap long answers in display(Markdown(reply))
Lat‑ency display	measure t0=now() before chat_agent(q), display seconds under reply
Tool traces	append result["trace"] collapsed in <details> HTML
Streaming output	switch to ipywidgets.Output + incremental writes


⸻

4  Why this fits your whole stack
	•	DSPy gives the planner & prompt optimisation.
	•	SIMBA can still mutate prompts offline—UI calls exactly the same agent object that SIMBA returns.
	•	Semantic memory (ReflectionStore) records every “lesson” inside the graph; no SQLite required.
	•	Thinking model o3 uses the scratch‑pad internally, hidden from UI, hashed into telemetry.

You now have a live notebook chat where you act as instructor, teach new triples or best‑practices, and the agent immediately incorporates them into future responses.

Happy teaching 🧑‍🏫—feel free to ping back if you need the optional polish snippets or run into widget quirks.