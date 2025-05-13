# Reflection Memory: Implementation & Evaluation Plan

This document describes a Cogitarelink-native design for a semantic Reflection Memory layer, fully integrated into DSPy’s declarative AI framework. We leverage existing JSON-LD, GraphManager, and DSPy tool-wrapper machinery to persist, retrieve, and evaluate agent “lessons learned.”

## 1. Schema & Registry Extension

- Define `ReflectionNote` in the vocabulary registry (`prefix: clref → https://w3id.org/cogitarelink#`):
  • `@id`, `@type: ReflectionNote`
  • `clref:text`: string
  • `clref:tags`: [str]
  • `schema:dateCreated`: ISO timestamp
  • Provenance via `cogitarelink.reason.prov.wrap_patch_with_prov`

## 2. ReflectionStore (cogitarelink_dspy/memory.py)

Implement:
```python
import datetime, uuid, dspy
from typing import List, Optional
from cogitarelink.core.graph import GraphManager
from cogitarelink.core.entity import Entity
from cogitarelink.reason.prov import wrap_patch_with_prov

REFLECTION_GRAPH = "urn:agent:reflections"
REFLECTION_TYPE  = "https://w3id.org/cogitarelink#ReflectionNote"

class ReflectionStore:
    def __init__(self, graph: GraphManager):
        self.graph = graph

    def add(self, text: str, tags: Optional[List[str]] = None) -> str:
        note_id = f"urn:uuid:{uuid.uuid4()}"
        now     = datetime.datetime.utcnow().isoformat()
        content = {
            "@id": note_id,
            "@type": REFLECTION_TYPE,
            "text": text,
            "tags": tags or [],
            "dateCreated": now
        }
        ent = Entity(vocab=["clref","schema"], content=content)
        with wrap_patch_with_prov(
            self.graph, source="urn:agent:self",
            agent="urn:agent:self", activity="urn:agent:addReflection"
        ):
            self.graph.ingest_entity(ent)
        return note_id

    def retrieve(self, limit: int = 5, tag_filter: Optional[str] = None) -> List[Entity]:
        """Fetch up to `limit` most recent notes, optionally filtering by tag."""
        # Use graph.query or SPARQL to select ?s by type & tags, ORDER BY dateCreated
        ...

    def as_prompt(self, limit: int = 5) -> str:
        notes = self.retrieve(limit)
        return "\n".join(f"• {e.content['text']}" for e in notes)
```

## 3. DSPy Tool Exposure

In `cogitarelink_dspy/components.py` add:
```python
"AddReflection": {
  "layer":"Utility","tool":"AddReflection",
  "doc":"Persist a reflection into semantic memory",
  "calls":"add(text:str, tags:list=None)->str",
  "module":"cogitarelink_dspy.memory"
},
"RecallReflection": {
  "layer":"Utility","tool":"RecallReflection",
  "doc":"Retrieve recent reflection notes",
  "calls":"retrieve(limit:int, tag_filter:str=None)->list",
  "module":"cogitarelink_dspy.memory"
},
"ReflectionPrompt": {
  "layer":"Utility","tool":"ReflectionPrompt",
  "doc":"Format recent notes for prompt injection",
  "calls":"as_prompt(limit:int)->str",
  "module":"cogitarelink_dspy.memory"
}
```

DSPy wrappers will auto-generate Modules for these tools.

## 4. Declarative Evaluation Harness

Create `tests/devset_memory.jsonl`:
```jsonl
{"q":"Remember that wdt:P1476 is title","exp_tool":"AddReflection"}
{"q":"What’s the Wikidata title property?","use_memory":true,"exp_tool":"RecallReflection"}
{"q":"Inject notes into system prompt","exp_tool":"ReflectionPrompt"}
```
Metric:
```python
def tool_match(pred, sample):
    return sample["exp_tool"] in pred.get("trace", [])
```

## 5. Compile-Time Optimization

```python
from dspy.teleprompt import BootstrapFewShot
trainer = BootstrapFewShot(devset=DEVSET, metric=tool_match)
agent   = dspy.compile(planner, trainer,
                      num_iterations=5,
                      search_space={
                        "RecallReflection.limit":[3,5,10],
                        "ReflectionPrompt.template":[...]
                      })
```

## 6. Success Criteria & Ablations

1. Agent without memory vs. random vs. semantically filtered recall.
2. QA accuracy lift on memory-dependent queries.
3. Coherence (semantic F1) of `ReflectionPrompt` output.
4. `% of devset entries where exp_tool appears in trace`.

With this plan we scaffold `memory.py`, update COMPONENTS, regenerate wrappers, author the devset, and run pytest + `dspy.compile` to validate end-to-end behavior.