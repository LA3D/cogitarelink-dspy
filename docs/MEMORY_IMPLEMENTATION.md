# Reflection Memory Implementation

This document summarizes the implementation of the Reflection Memory system for Cogitarelink DSPy agents.

## Overview

The Reflection Memory system enables agents to store and retrieve "lessons learned" as JSON-LD entities with proper semantic modeling and provenance tracking. This implementation fully integrates with the Cogitarelink graph system and DSPy's declarative AI framework.

## Components

### 1. ReflectionStore

The core memory component that provides:
- Persistent storage of reflections with timestamps and tags
- Retrieval mechanisms with filtering and sorting
- Formatting utilities for prompt injection

### 2. DSPy Tool Integration

Three tools have been exposed to the DSPy agent framework:

| Tool | Purpose | Method | Parameters |
|------|---------|--------|------------|
| AddReflection | Store new reflections | add | text: str, tags: list=None |
| RecallReflection | Retrieve reflection notes | retrieve | limit: int, tag_filter: str=None |
| ReflectionPrompt | Format notes for prompting | as_prompt | limit: int |

### 3. JSON-LD Schema

Reflections are stored using the following schema:
```json
{
  "@id": "urn:uuid:{unique-id}",
  "@type": "https://w3id.org/cogitarelink#ReflectionNote",
  "text": "The reflection content",
  "tags": ["tag1", "tag2"],
  "dateCreated": "ISO timestamp"
}
```

Each entity is stored with full provenance tracking using `wrap_patch_with_prov`.

## Evaluation Harness

A complete evaluation framework has been implemented:

1. **Devset**: `tests/devset_memory.jsonl` contains example queries targeting each memory operation
2. **Unit Tests**: `tests/test_memory.py` provides comprehensive unit tests for all components
3. **DSPy Compilation Tests**: `tests/integration/test_memory_compile.py` tests the DSPy integration and compilation

The test suite covers:
- Storage and retrieval of reflections
- Tag-based filtering
- Prompt formatting
- Tool registry integration
- DSPy wrapper generation

## Training and Optimization

To train and optimize the MemoryPlanner’s tool‐selection logic, we recommend using DSPy’s experimental `SIMBA` prompt optimizer. `SIMBA` efficiently searches over few‐shot examples and prompt templates to maximize a user‐defined metric.

1. Load your development set (as a trainset):
```python
from cogitarelink_dspy.nbs.memory_training import load_devset
trainset = load_devset()
```

2. Instantiate the base planner:
```python
from cogitarelink_dspy.nbs.memory_training import MemoryPlanner
planner = MemoryPlanner(graph_manager=...)
```

3. Define your metric function:
```python
from cogitarelink_dspy.nbs.memory_training import tool_match
```

4. Optimize with SIMBA:
```python
import dspy
simba = dspy.SIMBA(metric=tool_match, max_steps=20, max_demos=5)
optimized_planner = simba.compile(
    student=planner,
    trainset=trainset,
    seed=42,       # optional for reproducibility
)
```

5. Save the optimized planner:
```python
from cogitarelink_dspy.nbs.memory_training import save_optimized_planner
save_optimized_planner(optimized_planner)
```

Pre‐trained planners are saved under `cogitarelink_dspy/optimized/` for distribution and downstream use. Optionally, you can supply a separate validation set (`valset=`) to `simba.compile()` or adjust `max_steps`/`max_demos` to control the search budget.

## Success Criteria

The implementation satisfies the following success criteria:
1. **Tool Selection**: Evaluation tests verify that the appropriate memory tool is selected based on queries
2. **Semantic Integration**: Reflections are stored as proper JSON-LD entities with types and references
3. **Declarative Interface**: All memory capabilities are exposed as DSPy tools with proper signatures
4. **Optimization**: The DSPy compilation harness enables parameters to be optimized, like retrieval limits

## Usage Examples

**Adding a Reflection**
```python
reflection_store.add(
    text="The wdt:P1476 property in Wikidata represents the title of a work",
    tags=["wikidata", "property", "title"]
)
```

**Retrieving Reflections**
```python
notes = reflection_store.retrieve(limit=3, tag_filter="wikidata")
```

**Formatting for Prompts**
```python
prompt_text = reflection_store.as_prompt(limit=5)
# Returns:
# • The wdt:P1476 property in Wikidata represents the title of a work
# • Additional notes here...
```

**Using the Optimized Memory Planner**
```python
from cogitarelink_dspy.optimized import load_model
from cogitarelink.core.graph import GraphManager

# Create a graph manager
graph_manager = GraphManager()

# Load the pre-trained memory planner
memory_planner = load_model("memory_planner.pkl")

# Update the graph manager in the loaded planner
memory_planner.reflection_store.graph = graph_manager

# Use the planner
result = memory_planner(q="Remember that owl:sameAs is used for identity statements")
print(result["response"])
```

## Next Steps

1. **Performance Optimization**: Fine-tune the retrieval mechanism for larger databases
2. **Embeddings**: Add vector embedding support for semantic similarity retrieval
3. **Temporal Decay**: Implement importance weighting based on recency and use frequency
4. **Integration with Core Agent**: Fully integrate with agent planning and reasoning cycles