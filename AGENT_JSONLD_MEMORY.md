# Agentic Memory & Retrieval with JSON-LD Contexts

This document outlines how to build an agentic system around CogitareLink’s JSON-LD 1.1 context management, retrieval, and in-memory graph storage. By wrapping CogitareLink’s registry, composer, and retriever APIs as DSPy tools, you can give a language model structured, graph-backed memory and retrieval capabilities.

## 1. JSON-LD Context Registry

- **Location**: `cogitarelink/vocab/registry.py`
- **Key classes**:
  - `ContextBlock` (url, inline, derives_from + sha256 fingerprint)
  - `VocabEntry` (prefix, uris, context, versions, features)
  - `registry: _Registry` (lookup by prefix or URI alias)
- **API**: `registry[prefix].context_payload()` returns a cached `{ "@context": {...} }` dict.

## 2. Context Composer & Collision Resolver

- **Location**: `cogitarelink/vocab/composer.py` and `cogitarelink/vocab/collision.py`
- **Composer** merges multiple vocabularies into a safe `@context`:
  1. Load primary prefix context
  2. For each additional prefix, ask `resolver.choose(primary, prefix)` for a merge strategy
  3. Build combined `{ "@context": ... }`, optionally an array of contexts
  4. Inject JSON-LD 1.1 keywords via `support_nest` and `propagate`
- **Strategies**: property_scoped, nested_contexts, graph_partition, separate_graphs, etc.

## 3. Linked-Data Retrieval Tools

- **Location**: `cogitarelink/integration/retriever.py`
- **LODResource & LODRetriever**:
  - `retrieve(uri)` automatically picks a strategy (content negotiation, HTML analysis, direct Turtle, etc.)
  - Returns `success`, `content` (raw text), `format`, `headers`, plus parsed JSON-LD via `json_parse`/`rdf_to_jsonld`
- **Other helpers**:
  - `search_wikidata(query, limit, language)` → list of Wikidata entities
  - `rdf_to_jsonld(content, format, base_uri)` → JSON-LD conversion
  - `json_parse(content)` → recover broken JSON

## 4. Agent Memory & History

- **Location**: `cogitarelink/cli/cli.py`
- **AgentContext** stores:
  - `memory: Dict[str, Any]` via `remember(key, value)` / `recall(key)`
  - `history: List[actions]` via `context.log_action(tool, inputs, result)`
- **Agent** wraps `tools: ToolRegistry` and logs calls automatically in `run_tool` / `run_cached_tool`.

## 5. Exposing as DSPy Tools

Wrap these APIs as DSPy tools so that an LLM agent can call them in a ReAct loop.

```python
from dspy import tool
from cogitarelink.cli.agent_cli import AgentCLI
from cogitarelink.integration.retriever import LODRetriever, search_wikidata
from cogitarelink.vocab.registry import registry

cli = AgentCLI()
retriever = LODRetriever()

@tool(name="remember")
def remember(key: str, value: Any) -> None:
    return cli.context.remember(key, value)

@tool(name="recall")
def recall(key: str) -> Any:
    return cli.context.recall(key)

@tool(name="retrieve_resource")
def retrieve_resource(uri: str) -> Dict[str, Any]:
    return retriever.retrieve(uri)

@tool(name="search_wikidata")
def wikidata_search(query: str, limit: int = 10) -> list:
    return search_wikidata(query, limit, "en")

@tool(name="load_context")
def load_context(prefix: str) -> Dict:
    return {"context": registry[prefix].context_payload()}

@tool(name="compose_context")
def compose_context(prefixes: list[str], support_nest: bool = False, propagate: bool = True) -> Dict:
    from cogitarelink.vocab.composer import composer
    return {"context": composer.compose(prefixes, support_nest, propagate)}

@tool(name="apply_collision")
def apply_collision(data: dict, prefixes: list[str]) -> Dict:
    from cogitarelink.cli.vocab_tools import apply_collision_strategy
    return apply_collision_strategy(data, prefixes)
```

## 6. Agentic Workflow

1. **Instruction**: e.g. "Enrich this entity with Wikidata labels."
2. **Tool selection**: LLM picks `retrieve_resource(uri)` or `search_wikidata(query)`
3. **Fetch & parse**: JSON-LD is fetched and parsed
4. **Context binding**: `load_context(prefix)` + `compose_context([...])`
5. **Collision resolution**: `apply_collision(document, [prefixs])`
6. **Memory storage**: `remember(key, enriched_doc)`
7. **Recall**: Later steps can `recall(key)` to fetch stored graph fragments

## 7. Next Steps
- Define rich DSPy `Signature` types for each tool (structured inputs/outputs)
- Build a high-level `MemoryAgent` subclass of `dspy.Module` to orchestrate loops
- Prompt-optimize tool selection with DSPy’s SIMBA or MIPROv2
- Integrate evaluation metrics (e.g. retrieval accuracy, context coverage)

This architecture lets your agent maintain a graph-structured memory underpinned by JSON-LD, fetch external linked-data sources, and resolve vocabulary collisions in a principled, declarative way.