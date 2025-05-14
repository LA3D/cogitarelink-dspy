<!---
  Refactoring Roadmap for Cogitarelink-DSPy Agent
-->
# Refactoring Roadmap

This document captures the high-level, step-by-step refactoring roadmap to reorganize our Cogitarelink-DSPy integration into a structured, layered agent following Jeremy Howard’s approach.

## 0. Reflection & Telemetry Foundations
• Implement semantic memory and telemetry as first-class components before any agent layers:
  1. Add `cogitarelink_dspy/telemetry.py`:
     – Define `TelemetryStore(TelemetryEvent)` subclassing `ReflectionStore`, with `log(event_type, value, tool_iri, **kw)` that writes structured RDF events using `wrap_patch_with_prov`.
  2. Create `telemetry.context.jsonld` in `docs/` (or a context directory) mapping:
     – `cl:TelemetryEvent`: class
     – `schema:eventType`, `schema:quantity`, `schema:dateCreated`, `cl:relatesToTool`, `http:statusCode`.
  3. Extend `cogitarelink_dspy/components.py` with three memory tools:
     - `AddReflection`, `RecallReflection`, `ReflectionPrompt` (layer: Utility)
     - `LogTelemetry` → `TelemetryStore.log` (layer: Utility)
  4. Auto-generate DSPy wrappers for `ReflectionStore` and `TelemetryStore` via your tool generator; export as `TOOLS`.
  5. Write smoke-tests in `tests/test_memory.py` and `tests/test_telemetry.py`:
     – Verify `add`/`retrieve` and `as_prompt` on `ReflectionStore`.
     – Verify `log` writes a `TelemetryEvent` in the graph and captures `eventType`/`quantity`.
  6. Ensure default agent startup loads `99_reflection_seed.jsonl` and makes `TelemetryStore` available in `pipelines.py` for every wrapper.

## 0.5 Advanced JSON-LD & Doc-Ingestion
• Leverage JSON-LD 1.1 containers and indexes in semantic memory and telemetry:
  - Enforce array form for lists: wrap every `tags` or `schema:keywords` field as `{"@set": [...tags...]}` to avoid compaction to scalars.
  - Enable tag-based indexing: extend your `clref` context (e.g. in `vocab/contexts/clref.context.jsonld`) with:
    ```jsonld
    "tags": { "@id": "schema:keywords", "@container": ["@set","@index"] }
    ```
    so you can fast-SPARQL lookup like:
    ```sparql
    GRAPH <urn:agent:reflections> {
      ?note schema:keywords/sparql:timeouts ?_ .
    }
    ```
  - For telemetry, use similar indexed containers on date buckets:
    ```jsonld
    "daily": { "@container": ["@index","@set"] }
    ```
  - Provide JSON-LD framing documents to select the most recent N reflections without exposing private CoT hashes.

• Minimal patch in `ReflectionStore.add`:
  ```python
  content = {
      "@id": note_id,
      "@type": REFLECTION_TYPE,
      "text": text,
      "tags": { "@set": tags or [] },
      "dateCreated": now
  }
  ```
  For tag-indexed variant:
  ```python
  tag_dict = {t:[True] for t in tags}
  content["tags"] = tag_dict  # triggers @index lookup
  ```

• Ingest external reference docs via two new DSPy tools:
  1. **FetchDoc**(url:str, type:str='markdown'|'html'|'pdf') -> text
     - Fetches raw document from URL.
     - Logs a `TelemetryEvent` with HTTP status and byte count.
  2. **IngestDoc**(text:str, url:str) -> graph_id:str
     - Splits text by headings, calls `Cogitarelink.entity_from_text()` via the LLM.
     - Stores each chunk as a `cl:DocChunk` in `urn:doc:<sha256>` with provenance via `wrap_patch_with_prov`.
  
  Add both to `cogitarelink_dspy/components.py`; your wrapper generator will auto-create DSPy modules.
  Write smoke-tests in `tests/test_doc_ingest.py` to verify fetching, ingesting, and provenance.

## 1. Consolidate a Single-Source COMPONENTS Registry
• In `cogitarelink_dspy/components.py`, enumerate **all** Cogitarelink modules/tools with entries:
  - **layer**: one of {Context, Ontology, Instances, Rules, Verification, Utility}
  - **tool**: public DSPy name
  - **module**: Python import path
  - **calls**: exact Python signature

## 2. Auto-Generate DSPy Wrappers from COMPONENTS
• In a notebook or new `cogitarelink_dspy/tools.py`, loop over COMPONENTS:
  1. Import the real object via `importlib` and `getattr`
  2. Use `inspect.signature()` to build a `dspy.Signature`
  3. Define `class ToolX(dspy.Module): signature = …; def forward(self, **kwargs): return real_fn(**kwargs)`
• Export the resulting list as `TOOLS` and remove any ad-hoc parsers or mock fallbacks.

## 3. Define StructuredAgents & System Prompt
• Create `cogitarelink_dspy/pipelines.py` and define:
  ```python
  SEM_WEB_SYSTEM = '''
  You are a Semantic-Web agent that reasons over these layers:
  1. Context      (JSON-LD context ops)
  2. Ontology     (vocab & retriever)
  3. Instances    (data via SPARQL)
  4. Rules        (SHACL/SPARQL validation & inference)
  5. Verification (sign & verify graphs)
  Always pick the highest layer sufficient to answer.
  '''
  ```
• Define two pipelines:
  - **HelloLOD**: a minimal subset of tools for quick LOD tasks.
  - **FullPlanner**: all tools from `TOOLS`.
  Both as `dspy.StructuredAgent(tools=…, lm=LM, system=SEM_WEB_SYSTEM)`.

## 4. Develop a Layered Curriculum Dataset
• Create `data/curriculum/` with staged JSONL files:
  - `00_context.jsonl`
  - `01_ontology.jsonl`
  - `02_rules.jsonl`
  - `03_instances.jsonl`
  - `04_content_negotiation.jsonl`
  - `05_wikidata_quirks.jsonl`
  - `06_validation.jsonl`           (SHACL validation tasks)
  - `07_materialisation.jsonl`      (SHACL inference tasks)
  - `99_reflection_seed.jsonl`      (initial memory reflections)

• Each JSONL row includes:
  ```json
  {
    "q":        "Your query text…",
    "exp_tool": "ExpectedToolName",
    "answer":   "Atomic output (IRI / bool / int / literal)",
    "tags":     ["stage","detail"],
    "remember": "optional reflection text",
    "params":   { /* e.g. graph_id, shapes_id, mode */ }
  }
  ```
• Guidelines:
  - 8–12 examples per stage; split 70% train / 15% dev / 15% test.
  - Include at least one negative/failure example per stage.
  - Tag thoroughly for future recall-driven prompts.
  - Seed reflections in `99_reflection_seed.jsonl` to prime MemoryPlanner.

# 5. Codex-Assisted Curriculum Authoring
• Lock one canonical JSONL schema: include this example at the top of every file:
  ```json
  {"q":"Human-readable query",
   "exp_tool":"ExpectedToolName",
   "answer":"Atomic output (IRI | bool | int | literal)",
   "tags":["stage","detail"],
   "remember":"optional reflection text",
   "params":{"graph_id":"g1","mode":"validate"}}
  ```
• Playbook for each stage file:
  1. Prompt Codex:
     "Output FIVE JSON objects, each on its own line, following the schema above. Stage = \"Content negotiation\". exp_tool ∈ [FetchURL, HeadRequest]."
  2. Append Codex output to `data/curriculum/04_content_negotiation.jsonl`.
  3. Smoke-test:
     ```python
     from utils import load_jsonl, curriculum_metric
     exs = load_jsonl("data/curriculum/04_content_negotiation.jsonl")
     for e in exs:
         assert curriculum_metric({"trace":[e["exp_tool"]], "response":e["answer"]}, e) > 0
     ```
  4. Human review and tweak as needed.
  5. Commit the validated rows.
• Let Codex write the metric:
  1. In a notebook cell, run one example manually:
     ```python
     pred = hello_lod("How many cats on Wikidata?")
     gold = {"exp_tool":"ExecuteSPARQL","answer":"5"}
     ```
  2. Ask Codex to generate `curriculum_metric(pred,gold)` that returns 1.2 on exact match.
  3. Drop its answer into `metrics.py` and verify on a few rows.
• Automate validation (`scripts/check_curriculum.py`):
  - Load every `data/curriculum/*.jsonl`.
  - Ensure required keys exist.
  - Call `curriculum_metric` on a dummy pred for each row to catch KeyErrors.
  - Integrate this script as a pre-commit hook or CI step.
• Interactive lesson harvesting:
  1. When you discover a new quirk, prompt Codex to draft a reflection row:
     "Create an AddReflection row capturing: 'PersonShape often fails because birthDate is missing.'"
  2. Append to `99_reflection_seed.jsonl` and call `AddReflection` on the live graph.
  3. Next SIMBA run sees both the reflection and a matching curriculum example.
• (Optional) Codex-driven grading:
  - For SHACL or complex outputs, have Codex analyse reports and return structured counts.
  - Feed those counts into `curriculum_metric` instead of raw string compare.
• Version control:
  - Commit all JSONL files in `data/curriculum/`.
  - Store helper scripts in `scripts/` and the prompt template in `scripts/curriculum_prompt.txt`.
  - Keep `metrics.py` and `tests/test_curriculum.py` under version control.

## 6. Training Loop: Bootstrap & SIMBA
• In `nbs/04_agent.ipynb` (or new notebook):
  ```python
  from dspy.teleprompt import BootstrapFewShot
  from dspy import SIMBA
  # Load curriculum splits
  train = load_jsonl("data/curriculum/*[0-5].jsonl", split="train")
  dev   = load_jsonl("data/curriculum/*[0-5].jsonl", split="dev")
  test  = load_jsonl("data/curriculum/*[0-5].jsonl", split="test")

  # Metric rewards correct tool routing, provenance, and memory usage
  def curriculum_metric(pred, gold):
      score = 1.0 if gold["exp_tool"] in pred.get("trace", []) else 0.0
      # +0.2 for correct count or provenance, +0.2 for storing expected memory
      if gold.get("answer_key") and pred.get(gold["answer_key"]) == gold.get("answer"): score += 0.2
      if gold.get("remember") and "AddReflection" in pred.get("trace", []):           score += 0.2
      return score

  # 1) Bootstrap few-shot on dev set
  bootstrap = BootstrapFewShot(devset=dev, metric=curriculum_metric)
  agent0    = dspy.compile(HelloLODWithMemory(), bootstrap, num_iterations=2)

  # 2) SIMBA fine-tune on train/dev
  simba     = SIMBA(metric=curriculum_metric, max_steps=24, batch_size=8)
  agent1    = simba.compile(agent0, trainset=train, devset=dev, seed=42)

  # 3) Final evaluation
  evaluate(agent1, test)
  ```

## 6. Plug in ReflectionStore (Memory)
• Wrap `AddReflection`, `RecallReflection`, `ReflectionPrompt` as DSPy tools and include in your agent’s tool list.
• Ensure `99_reflection_seed.jsonl` is loaded at boot so the agent starts with pre-seeded notes.
• Add curriculum rows with `remember` fields to teach reflection writing.
• Re-run SIMBA to optimise memory routing and content-driven reflections.

## 7. End-to-End Pipelines & Evaluation
• In `cogitarelink_dspy/pipelines.py` define:
  - `HelloLODWithMemory` (HelloLOD + memory tools)
  - `FullAgent` (the final optimized agent)
• Create `nbs/05_eval.ipynb` to:
  - Run the test split through both pipelines.
  - Compute exact-match + provenance bonus.
  - Benchmark 99th-percentile latency.

## 8. Clean-Up & Tests
• Default to `GraphManager(use_rdflib=False)` instead of mocks.
• Remove obsolete wrappers/tests for the old approach.
• Add unit tests for every generated tool in `tests/test_tools.py`.
• Ensure all nbdev exports and pytest pass.

## 9. Advanced Guardrails & Observability
Below are additional engineering guardrails inspired by the reflection and telemetry plan.

### 9.1 Multi-Tool Scenario Curriculum
• Create a new curriculum file `data/curriculum/08_multi_tool.jsonl` for end-to-end tasks that chain tools (e.g. `["RunSHACL","AddReflection","QueryInstances"]`).
• Extend your JSON schema: allow an optional field `exp_trace`: list of tool names in order.
• Update `curriculum_metric` to compare full `pred["trace"]` sequence against `gold.get("exp_trace")` when present, rewarding correct ordering.

### 9.2 Latency-Aware Metric & Caching Incentives
• Instrument every wrapper’s `forward()` to record elapsed milliseconds in `pred["stats"]["ms"]` and whether it was a cache hit.
• Extend `curriculum_metric` to subtract a small penalty (e.g. `0.01 * seconds`) from the score, encouraging cached/local ops.

### 9.3 Reflection Garbage Collection
• Define a new DSPy tool `SummarizeReflections(limit:int)`:
  – In `cogitarelink_dspy/components.py` add a registry entry for `SummarizeReflections` (layer Utility).
  – Auto-generate wrapper via your existing tool generator.
  – Implement `forward()` to take the oldest N notes, ask the LM for a 3-bullet summary, replace them in the graph.
• Add a stage in curriculum: `data/curriculum/09_reflection_gc.jsonl` with examples expecting this tool.
• Reward its use in `curriculum_metric` when memory size exceeds threshold.

### 9.4 Telemetry & In-Graph Observability
• Define a `telemetry.context.jsonld` mapping `cl:TelemetryEvent`, `schema:eventType`, `schema:quantity`, etc.
• Implement `TelemetryStore` (subclass of `ReflectionStore`) in `cogitarelink_dspy/telemetry.py` and register it as a DSPy tool.
• In every wrapper, log latency, cache hits, and errors via `TelemetryStore.log(...)` so events live in your semantic graph with provenance.

### 9.5 Golden-Run Replay & CI
• In your CI pipeline, capture every agent `pred` dict (compressed) into `artefacts/YYYY-MM-DD/`.
• Nightly, re-run `nbs/05_eval.ipynb` or a lightweight script to diff answers, latency, token use; fail on regressions.

### 9.6 Rapid Triage UI
• Build a minimal Streamlit or Gradio app in `scripts/triage_ui.py` that:
  – Loads new rows from `data/curriculum/*.jsonl`
  – Displays JSON with accept/reject buttons
  – Appends accepted rows to the file, reruns smoke-tests, or feeds rejects back to Codex for rewrite.

### 9.7 Safety Shim for Reflections
• Wrap `AddReflection` to first run a PII detector or regex filter, hash or redact sensitive tokens, then call the real tool.
• Add this guardrail as a separate DSPy tool or decorator so it’s versioned in COMPONENTS.

## 10. Ongoing Maintenance
• To add a new Cogitarelink module, append it to COMPONENTS and rerun your notebooks.
• To teach SIMBA new demos or docs, update `data/curriculum/*.jsonl` and recompile.