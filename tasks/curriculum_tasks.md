<!---
  Curriculum Tasks Playbook
-->
# Curriculum Tasks Playbook

This document details the human-in-the-loop, Codex-assisted workflow for authoring and validating the curriculum JSONL files that drive SIMBA training.

1. Lock a Canonical JSONL Schema
• Define this example as the first line in every curriculum file:
  ```json
  {"q":"Human-readable query",
   "exp_tool":"ExpectedToolName",
   "answer":"Atomic output (IRI | bool | int | literal)",
   "tags":["stage","detail"],
   "remember":"optional reflection text",
   "params":{}}
  ```
• `params` holds any tool arguments (e.g. graph_id, shapes_id, mode).

2. Codex-Assisted Authoring Loop
• Prompt template (in `scripts/curriculum_prompt.txt`):
  ```text
  You are drafting training examples for the Cogitarelink DSPy agent.
  Output FIVE JSON objects, each on its own line, following this schema:
  {q, exp_tool, answer, tags, params, remember?}
  Stage = "<stage name>".
  exp_tool must be one of [list of valid tools for this stage].
  ```
• Workflow per stage:
  1. Run helper `generate_batch(stage, n=5)` which reads the prompt template, fills in `<stage name>` and `<tool list>`, calls OpenAI, and appends parsed lines to `data/curriculum/{stage}.jsonl`.
  2. Parse & filter: the helper discards non-JSON lines and invalid rows.
  3. Smoke-test:
     ```python
     from utils import load_jsonl, curriculum_metric
     exs = load_jsonl(f"data/curriculum/{stage}.jsonl")
     for e in exs:
         assert curriculum_metric({"trace":[e["exp_tool"]], "response":e["answer"]}, e) > 0
     ```
  4. Human review: open the JSONL file, tweak phrasing, answers, remember notes.
  5. Commit changes.

3. Let Codex Generate the Metric
• In a notebook:
  ```python
  pred = HelloLODWithMemory().run("How many cats on Wikidata?")
  gold = {"exp_tool":"ExecuteSPARQL","answer":"5"}
  ```
• Prompt Codex to write `curriculum_metric(pred, gold)` returning 1.2 on exact match.
• Paste into `metrics.py` and verify by running on sample rows.

4. Automate Curriculum Validation
• Create `scripts/check_curriculum.py` that:
  - Loads every `data/curriculum/*.jsonl`.
  - Ensures each row has required keys (`q`, `exp_tool`, `answer`, `tags`, `params`).
  - Verifies `exp_tool` is in your master COMPONENTS list.
  - Calls `curriculum_metric` on a dummy pred for each row to catch KeyErrors.
  - Reports errors with file/line numbers.
• Run this script in CI or as a pre-commit hook to block invalid changes.

**2.5 Multi-Tool & Adversarial Examples**
• For end-to-end chaining tasks, create stage `08_multi_tool`:
  - In the prompt, ask for a field `exp_trace`: an ordered JSON array of tools to invoke.
  - Append rows to `data/curriculum/08_multi_tool.jsonl`.
  - Ensure `curriculum_metric` is updated to compare full `pred["trace"]` vs `exp_trace`.
• For adversarial/fuzzed inputs, create stage `09_adversarial`:
  - Prompt Codex with malformed contexts or headers that should fail.
  - Mark examples with an error taxonomy tag (e.g. `"tags":["adversarial","parseError"]`).
  - Append to `data/curriculum/09_adversarial.jsonl`.

**2.6 Reflection Garbage-Collection Examples**
• To teach SummarizeReflections, create stage `10_reflection_gc`:
  - In the prompt, ask for examples where `SummarizeReflections(limit)` should be invoked to compress old notes.
  - Include `exp_tool":"SummarizeReflections"` and `remember` fields summarizing the GC rationale.
  - Append to `data/curriculum/10_reflection_gc.jsonl`.

**2.7 JSON-LD Container & Indexing Examples**
• Create stage `11_jsonld_indexing`:
  - Prompt for examples requiring O(1) SPARQL lookup via indexed tags or JSON-LD framing to extract recent notes.
  - Use `exp_tool: "QueryInstances"` with a SPARQL path like `schema:keywords/sparql:timeouts` or a new `FrameReflections` tool.
  - Append to `data/curriculum/11_jsonld_indexing.jsonl`.

**2.8 External Document Ingestion Examples**
• Create stage `12_doc_ingestion`:
  - Examples using `FetchDoc` then `IngestDoc`:
    ```json
    {"q":"Load the JSON-LD 1.1 spec and store it",
     "exp_tool":"FetchDoc",
     "answer":"urn:doc:<sha256>",
     "params":{"url":"https://www.w3.org/TR/json-ld11/","type":"html"}}
    ```
    ```json
    {"q":"What does @container [\"@set\",\"@index\"] mean?",
     "exp_tool":"QueryInstances",
     "answer":"It defines an indexed set of values",
     "params":{"q":"SELECT ?def WHERE {...}"}}
    ```
  - Append both to `data/curriculum/12_doc_ingestion.jsonl`.

3. Interactive Lesson Harvesting
• When the agent encounters a new quirk during testing or ad-hoc runs:
  1. In a notebook cell, prompt Codex:
     "Create an AddReflection JSON row capturing: 'PersonShape often fails because birthDate is missing.'"
  2. Append the returned row to `data/curriculum/99_reflection_seed.jsonl`.
  3. Immediately call `AddReflection(text=..., tags=[...])` on your live agent to ingest the note.
• Next training/compile run includes that reflection seed.

6. (Optional) Codex-Driven Grading for Non-Atomic Outputs
• For complex tools (e.g. SHACL reports), have the wrapper return structured fields (`violations`, `warnings`, `report`).
• Prompt Codex to parse the raw report into those fields and return a dict.
• Extend `curriculum_metric` to compare numeric fields instead of raw strings.

7. Version Control & Repository Layout
• `data/curriculum/` – all staged JSONL files, 100% in git
• `scripts/` – helper scripts:
  - `generate_batch.py` (Codex prompt + append)
  - `check_curriculum.py` (CI validation)
  - `curriculum_prompt.txt` (prompt template)
• `metrics.py` – single source of truth for `curriculum_metric` and any stage-specific variants
• `tests/test_curriculum.py` – smoke-tests for key stages and metric behavior

By following this playbook, you get a tight draft→run→grade→revise loop that keeps your curriculum aligned with your metric and your agent’s real behavior.