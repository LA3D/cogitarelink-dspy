 # Component Catalog Implementation Issues

 This document captures the discrepancies, bugs, and missing features identified in the current
 component-catalog notebooks and auto-generated DSPy wrappers.  Addressing these will align the
 wrappers with the real `cogitarelink` APIs, improve robustness, and enable effective testing.

 ## 1. Core Agent (`nbs/00_core.ipynb` → `cogitarelink_dspy/core.py`)
- Ensure consistent use of `dspy.configure(lm=...)` as per the current DSPy API; remove any legacy references to `dspy.settings.configure`.
- `HelloAgent` class missing a `signature` attribute; DSPy cannot introspect inputs/outputs without it.
- `HelloAgent` lacks a class docstring to describe its purpose in generated documentation.
- `HelloAgent.forward` has no type annotations on `message` or the returned dict, reducing clarity and safety.
- Tools list is hard-coded (`self.tools = [Echo()]`)—should dynamically pull from generated wrappers.

 ## 2. Component Registry (`nbs/01_components.ipynb` → `cogitarelink_dspy/components.py`)
- Missing import paths: `COMPONENTS` entries omit a `module` or `import_path` key, so wrappers cannot locate real implementations.
- `calls` strings do not support optional return types (`-> return_type`), leading to imprecise DSPy signatures.
- Helper functions `get_tools_by_layer` and `list_layers` lack Python type hints on parameters and return values.
- `validate_component_registry` only checks for presence of required fields; should also validate that
  - each `tool` value is a valid Python identifier,
  - `calls` strings can be successfully parsed,
  - no duplicate layer/tool mappings.
- Notebook-only asserts: core validation logic lives in notebooks rather than standalone pytest tests.

 ## 3. Wrapper Generator (`nbs/02_wrappers.ipynb` → `cogitarelink_dspy/wrappers.py`)
- **Closure capture bug**: inner class `ToolWrapper.forward` captures loop variable `meta`, so every wrapper ends up bound to the last component.
- `parse_signature` is too simplistic: does not handle defaults, nested commas, or complex type hints (e.g. `List[str]`).
- Stub `forward` methods only print and return dummy strings; must import the real function/class via `meta['module']` and invoke it with `**kwargs`.
- Signature attachment: ensure `ToolWrapper.signature` is actually set on the exported class, not just inside the class body.
- Layer grouping via regex on docstrings is brittle; better to assign a `layer` attribute on each wrapper class at creation time.
- Dead import: `import inspect` is unused and should be removed.
- Lazy initialization in `get_tools()` can lead to stale state on repeated imports; consider explicit factory functions instead.

 ## 4. General Testing & Documentation
- No end-to-end (smoke) tests instantiate wrappers and call `.forward(...)` against real `cogitarelink` implementations.
- COMPONENTS entries and generated wrappers still refer to placeholder tool names—must be updated to match the real package API (e.g. `ContextProcessor`, `GraphManager.query`, `sign`, `verify`).
- Missing documentation of new fields (`module`, return types) in `COMPONENTS` so maintainers understand how wrappers are wired.
- Add standalone pytest modules to cover:
  - Validation of the `COMPONENTS` dict (including module resolution).
  - Wrapper generation factory function correctness (one class per component, correct name, doc, signature, and bound implementation).
  - Actual invocation of each generated wrapper against the underlying Cogitarelink functions.