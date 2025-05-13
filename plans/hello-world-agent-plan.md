<!--
  Plan document for the Hello-World DSPy Agent demo
  Created: 2025-05-13
-->
# Hello-World DSPy Agent Plan

This document tracks the high-level plan for building and testing a minimal "Hello-World" DSPy agent
in the `nbs/00_core.ipynb` notebook, verifying our development workflow with nbdev and pytest.

## Objectives
1. Validate the DSPy environment and package installation.
2. Demonstrate a basic DSPy agent that echoes or computes a simple input.
3. Exercise the nbdev export pipeline (`nbdev_export`) and integration tests.

## Scope
- A minimal `Echo` (or simple compute) tool implemented as a `dspy.Module` in the notebook.
- A factory function `make_hello_agent` returning a `dspy.StructuredAgent` with the toy tool.
- A demonstration cell showing agent interaction in the notebook.
- A pytest test under `tests/` to assert the agent’s behavior.

## Notebook (`nbs/00_core.ipynb`) Changes
1. Update title and description to “Hello-World DSPy Agent”.
2. Add an **Imports** cell:
   - `import dspy`
   - (Optional) display or configure a dummy LLM (e.g. local/unmocked).
3. Define and export a trivial tool:
   ```python
   #| export
   class Echo(dspy.Module):
       """Echoes the input message back."""
       def forward(self, message: str) -> str:
           return message
   ```
4. Create and export an agent factory:
   ```python
   #| export
   def make_hello_agent(llm=None) -> dspy.StructuredAgent:
       # use provided or default LLM
       agent = dspy.StructuredAgent(tools=[Echo()], lm=llm)
       return agent
   ```
5. Add a demo cell:
   - Instantiate the agent
   - Run `agent.predict(message="Hello, DSPy!")`
   - Display the trace and output
6. Ensure the hidden `nbdev_export()` cell remains at the bottom for publishing.

## Testing
1. Create `tests/test_hello_agent.py`:
   - Import `Echo` and `make_hello_agent` from the exported module.
   - Instantiate the agent and run a sample prompt.
   - Assert the returned value matches the input (e.g. "Hello, DSPy!").
2. Run `pytest` and verify the test passes.

## Validation
- Run `nbdev_export` to update the package sources.
- Execute `pytest` and `pre-commit run --all-files` to ensure no errors.

Once complete, this demo will confirm our DSPy/nbdev/testing workflow before proceeding
to the full Cogitarelink Semantic-Web agent implementation.