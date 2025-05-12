<!--
  cogitarelink-dspy: DSPy Agent Integration Overview
-->
# Cogitarelink-dspy Experiment: Agent Overview

This document outlines our goals and approach for integrating the Cogitarelink framework as tools within a DSPy-based agent. The aim is to prototype how Cogitarelink can be driven by agentic workflows, identify gaps, and evolve the framework for better agent support.

## Background

- **Cogitarelink**: A Python library for Linked Data navigation and AI-guided workflows.
- **DSPy**: A declarative framework for building modular AI systems by programming LLM behavior.

We will wrap Cogitarelink CLI/API calls as DSPy tools to compose agent workflows that can query, register, and manipulate linked-data contexts.

## Objectives

1. **Expose Cogitarelink functionality as DSPy tools**
   - List available tools (`list_tools`)
   - Run arbitrary Cogitarelink CLI tools (`run_tool`)
   - Additional domain-specific helpers (e.g., `retrieve_entity`, `create_entity`)

2. **Build a minimal DSPy agent**  
   - Configure a predictor (e.g., OpenAI or local LM)  
   - Register simple tool signatures  
   - Demonstrate agent-driven tasks (tool listing, registering vocabularies, generating examples)

3. **Iterate and identify framework enhancements**  
   - Observe pain points in CLI/API design  
   - Add helper functions or new tool hooks in Cogitarelink  
   - Refine tool schemas and input/output types for robust agent integration

## Getting Started

1. Install dependencies with **uv**:
   ```bash
   cd cogitarelink-dspy
   uv add "dspy>=0.1"
   uv add "cogitarelink@git+https://github.com/LA3D/cogitarelink.git@main"
   uv sync
   ```

2. Configure DSPy:
   ```python
   import dspy
   from dspy import OpenAIPredictor

   predictor = OpenAIPredictor(api_key="YOUR_OPENAI_API_KEY")
   dspy.configure(predictor=predictor)
   ```

3. Scaffold agent and tools:
   ```python
   from dspy import tool, Agent
   from cogitarelink.cli.agent_cli import AgentCLI

   @tool(name="list_tools")
   def list_tools() -> str:
       return AgentCLI().list_tools()

   @tool(name="run_tool")
   def run_tool(tool: str, args: dict) -> str:
       return AgentCLI().run_tool(tool, args)

   agent = Agent(
       tools=[list_tools, run_tool],
       predictor=predictor,
       verbose=True
   )
   ```

4. Run a simple query:
   ```python
   result = agent.run("List all available Cogitarelink tools")
   print(result)
   ```

## References

- DSPy documentation: https://github.com/LA3D/cogitarelink/blob/main/llmstxt/dspy.md
- Cogitarelink integration guide: https://github.com/LA3D/cogitarelink/blob/main/docs/framework_integration.md
- Deprecated DSPy examples: `cogitarelink/deprecated/code`

## Next Steps

- Expand tool wrappers to cover key CLI commands (e.g., `register_earth616`, `compose_context`, `generate_earth616_example`).
- Define structured input/output schemas for DSPy tools.
- Automate test workflows to validate agent-driven scenarios.
- Propose API changes in Cogitarelink to streamline agent interactions.