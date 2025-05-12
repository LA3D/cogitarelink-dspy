#!/usr/bin/env python3
"""
Example: Build a DSPy-powered agent to drive CogitareLink tools.
See llmstxt/dspy-agent.md for patterns on building and optimizing agents.
"""
import os
import sys
from dotenv import load_dotenv
import dspy
from dspy import Signature, ChainOfThought, Prediction
from cogitarelink.cli.agent_cli import AgentCLI

def main():
    # Load environment (OPENAI_API_KEY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Configure the LLM
    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
    dspy.configure(lm=lm)

    # Instantiate CogitareLink CLI-backed agent
    cli = AgentCLI()

    # DSPy ReAct signature: pick a tool and its args
    # Define a ReAct signature: select the next tool and its args
    sig = Signature(
        "instruction, history, tools -> next_tool: str, tool_args: dict[str, Any]",
        "Given an instruction, past history, and available tools, select the next tool to call"
    )
    react = ChainOfThought(sig)

    # Build a simple module class
    class CogitareAgent(dspy.Module):  # noqa: D102
        def __init__(self, max_steps: int = 3):
            self.max_steps = max_steps
            self.react = react

        def forward(self, instruction: str) -> Prediction:
            history = []
            # List available tools with metadata
            # Retrieve available tools via CogitareLink
            tools = cli.run_tool("list_tools")
            last_result = None
            # Run a short ReAct loop
            for _ in range(self.max_steps):
                pred = self.react(
                    instruction=instruction,
                    history=history,
                    tools=tools
                )
                # Normalize tool name and call
                tool_name = pred.next_tool.strip('"').strip("'")
                args = pred.tool_args or {}
                # Execute via CogitareLink
                result = cli.run_tool(tool_name, **args)
                history.append({
                    "thought": pred.reasoning,
                    "action": tool_name,
                    "args": args,
                    "result": result,
                })
                last_result = result
            return Prediction(answer=last_result, history=history)

    # Instantiate and run
    agent = CogitareAgent()
    # Instruction from CLI or default
    instruction = sys.argv[1] if len(sys.argv) > 1 else "List all available CogitareLink tools"
    pred = agent(instruction)
    # Print final answer and trace
    print(pred.answer)
    for step in pred.history:
        print(f">>> {step['thought']}\n-> {step['action']}({step['args']}) => {step['result']}\n")

if __name__ == "__main__":
    main()