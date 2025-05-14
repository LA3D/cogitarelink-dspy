#!/usr/bin/env python
"""
Integration test for Cogitarelink DSPy Memory tools.

This script tests the DSPy compilation process for memory components.
"""

import os
import sys
import json
import pytest
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

# DSPy and Cogitarelink imports
import dspy
from dspy.teleprompt import BootstrapFewShot

from cogitarelink_dspy.wrappers import get_tools, get_tool_by_name
from cogitarelink_dspy.components import COMPONENTS

# Path to the devset
DEVSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "devset_memory.jsonl")

def load_devset() -> List[Dict[str, Any]]:
    """Load the memory devset from JSONL."""
    with open(DEVSET_PATH, 'r') as f:
        return [json.loads(line) for line in f]

def tool_match(pred: Dict[str, Any], sample: Dict[str, Any]) -> bool:
    """Metric function: check if expected tool is in the trace."""
    return sample["exp_tool"] in pred.get("trace", [])

class TestMemoryCompilation:
    """Test the DSPy compilation process for memory components."""
    
    @pytest.fixture
    def mock_graphmanager(self):
        """Create a mock GraphManager for testing."""
        with patch('cogitarelink.core.graph.GraphManager') as mock:
            instance = mock.return_value
            # Configure the mock
            instance.query.return_value = []
            instance.ingest_entity = MagicMock()
            yield instance
    
    @pytest.fixture
    def memory_tools(self):
        """Get all memory-related DSPy tools."""
        all_tools = get_tools()
        return [t for t in all_tools if t.__name__ in ["AddReflection", "RecallReflection", "ReflectionPrompt"]]
    
    def test_memory_tools_exist(self, memory_tools):
        """Test that memory tool wrappers were successfully generated."""
        tool_names = [t.__name__ for t in memory_tools]
        assert "AddReflection" in tool_names
        assert "RecallReflection" in tool_names
        assert "ReflectionPrompt" in tool_names
    
    def test_tool_signatures(self, memory_tools):
        """Test that memory tools have correct signatures."""
        for tool in memory_tools:
            if tool.__name__ == "AddReflection":
                assert "text" in tool.signature.parameters
                assert "tags" in tool.signature.parameters
            elif tool.__name__ == "RecallReflection":
                assert "limit" in tool.signature.parameters
                assert "tag_filter" in tool.signature.parameters
            elif tool.__name__ == "ReflectionPrompt":
                assert "limit" in tool.signature.parameters
    
    def test_devset_format(self):
        """Test that devset is correctly formatted."""
        devset = load_devset()
        assert len(devset) >= 3, "Devset should have at least 3 examples"
        
        for sample in devset:
            assert "q" in sample, "Sample missing query field"
            assert "exp_tool" in sample, "Sample missing expected tool field"
    
    @pytest.mark.skipif(not os.path.exists(DEVSET_PATH), reason="Devset file not found")
    def test_bootstrap_configuration(self, mock_graphmanager):
        """Test bootstrap configuration with the memory devset."""
        devset = load_devset()
        
        # Create a simple test agent using the memory components
        class MemoryPlanner(dspy.Module):
            def __init__(self):
                super().__init__()
                # Get all memory tools
                self.add_reflection = get_tool_by_name("AddReflection")()
                self.recall_reflection = get_tool_by_name("RecallReflection")()
                self.reflection_prompt = get_tool_by_name("ReflectionPrompt")()
                
            def forward(self, query):
                trace = []
                
                # Simple keyword matching for dispatching
                if "remember" in query.lower():
                    note_id = self.add_reflection(text=query)
                    trace.append("AddReflection")
                    return {"response": f"Added reflection with ID: {note_id}", "trace": trace}
                
                elif "what" in query.lower() and "?" in query:
                    notes = self.recall_reflection(limit=3)
                    trace.append("RecallReflection")
                    return {"response": f"Found {len(notes)} reflections", "trace": trace}
                
                elif "inject" in query.lower() or "prompt" in query.lower():
                    formatted = self.reflection_prompt(limit=5)
                    trace.append("ReflectionPrompt")
                    return {"response": f"Formatted {formatted.count('•')} notes for prompting", "trace": trace}
                
                return {"response": "I don't know how to handle that query", "trace": trace}
        
        # Create the agent
        planner = MemoryPlanner()
        
        # Set up the bootstrap trainer
        trainer = BootstrapFewShot(devset=devset, metric=tool_match)
        
        # Only test the trainer configuration, don't actually compile
        # (full compilation would require LLM access)
        assert trainer.devset == devset
        assert trainer.metric == tool_match
        
        # Test prediction on the devset without compilation
        for sample in devset:
            pred = planner(sample["q"])
            assert "trace" in pred
            assert sample["exp_tool"] in pred["trace"], f"Expected {sample['exp_tool']} in trace for: {sample['q']}"

    @pytest.mark.skipif(True, reason="Requires LLM access, skipping by default")
    def test_full_compilation(self, mock_graphmanager):
        """Test full compilation with the memory devset.
        
        Note: This test is skipped by default as it requires LLM access.
        Remove the skipif decorator to run this test with a real LLM.
        """
        devset = load_devset()
        
        # Define a simple memory planner
        class MemoryPlanner(dspy.Module):
            def __init__(self):
                super().__init__()
                # Get all memory tools
                self.add_reflection = get_tool_by_name("AddReflection")()
                self.recall_reflection = get_tool_by_name("RecallReflection")()
                self.reflection_prompt = get_tool_by_name("ReflectionPrompt")()
                
            def forward(self, query):
                # Full compilation would optimize this decision logic using an LLM
                trace = []
                
                # Set up a default response
                response = "I'm not sure how to handle that query"
                
                return {"response": response, "trace": trace}
        
        # Create the compiler
        trainer = BootstrapFewShot(devset=devset[:2], metric=tool_match)
        
        # Configure the search space for optimization
        search_space = {
            "RecallReflection.limit": [3, 5, 10],
            "ReflectionPrompt.limit": [3, 5, 10]
        }
        
        # Create the base planner
        planner = MemoryPlanner()
        
        # Note: We're not actually running compile() since it requires LLM access
        # This would be the code to run compilation:
        # optimized = dspy.compile(planner, trainer,
        #                         num_iterations=2,
        #                         search_space=search_space)
        
        # Instead, we'll just test the configuration
        for sample in devset:
            result = planner(sample["q"])  
            assert "trace" in result


if __name__ == "__main__":
    # Initialize pytest fixtures
    test = TestMemoryCompilation()
    graphmanager = test.mock_graphmanager.__get__(test, TestMemoryCompilation)
    memory_tools = test.memory_tools.__get__(test, TestMemoryCompilation)
    
    # Run tests
    test.test_memory_tools_exist(memory_tools)
    test.test_tool_signatures(memory_tools)
    test.test_devset_format()
    test.test_bootstrap_configuration(graphmanager)
    
    print("All tests completed successfully!")