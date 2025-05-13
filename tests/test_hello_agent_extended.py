import pytest
import dspy
from typing import Dict, Any
from cogitarelink_dspy.core import make_hello_agent
from cogitarelink_dspy.wrappers import get_tools, group_tools_by_layer
from cogitarelink_dspy.components import list_layers

class TestHelloAgent:
    """Test suite for the HelloAgent implementation."""
    
    def test_agent_creation(self):
        """Test that we can create a HelloAgent instance."""
        agent = make_hello_agent()
        assert agent is not None, "Failed to create HelloAgent instance"
    
    def test_agent_has_tools(self):
        """Test that the agent has loaded tools from the registry."""
        agent = make_hello_agent()
        assert hasattr(agent, "tools"), "Agent missing tools attribute"
        assert len(agent.tools) > 0, "Agent has no tools loaded"
        
        # Compare with expected tool count from wrapper generator
        expected_tools = get_tools()
        assert len(agent.tools) == len(expected_tools), \
            f"Expected {len(expected_tools)} tools, agent has {len(agent.tools)}"
    
    def test_agent_has_layer_organization(self):
        """Test that the agent has tools organized by layer."""
        agent = make_hello_agent()
        assert hasattr(agent, "tools_by_layer"), "Agent missing tools_by_layer attribute"
        
        # Check that all expected layers are present
        expected_layers = list_layers()
        agent_layers = list(agent.tools_by_layer.keys())
        
        for layer in expected_layers:
            assert layer in agent_layers, f"Agent missing tools for layer: {layer}"
    
    def test_agent_response_structure(self):
        """Test that the agent returns responses with the expected structure."""
        agent = make_hello_agent()
        result = agent("Test message")
        
        # Check required fields in response
        required_fields = ["echo_result", "llm_response", "layer_used", "tool_used", "tool_result"]
        for field in required_fields:
            assert field in result, f"Agent response missing required field: {field}"
    
    def test_layer_detection(self):
        """Test that the agent correctly detects the semantic layer from messages."""
        agent = make_hello_agent()
        
        # Test layer detection with different messages
        test_cases = [
            ("Hello world", "Utility"),  # Default layer
            ("Load the JSON-LD context", "Context"),
            ("Can you fetch the FOAF ontology?", "Ontology"),
            ("Validate this triple", "Rules"),
            ("Store this in the graph", "Instances"),
            ("Verify the signature", "Verification")
        ]
        
        for message, expected_layer in test_cases:
            result = agent(message)
            assert result["layer_used"] == expected_layer, \
                f"For message '{message}', expected layer {expected_layer}, got {result['layer_used']}"
    
    def test_tool_selection(self):
        """Test that the agent selects appropriate tools based on the layer."""
        agent = make_hello_agent()
        
        # For each layer, send a message targeting that layer and check the selected tool
        for layer in agent.tools_by_layer:
            if not agent.tools_by_layer[layer]:
                continue  # Skip empty layers
                
            # Create a message mentioning the layer
            if layer == "Utility":
                message = "Hello world"  # Default layer
            else:
                message = f"Use a {layer.lower()} tool please"
                
            result = agent(message)
            assert result["layer_used"] == layer, \
                f"For message '{message}', expected layer {layer}, got {result['layer_used']}"
            
            # Verify selected tool belongs to the detected layer
            if result["tool_used"] != "None":
                tool_class = next((t for t in agent.tools if t.__name__ == result["tool_used"]), None)
                assert tool_class is not None, f"Selected tool {result['tool_used']} not found in agent tools"
                assert tool_class.layer == layer, \
                    f"Selected tool {result['tool_used']} belongs to layer {tool_class.layer}, expected {layer}"

if __name__ == "__main__":
    pytest.main()