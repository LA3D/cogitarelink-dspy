"""Test the Hello-World DSPy Agent."""

import pytest
from cogitarelink_dspy.core import Echo, make_hello_agent

def test_echo_tool():
    """Test that the Echo tool returns the input message."""
    echo = Echo()
    message = "Hello, DSPy!"
    result = echo.forward(message)
    assert result == message, f"Expected '{message}', got '{result}'"

def test_hello_agent():
    """Test that the hello agent works with the Echo tool and LLM."""
    agent = make_hello_agent()
    message = "Hello, DSPy!"
    result = agent(message)
    
    # Check that the echo result matches the input
    assert result["echo_result"] == message, f"Expected echo result '{message}', got '{result['echo_result']}'"
    
    # Check that we got an LLM response
    assert "llm_response" in result, "LLM response missing from result"
    assert result["llm_response"], "LLM response is empty"
    
    # Check that we got reasoning
    assert "reasoning" in result, "Reasoning missing from result"