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
    """Test that the hello agent works with the Echo tool."""
    agent = make_hello_agent()
    message = "Hello, DSPy!"
    result = agent(message)
    assert result == message, f"Expected '{message}', got '{result}'"