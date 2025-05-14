"""Tests for the memory module."""

import pytest
import json
from unittest.mock import MagicMock, patch
from typing import List

from cogitarelink.core.graph import GraphManager
from cogitarelink.core.entity import Entity

from cogitarelink_dspy.memory import (
    REFLECTION_GRAPH,
    REFLECTION_TYPE,
    ReflectionStore
)
from cogitarelink_dspy.components import COMPONENTS
from cogitarelink_dspy.wrappers import TOOLS

class TestReflectionStore:
    """Test the ReflectionStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_graph = MagicMock(spec=GraphManager)
        self.store = ReflectionStore(self.mock_graph)
        
    def test_add_reflection(self):
        """Test adding a reflection."""
        # Configure mock
        self.mock_graph.ingest_entity = MagicMock()
        
        # Execute
        note_id = self.store.add("Test reflection", ["test", "memory"])
        
        # Verify
        assert note_id.startswith("urn:uuid:")
        self.mock_graph.ingest_entity.assert_called_once()
        
        # Get the Entity that was passed to ingest_entity
        args, _ = self.mock_graph.ingest_entity.call_args
        entity = args[0]
        
        # Verify Entity properties
        assert isinstance(entity, Entity)
        assert entity.content["@id"] == note_id
        assert entity.content["@type"] == REFLECTION_TYPE
        assert entity.content["text"] == "Test reflection"
        assert set(entity.content["tags"]) == {"test", "memory"}
        assert "dateCreated" in entity.content
        
    def test_retrieve_reflections(self):
        """Test retrieving reflections."""
        # Configure mock to return some test triples for dateCreated
        test_triples = [
            ("note1", "http://schema.org/dateCreated", "2023-05-13T12:00:00"),
            ("note2", "http://schema.org/dateCreated", "2023-05-12T12:00:00")
        ]
        self.mock_graph.query.side_effect = lambda **kwargs: (
            test_triples if kwargs.get("pred") == "http://schema.org/dateCreated" else 
            [("note1", "http://schema.org/text", "Reflection 1")] if kwargs.get("subj") == "note1" else
            [("note2", "http://schema.org/text", "Reflection 2")]
        )
        
        # Execute
        notes = self.store.retrieve(limit=2)
        
        # Verify
        assert len(notes) == 2
        assert notes[0].content["text"] == "Reflection 1"
        assert notes[1].content["text"] == "Reflection 2"
        
    def test_retrieve_with_tag_filter(self):
        """Test retrieving reflections with tag filtering."""
        # Configure mock
        test_date_triples = [
            ("note1", "http://schema.org/dateCreated", "2023-05-13T12:00:00"),
            ("note2", "http://schema.org/dateCreated", "2023-05-12T12:00:00")
        ]
        
        def mock_query(**kwargs):
            if kwargs.get("pred") == "http://schema.org/dateCreated":
                return test_date_triples
            elif kwargs.get("pred") == "http://schema.org/tags":
                if kwargs.get("subj") == "note1":
                    return [("note1", "http://schema.org/tags", "test")]
                else:
                    return [("note2", "http://schema.org/tags", "other")]
            elif kwargs.get("pred") == "http://schema.org/text":
                if kwargs.get("subj") == "note1":
                    return [("note1", "http://schema.org/text", "Reflection 1")]
                else:
                    return [("note2", "http://schema.org/text", "Reflection 2")]
            return []
        
        self.mock_graph.query.side_effect = mock_query
        
        # Execute with tag filter
        notes = self.store.retrieve(limit=2, tag_filter="test")
        
        # Verify only note1 is returned (has the "test" tag)
        assert len(notes) == 1
        assert notes[0].content["text"] == "Reflection 1"
        
    def test_as_prompt(self):
        """Test the as_prompt method formats notes correctly."""
        # Mock retrieve to return test notes
        self.store.retrieve = MagicMock(return_value=[
            Entity(vocab=["clref","schema"], content={
                "@id": "note1",
                "@type": REFLECTION_TYPE,
                "text": "First reflection"
            }),
            Entity(vocab=["clref","schema"], content={
                "@id": "note2",
                "@type": REFLECTION_TYPE,
                "text": "Second reflection"
            })
        ])
        
        # Execute
        prompt = self.store.as_prompt(limit=2)
        
        # Verify
        assert prompt == "• First reflection\n• Second reflection"
        self.store.retrieve.assert_called_once_with(2)

def test_component_registry_has_memory_tools():
    """Test that the component registry includes memory tools."""
    memory_tools = ["AddReflection", "RecallReflection", "ReflectionPrompt"]
    for tool_name in memory_tools:
        assert tool_name in COMPONENTS, f"Missing memory tool: {tool_name}"
        
def test_memory_tool_wrappers_exist():
    """Test that memory tool wrappers were generated."""
    memory_tool_classes = ["AddReflection", "RecallReflection", "ReflectionPrompt"]
    tool_class_names = [tool.__name__ for tool in TOOLS]
    for class_name in memory_tool_classes:
        assert class_name in tool_class_names, f"Missing tool wrapper: {class_name}"

def test_devset_memory_exists():
    """Test that the devset_memory.jsonl file exists and has the expected format."""
    import os
    
    # Check file exists
    devset_path = os.path.join(os.path.dirname(__file__), "devset_memory.jsonl")
    assert os.path.exists(devset_path), "devset_memory.jsonl file does not exist"
    
    # Check file contents
    with open(devset_path, 'r') as f:
        lines = f.readlines()
    
    # There should be at least 3 entries
    assert len(lines) >= 3, "devset_memory.jsonl should have at least 3 entries"
    
    # Each line should be valid JSON
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            assert "q" in entry, f"Entry {i+1} is missing 'q' field"
            assert "exp_tool" in entry, f"Entry {i+1} is missing 'exp_tool' field"
        except json.JSONDecodeError:
            pytest.fail(f"Line {i+1} in devset_memory.jsonl is not valid JSON")