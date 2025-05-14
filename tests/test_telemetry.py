"""Smoke tests for the TelemetryStore and LogTelemetry tool."""

import pytest
from unittest.mock import MagicMock

from cogitarelink.core.graph import GraphManager
from cogitarelink.core.entity import Entity

from cogitarelink_dspy.telemetry import (
    TELEMETRY_GRAPH,
    TELEMETRY_TYPE,
    TelemetryStore
)
from cogitarelink_dspy.components import COMPONENTS
from cogitarelink_dspy.wrappers import TOOLS

class TestTelemetryStore:
    """Test suite for logging telemetry events."""

    def setup_method(self):
        self.mock_graph = MagicMock(spec=GraphManager)
        self.store = TelemetryStore(self.mock_graph)

    def test_log_event(self):
        """Test that logging creates a TelemetryEvent entity."""
        # Stub ingest_entity
        self.mock_graph.ingest_entity = MagicMock()
        # Log an event with extra data
        event_id = self.store.log(
            event_type="httpRequest",
            value=123,
            tool_iri="urn:tool:TestTool",
            statusCode=200,
            extraField="extra"
        )
        # ID should be a URN UUID
        assert event_id.startswith("urn:uuid:"), "Invalid event IRI"
        # ingest_entity should have been called once
        self.mock_graph.ingest_entity.assert_called_once()
        # Inspect the created Entity
        args, _ = self.mock_graph.ingest_entity.call_args
        entity = args[0]
        assert isinstance(entity, Entity)
        c = entity.content
        assert c["@id"] == event_id
        assert c["@type"] == TELEMETRY_TYPE
        assert c["eventType"] == "httpRequest"
        assert c["quantity"] == 123
        assert c["relatesToTool"] == "urn:tool:TestTool"
        assert c.get("statusCode") == 200
        assert c.get("extraField") == "extra"
        assert "dateCreated" in c

def test_component_and_wrapper_exist():
    """Ensure LogTelemetry is registered and wrapped."""
    # COMPONENTS registry
    assert "LogTelemetry" in COMPONENTS, "LogTelemetry missing from COMPONENTS"
    meta = COMPONENTS["LogTelemetry"]
    assert meta["tool"] == "LogTelemetry"
    # DSPy wrapper
    tool_names = {t.__name__ for t in TOOLS}
    assert "LogTelemetry" in tool_names, "LogTelemetry wrapper not generated"