"""Telemetry capabilities for Cogitarelink DSPy agents"""

# =====================================================================
# TelemetryStore: logs structured telemetry events into the semantic graph
# =====================================================================

import datetime
import uuid
import os
import json

from typing import Any
from cogitarelink.core.graph import GraphManager
from cogitarelink.core.entity import Entity
from cogitarelink.reason.prov import wrap_patch_with_prov

from .memory import ReflectionStore

__all__ = ['TELEMETRY_GRAPH', 'TELEMETRY_TYPE', 'TelemetryStore']

# Named graph for telemetry events
TELEMETRY_GRAPH = "urn:agent:telemetry"
# JSON-LD @type for telemetry events
TELEMETRY_TYPE = "https://w3id.org/cogitarelink#TelemetryEvent"

class TelemetryStore(ReflectionStore):
    """Persist structured telemetry events as JSON-LD entities in the semantic graph."""

    def __init__(self, graph: GraphManager):
        """Initialize the telemetry store with a graph manager."""
        super().__init__(graph)

    def log(self, event_type: str, value: Any, tool_iri: str, **kw) -> str:
        """Log a telemetry event.

        Args:
            event_type: A string identifying the type of event (e.g. 'httpRequest').
            value: Numeric or other quantity associated with the event.
            tool_iri: IRI of the tool or component generating this event.
            **kw: Additional key-value pairs to include in the event.

        Returns:
            str: The IRI of the newly created TelemetryEvent.
        """
        event_id = f"urn:uuid:{uuid.uuid4()}"
        now = datetime.datetime.utcnow().isoformat()
        content = {
            "@id": event_id,
            "@type": TELEMETRY_TYPE,
            "eventType": event_type,
            "quantity": value,
            "dateCreated": now,
            "relatesToTool": tool_iri,
        }
        # Merge any extra fields
        content.update(kw)
        # Include inline JSON-LD context to define telemetry terms without requiring registry entry
        ctx_path = os.path.join(os.path.dirname(__file__), "contexts", "telemetry.context.jsonld")
        with open(ctx_path, "r") as f:
            ctx_json = json.load(f)
        # Inline context mapping
        content["@context"] = ctx_json.get("@context", {})
        # Use schema vocab placeholder (will be overridden by inline context)
        ent = Entity(vocab=["schema"], content=content)
        # Persist the telemetry event
        self.graph.ingest_entity(ent)
        return event_id