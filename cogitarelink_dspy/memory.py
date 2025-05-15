"""Semantic memory capabilities for Cogitarelink DSPy agents"""
__all__ = ['REFLECTION_GRAPH', 'REFLECTION_TYPE', 'ReflectionStore']

# %% ../nbs/03_memory.ipynb 2
import datetime, uuid, dspy
from typing import List, Optional
from cogitarelink.core.graph import GraphManager
from cogitarelink.core.entity import Entity
from cogitarelink.reason.prov import wrap_patch_with_prov

# Ensure 'clref' prefix is registered in the global registry (alias to 'schema')
from cogitarelink.vocab.registry import registry
_schema_entry = registry._v.get('schema')
if _schema_entry and 'clref' not in registry._v:
    registry._v['clref'] = _schema_entry

REFLECTION_GRAPH = "urn:agent:reflections"
REFLECTION_TYPE  = "https://w3id.org/cogitarelink#ReflectionNote"

class ReflectionStore:
    """Persist 'lesson learned' notes as JSON-LD entities in the Cogitarelink graph.
    
    This class provides a semantic memory store for agent reflections, storing them
    as properly typed JSON-LD entities in the Cogitarelink graph with full provenance.
    It enables storing, retrieving, and formatting reflections for use in prompts.
    
    Attributes:
        graph (GraphManager): The Cogitarelink graph manager instance to use for storage
    """
    def __init__(self, graph: GraphManager):
        """Initialize the reflection store with a graph manager.
        
        Args:
            graph (GraphManager): The Cogitarelink graph manager for accessing the knowledge graph
        """
        self.graph = graph

    def add(self, text: str, tags: Optional[List[str]] = None) -> str:
        """Add a new reflection note to the semantic memory.
        
        This method creates a new ReflectionNote entity with the given text and tags,
        assigns it a UUID, timestamps it, and stores it in the graph with provenance.
        
        Args:
            text (str): The reflection text to store
            tags (List[str], optional): A list of tags to categorize this reflection
            
        Returns:
            str: The ID of the newly created reflection
        """
        note_id = f"urn:uuid:{uuid.uuid4()}"
        now     = datetime.datetime.utcnow().isoformat()
        content = {
            "@id": note_id,
            "@type": REFLECTION_TYPE,
            "text": text,
            "tags": tags or [],
            "dateCreated": now
        }
        ent = Entity(vocab=["clref","schema"], content=content)
        # Persist the reflection note
        self.graph.ingest_entity(ent)
        return note_id

    def retrieve(self, limit: int = 5, tag_filter: Optional[str] = None) -> List[Entity]:
        """Fetch up to `limit` most recent notes, optionally filtering by tag.
        
        This method retrieves reflection notes from the graph, sorted by creation date
        (newest first), and optionally filtered by a specific tag.
        
        Args:
            limit (int, optional): Maximum number of notes to retrieve. Defaults to 5.
            tag_filter (str, optional): Only return notes with this tag. Defaults to None.
            
        Returns:
            List[Entity]: List of reflection notes as Entity objects
        """
        ids = []
        # GraphManager.query only supports (subj, pred, obj) 
        triples = self.graph.query(pred="http://schema.org/dateCreated")
        triples.sort(key=lambda t: t[2], reverse=True)
        for s,_,_ in triples[:limit]:
            if tag_filter:
                tag_triples = self.graph.query(subj=s, pred="http://schema.org/tags")
                tags = [o for (_,_,o) in tag_triples]
                if tag_filter not in tags:
                    continue
            ids.append(s)
        ents = []
        for nid in ids:
            t = self.graph.query(subj=nid, pred="http://schema.org/text")
            text = t[0][2] if t else ""
            ents.append(Entity(vocab=["clref","schema"], content={
                "@id": nid,
                "@type": REFLECTION_TYPE,
                "text": text
            }))
        return ents

    def as_prompt(self, limit: int = 5) -> str:
        """Format retrieved reflections for inclusion in a system prompt.
        
        This method retrieves recent reflections and formats them as a 
        bulleted list suitable for inclusion in system prompts.
        
        Args:
            limit (int, optional): Maximum number of notes to include. Defaults to 5.
            
        Returns:
            str: Formatted string with bullet points for each reflection
        """
        notes = self.retrieve(limit)
        return "\n".join(f"• {e.content['text']}" for e in notes)
