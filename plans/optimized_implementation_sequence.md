# Optimized Implementation Sequence

This document outlines the preferred sequence for implementing features from the roadmap, prioritizing core working functionality before optimization.

## 1. Structured Agents and System Prompt (Roadmap Step 3)

**Goal**: Create a functional minimum viable product with basic agent capabilities

- Create `cogitarelink_dspy/pipelines.py` with agent definitions
- Define the Semantic Web system prompt with layers
- Implement `HelloLOD` for basic tasks
- Implement `FullPlanner` with all available tools
- Ensure basic functionality using our fixed wrappers

**Success criteria**: Agent can correctly select tools based on query, with sensible layer detection

## 2. Layered Curriculum Dataset (Roadmap Step 4)

**Goal**: Build test cases for training and evaluating agent behavior

- Create `data/curriculum/` directory structure
- Implement the curriculum schema with fields for validation
- Develop initial examples for each semantic layer:
  - `00_context.jsonl` - JSON-LD context operations
  - `01_ontology.jsonl` - vocabulary lookups
  - `02_rules.jsonl` - validation tasks
  - `03_instances.jsonl` - data manipulation
- Add `99_reflection_seed.jsonl` with initial memory entries
- Develop metrics for measuring agent performance

**Success criteria**: Curriculum examples pass validation and represent diverse tasks for each layer

## 3. Advanced JSON-LD Features (Roadmap Step 0.5)

**Goal**: Optimize memory and telemetry with better data structures

- Upgrade tag representation to use JSON-LD `@set` containers
- Implement tag-based indexing with `@container: ["@set","@index"]`
- Add date indexing for temporal telemetry queries
- Create JSON-LD framing templates for reflection retrieval
- Optimize memory.py and telemetry.py with these enhancements

**Success criteria**: O(1) lookups for tag and date-based queries with clean JSON-LD serialization

## 4. Training and Evaluation Pipeline (Roadmap Steps 6-7)

**Goal**: Train the agent using the curriculum and evaluate its performance

- Implement Bootstrap few-shot learning with dev set examples
- Set up SIMBA training with curriculum splits
- Create evaluation metrics that consider:
  - Tool selection accuracy
  - Response correctness
  - Memory usage
  - Latency
- Add comprehensive evaluation notebook

**Success criteria**: Trained agent outperforms baseline on test split with measurable metrics

## 5. Advanced Features (Roadmap Step 9)

**Goal**: Add sophisticated features after core functionality proven

- Multi-tool chaining with `exp_trace` field
- Latency-aware metrics
- Reflection garbage collection
- Golden-run replay
- Safety features

**Success criteria**: Enhanced agent with robust production-grade capabilities

## Progress Tracking

| Stage | Status | Next Actions | Owner | Completed Date |
|-------|--------|--------------|-------|---------------|
| Memory & Telemetry Foundations | ✅ | | | 2025-05-15 |
| Fix Wrappers | ✅ | | | 2025-05-15 |
| Structured Agents | 🔄 | Create pipelines.py | | |
| Curriculum Dataset | 📝 | | | |
| Advanced JSON-LD | 📝 | | | |
| Training Pipeline | 📝 | | | |
| Evaluation | 📝 | | | |

Legend:
- ✅ Complete
- 🔄 In Progress
- 📝 Planned
- ❌ Blocked