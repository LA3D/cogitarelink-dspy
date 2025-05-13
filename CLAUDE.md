# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements a DSPy agent for Cogitarelink, focusing on Semantic Web integration. The project follows Jeremy Howard's literate programming approach using nbdev, with an emphasis on exploratory development in notebooks first, then exporting to clean Python modules.

## Development Environment

- **Python Package Management**: Use `uv` with the existing virtualenv in `venv/`
- **Development Style**: Follow nbdev/Jeremy Howard's literate programming principles
  - Code is developed in notebooks (in `nbs/`) first
  - Never edit Python modules directly - only modify notebooks and export
  - Use in-notebook testing cells extensively

## Key Commands

- **Export Notebooks to Modules**: `nbdev_export`
- **Testing**: 
  - Run all tests: `python -m pytest`
  - Run a specific test: `python -m pytest tests/test_file.py::test_function -v`
- **Python Environment**: Activate with `source .venv/bin/activate`

## Repository Structure

- `nbs/`: Notebooks where all development happens
  - `00_core.ipynb`: Basic DSPy agent implementation
  - `01_components.ipynb`: Component catalog for Semantic Web tools
  - `02_wrappers.ipynb`: Tool wrapper generator
- `cogitarelink_dspy/`: Auto-generated Python modules (don't edit directly)
- `tests/`: Pytest test files
- `plans/`: Planning documents and implementation strategies
- `docs/`: Project documentation including architecture notes

## Architecture

This project implements a 4-layer Semantic Web architecture:
1. **Context** - Working with JSON-LD contexts and namespaces
2. **Ontology** - Using vocabularies and ontology terms
3. **Rules** - Applying validation rules and shapes
4. **Instances** - Managing actual data instances
5. **Verification** - Verifying and signing graphs

Components are organized by their semantic layer, and the agent is designed to select the highest appropriate layer for each user request.

## Development Patterns

1. **Notebook-First Development**:
   - All code is developed in notebooks with nbdev annotations
   - Use `#| export` to mark cells for inclusion in modules
   - Use `#| hide` for notebook-only code (setup, tests, etc.)

2. **Testing Approach**:
   - Include tests within notebooks as assertion cells
   - Formalize tests in the `tests/` directory for automation
   - Use pytest for running test suites

3. **Component Registry Pattern**:
   - New components are added to the catalog in `components.py`
   - Tool wrappers are auto-generated from component metadata
   - Agent dynamically selects tools based on semantic layer

## Adding New Components

To add a new component:
1. Add the component definition to `COMPONENTS` in `nbs/01_components.ipynb`
2. Include `layer`, `tool`, `doc`, and `calls` fields
3. Run `nbdev_export` to update the modules
4. Test with both in-notebook tests and pytest

## Common Issues

- If changes to Python modules are lost, you likely edited them directly instead of modifying the source notebooks
- If imports fail, ensure you're running with the virtual environment activated
- If tool generation fails, check the signature format in component definitions

## Documentation

- Architecture overview in `docs/AGENT_OVERVIEW.md`
- Implementation plan in `docs/cl-dspy-plan.md`
- Component catalog plan in `plans/component-catalog-plan.md`