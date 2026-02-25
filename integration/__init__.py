"""Integration adapter package.

Provides the two-function contract for wiring the child ToolGeneratorState graph
into a parent AnalysisPipelineState graph without modifying the parent schema.

    from integration import build_child_input, apply_child_output

See mapper.py for full documentation.
"""

from .mapper import build_child_input, apply_child_output

__all__ = ["build_child_input", "apply_child_output"]
