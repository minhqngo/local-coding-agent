"""Local multi-model coding agent.

Qwen writes the code; Liquid LFM2 models do everything around it — planning
support, tool-call formatting, context compression, and repo Q&A.

The public surface is intentionally small:

    from coding_agent.config import load_config
    from coding_agent.registry import ModelRegistry
    from coding_agent.loop import AgentLoop
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
