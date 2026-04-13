"""
core/knowledge_graph/__init__.py
NetworkX-based in-memory knowledge graph for BioVoice.
"""

from .graph_store import BioVoiceGraph, NodeType, EdgeType

__all__ = ["BioVoiceGraph", "NodeType", "EdgeType"]
