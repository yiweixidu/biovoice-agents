"""
core/knowledge_graph/graph_store.py
In-memory knowledge graph for antibody-epitope-virus-literature relationships.

Uses NetworkX (no server required). Supports:
  - Entity nodes: Antibody, Epitope, Virus, Publication, Drug
  - Relationship edges: TARGETS, BINDS, NEUTRALISES, CITED_IN, INHIBITS
  - Cypher-style queries via a simple Python API
  - Graph serialisation to GraphML / JSON for inspection
  - Export of subgraphs as matplotlib figures for PPT embedding

White paper section 4.4: "Neo4j, stores entities (antibodies, epitopes,
virus subtypes, literature) and relationships (targets, binds, cites)."

This implementation uses NetworkX instead of Neo4j to:
  - Remove server infrastructure requirement
  - Allow embedding in the same process as the orchestrator
  - Export to GraphML for later Neo4j import if needed

A graph built on a full bnAb corpus answers questions like:
  "Which epitopes are targeted by the most broadly neutralising antibodies?"
  "Which publications cite both CR6261 and MEDI8852?"
  "What is the neutralisation breadth distribution of IGHV1-69 antibodies?"
"""

from __future__ import annotations

import io
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


# ── Node types ────────────────────────────────────────────────────────────────

class NodeType:
    ANTIBODY    = "Antibody"
    EPITOPE     = "Epitope"
    VIRUS       = "Virus"
    PUBLICATION = "Publication"
    DRUG        = "Drug"
    TARGET      = "Target"       # protein target (HA, NA, etc.)


class EdgeType:
    TARGETS     = "TARGETS"       # Antibody → Target
    BINDS       = "BINDS"         # Antibody → Epitope
    NEUTRALISES = "NEUTRALISES"   # Antibody → Virus
    CITED_IN    = "CITED_IN"      # Antibody/Epitope → Publication
    INHIBITS    = "INHIBITS"      # Drug → Target
    ENCODES     = "ENCODES"       # Virus → Target


# ── BioVoiceGraph ─────────────────────────────────────────────────────────────

class BioVoiceGraph:
    """
    In-memory knowledge graph.

    Nodes carry a `type` attribute (NodeType constant).
    Edges carry a `rel` attribute (EdgeType constant) plus arbitrary metadata.

    Usage:
        g = BioVoiceGraph()
        g.build_from_corpus(articles)
        # Relationship query:
        top = g.query_top_targeted_epitopes(n=5)
        # Neighbourhood:
        nb = g.neighbourhood("CR6261", depth=2)
    """

    def __init__(self):
        if not _NX_AVAILABLE:
            raise ImportError(
                "networkx is required for the knowledge graph. "
                "Install: pip install networkx"
            )
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._entity_map: Dict[str, str] = {}   # normalised_name → node_id

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_corpus(self, articles: List[Dict]) -> "BioVoiceGraph":
        """
        Extract entities and relationships from a list of corpus items.
        Each item is a dict with keys: title, abstract, pmid, source, etc.
        """
        for article in articles:
            pmid = str(article.get("pmid") or article.get("doi") or "")
            if not pmid:
                continue
            self._add_publication(pmid, article)
            self._extract_entities_from_article(pmid, article)
        return self

    def build_from_antibodies(self, antibodies: List[Dict]) -> "BioVoiceGraph":
        """
        Build graph from structured antibody extraction results
        (as returned by orchestrator._extract_antibodies).
        """
        for ab in antibodies:
            ab_name = ab.get("antibody_name", "").strip()
            if not ab_name:
                continue
            self._add_antibody(ab_name, ab)
            # Target relationship
            target = ab.get("target_protein", "").strip()
            if target:
                self._add_target(target)
                self._add_edge(
                    self._node_id("antibody", ab_name),
                    self._node_id("target", target),
                    rel=EdgeType.TARGETS,
                    epitope=ab.get("epitope_region", ""),
                )
            # Epitope relationship
            epitope = ab.get("epitope_region", "").strip()
            if epitope:
                self._add_epitope(epitope)
                self._add_edge(
                    self._node_id("antibody", ab_name),
                    self._node_id("epitope", epitope),
                    rel=EdgeType.BINDS,
                )
            # PMID citation
            for pmid in (ab.get("key_pmids") or []):
                pmid = str(pmid).strip()
                if pmid:
                    pub_node = self._node_id("publication", pmid)
                    if pub_node not in self.G:
                        self.G.add_node(pub_node, type=NodeType.PUBLICATION,
                                        pmid=pmid, label=f"PMID:{pmid}")
                    self._add_edge(
                        self._node_id("antibody", ab_name),
                        pub_node,
                        rel=EdgeType.CITED_IN,
                    )
        return self

    # ── Queries ───────────────────────────────────────────────────────────────

    def query_top_targeted_epitopes(self, n: int = 10) -> List[Dict]:
        """
        Return the n epitopes targeted by the most antibodies.
        Answers: "Which conserved epitopes are most broadly targeted?"
        """
        epitope_nodes = [
            nid for nid, data in self.G.nodes(data=True)
            if data.get("type") == NodeType.EPITOPE
        ]
        scored = []
        for ep_node in epitope_nodes:
            ab_count = self.G.in_degree(ep_node)
            scored.append({
                "epitope":         self.G.nodes[ep_node].get("label", ep_node),
                "antibody_count":  ab_count,
                "node_id":         ep_node,
            })
        return sorted(scored, key=lambda x: -x["antibody_count"])[:n]

    def query_antibodies_for_epitope(self, epitope_keyword: str) -> List[str]:
        """Return antibody names that target a given epitope (partial match)."""
        results = []
        kw = epitope_keyword.lower()
        for nid, data in self.G.nodes(data=True):
            if data.get("type") != NodeType.EPITOPE:
                continue
            if kw in data.get("label", "").lower():
                for pred in self.G.predecessors(nid):
                    pred_data = self.G.nodes[pred]
                    if pred_data.get("type") == NodeType.ANTIBODY:
                        results.append(pred_data.get("label", pred))
        return list(set(results))

    def query_shared_publications(self,
                                   entity_a: str,
                                   entity_b: str) -> List[str]:
        """Return PMIDs cited by both entity_a and entity_b."""
        na = self._find_node(entity_a)
        nb = self._find_node(entity_b)
        if not na or not nb:
            return []
        pubs_a = {n for n in self.G.successors(na)
                  if self.G.nodes[n].get("type") == NodeType.PUBLICATION}
        pubs_b = {n for n in self.G.successors(nb)
                  if self.G.nodes[n].get("type") == NodeType.PUBLICATION}
        shared = pubs_a & pubs_b
        return [self.G.nodes[p].get("pmid", p) for p in shared]

    def query_ighv_antibodies(self, ighv_gene: str) -> List[Dict]:
        """Return antibodies using a specific IGHV germline gene."""
        kw = ighv_gene.upper()
        return [
            {"name": data.get("label", nid), "ighv": data.get("ighv", "")}
            for nid, data in self.G.nodes(data=True)
            if data.get("type") == NodeType.ANTIBODY
            and kw in data.get("ighv", "").upper()
        ]

    def neighbourhood(self, entity_name: str, depth: int = 1) -> Dict:
        """
        Return a subgraph of all nodes within `depth` hops from entity_name.
        Returns a dict with nodes and edges lists (JSON-serialisable).
        """
        source = self._find_node(entity_name)
        if not source:
            return {"nodes": [], "edges": [], "error": f"Not found: {entity_name}"}

        visited: Set[str] = set()
        frontier = {source}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for n in frontier:
                for nb in list(self.G.successors(n)) + list(self.G.predecessors(n)):
                    if nb not in visited:
                        next_frontier.add(nb)
            visited |= frontier
            frontier = next_frontier
        visited |= frontier

        subgraph = self.G.subgraph(visited)
        nodes = [
            {"id": n, "label": self.G.nodes[n].get("label", n),
             "type": self.G.nodes[n].get("type", "Unknown")}
            for n in subgraph.nodes()
        ]
        edges = [
            {"from": u, "to": v,
             "rel": data.get("rel", "")}
            for u, v, data in subgraph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def statistics(self) -> Dict:
        """Return graph-level statistics."""
        type_counts: Dict[str, int] = defaultdict(int)
        for _, data in self.G.nodes(data=True):
            type_counts[data.get("type", "Unknown")] += 1
        return {
            "nodes":              self.G.number_of_nodes(),
            "edges":              self.G.number_of_edges(),
            "antibody_nodes":     type_counts.get(NodeType.ANTIBODY, 0),
            "virus_nodes":        type_counts.get(NodeType.VIRUS, 0),
            "publication_nodes":  type_counts.get(NodeType.PUBLICATION, 0),
            "epitope_nodes":      type_counts.get(NodeType.EPITOPE, 0),
            "by_type":            dict(type_counts),
            "top_epitopes":       self.query_top_targeted_epitopes(n=3),
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def to_graphml(self, path: str):
        """Export to GraphML for Neo4j import or Gephi visualisation."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        nx.write_graphml(self.G, path)
        print(f"[Graph] Exported GraphML: {path}")

    def to_json(self, path: str):
        """Export node-link JSON (D3.js compatible)."""
        data = nx.node_link_data(self.G)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Graph] Exported JSON: {path}")

    def plot_subgraph(
        self,
        entity_name: Optional[str] = None,
        depth: int = 1,
        max_nodes: int = 40,
    ) -> Optional[io.BytesIO]:
        """
        Render a neighbourhood subgraph (or whole graph if small) as a PNG
        (BytesIO) for PPT embedding. Nodes are coloured by type.

        Parameters
        ----------
        entity_name : anchor node label (partial match). If None, picks the
                      highest-degree antibody node, or returns None if graph
                      has no antibody nodes.
        depth       : hop depth from anchor.
        max_nodes   : cap — if subgraph would exceed this, trim to max_nodes.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        if self.G.number_of_nodes() == 0:
            return None

        # Pick anchor node
        if entity_name is None:
            ab_nodes = [
                (nid, self.G.degree(nid))
                for nid, data in self.G.nodes(data=True)
                if data.get("type") == NodeType.ANTIBODY
            ]
            if not ab_nodes:
                return None
            entity_name = self.G.nodes[
                max(ab_nodes, key=lambda x: x[1])[0]
            ].get("label", "")

        nb = self.neighbourhood(entity_name, depth)
        if not nb["nodes"]:
            return None

        sub_node_ids = [n["id"] for n in nb["nodes"]][:max_nodes]
        subgraph = self.G.subgraph(sub_node_ids)

        TYPE_COLORS = {
            NodeType.ANTIBODY:    "#2E86AB",
            NodeType.EPITOPE:     "#3BB273",
            NodeType.VIRUS:       "#E84855",
            NodeType.PUBLICATION: "#F4A259",
            NodeType.DRUG:        "#9B5DE5",
            NodeType.TARGET:      "#1A3557",
        }
        node_colors = [
            TYPE_COLORS.get(self.G.nodes[n].get("type", ""), "#cccccc")
            for n in subgraph.nodes()
        ]
        labels = {n: self.G.nodes[n].get("label", n)[:20]
                  for n in subgraph.nodes()}

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#F5F6FA")
        ax.set_facecolor("#F5F6FA")

        pos = nx.spring_layout(subgraph, seed=42, k=2.5)
        nx.draw_networkx_nodes(subgraph, pos, ax=ax,
                               node_color=node_colors, node_size=800, alpha=0.9)
        nx.draw_networkx_labels(subgraph, pos, labels, ax=ax,
                                font_size=7, font_color="#1C1C1E")
        nx.draw_networkx_edges(subgraph, pos, ax=ax,
                               edge_color="#D0D3DA", arrows=True,
                               arrowsize=15, width=1.2, alpha=0.8)
        # Legend
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=t)
                   for t, c in TYPE_COLORS.items() if t in
                   {self.G.nodes[n].get("type") for n in subgraph.nodes()}]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.8)
        ax.set_title(f"Knowledge graph: {entity_name} (depth={depth})",
                     fontsize=11, color="#1A3557", fontweight="bold")
        ax.set_axis_off()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor="#F5F6FA")
        buf.seek(0)
        plt.close(fig)
        return buf

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _node_id(self, type_prefix: str, name: str) -> str:
        key = f"{type_prefix}:{name.lower().strip()}"
        return key

    def _find_node(self, name: str) -> Optional[str]:
        """Find node by partial label match."""
        name_lower = name.lower()
        for nid, data in self.G.nodes(data=True):
            if name_lower in data.get("label", "").lower():
                return nid
        return None

    def _add_edge(self, src: str, dst: str, rel: str, **attrs):
        if src in self.G and dst in self.G:
            self.G.add_edge(src, dst, rel=rel, **attrs)

    def _add_antibody(self, name: str, data: Dict):
        nid = self._node_id("antibody", name)
        if nid not in self.G:
            self.G.add_node(nid,
                            type=NodeType.ANTIBODY,
                            label=name,
                            ighv=data.get("gene_usage", ""),
                            clinical_phase=data.get("clinical_phase", ""),
                            breadth=data.get("neutralization_spectrum", ""))
        else:
            # Update with richer data if available (e.g. from structured extraction)
            node = self.G.nodes[nid]
            if not node.get("ighv") and data.get("gene_usage"):
                node["ighv"] = data["gene_usage"]
            if not node.get("clinical_phase") and data.get("clinical_phase"):
                node["clinical_phase"] = data["clinical_phase"]
            if not node.get("breadth") and data.get("neutralization_spectrum"):
                node["breadth"] = data["neutralization_spectrum"]

    def _add_epitope(self, region: str):
        nid = self._node_id("epitope", region)
        if nid not in self.G:
            self.G.add_node(nid, type=NodeType.EPITOPE, label=region)

    def _add_target(self, target: str):
        nid = self._node_id("target", target)
        if nid not in self.G:
            self.G.add_node(nid, type=NodeType.TARGET, label=target)

    def _add_virus(self, name: str):
        nid = self._node_id("virus", name)
        if nid not in self.G:
            self.G.add_node(nid, type=NodeType.VIRUS, label=name)

    def _add_publication(self, pmid: str, data: Dict):
        nid = self._node_id("publication", pmid)
        if nid not in self.G:
            self.G.add_node(nid,
                            type=NodeType.PUBLICATION,
                            label=data.get("title", f"PMID:{pmid}")[:60],
                            pmid=pmid,
                            year=str(data.get("year", "")),
                            source=data.get("source", ""))

    # ── Entity extraction from free text ─────────────────────────────────────

    _ANTIBODY_PATTERN = re.compile(
        r"\b("
        r"CR\d{4}|FI6[v\d]*|MEDI\d{4}|VIS\d{3}|CT\d{3}|"
        r"[A-Z]{2,4}-[0-9]{3,6}|"
        r"[A-Z]{1,4}\d{3,5}[a-zA-Z]?\d*"
        r")\b"
    )
    _SUBTYPE_PATTERN = re.compile(
        r"\b(H\d{1,2}N\d{1,2}|H\dN\d|H1N1pdm09|B/Victoria|B/Yamagata|"
        r"influenza [AB])\b",
        re.IGNORECASE,
    )
    _EPITOPE_PATTERN = re.compile(
        r"\b(HA stalk|HA stem|HA head|RBS|receptor.binding.site|"
        r"fusion peptide|trimer interface|group [12] conserved|"
        r"[Ss]ite [IVX]+)\b",
        re.IGNORECASE,
    )

    def _extract_entities_from_article(self, pmid: str, article: Dict):
        text = f"{article.get('title','')} {article.get('abstract','')}"
        pub_node = self._node_id("publication", pmid)

        # Antibodies
        for m in self._ANTIBODY_PATTERN.finditer(text):
            ab_name = m.group(1)
            self._add_antibody(ab_name, {})
            self._add_edge(self._node_id("antibody", ab_name), pub_node,
                           rel=EdgeType.CITED_IN)

        # Virus subtypes
        for m in self._SUBTYPE_PATTERN.finditer(text):
            virus = m.group(0)
            self._add_virus(virus)

        # Epitope regions
        for m in self._EPITOPE_PATTERN.finditer(text):
            ep = m.group(0)
            self._add_epitope(ep)
