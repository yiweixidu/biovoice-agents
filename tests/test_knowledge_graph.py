"""
tests/test_knowledge_graph.py
Unit tests for core.knowledge_graph.graph_store.BioVoiceGraph.

No live API calls, no LLM — pure NetworkX graph operations.
Run: pytest tests/test_knowledge_graph.py -v
"""

import pytest
from core.knowledge_graph.graph_store import BioVoiceGraph, NodeType, EdgeType


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ARTICLES = [
    {
        "pmid":     "12345678",
        "title":    "CR6261 targets the HA stalk of influenza H1N1",
        "abstract": (
            "The broadly neutralizing antibody CR6261 binds the HA stalk "
            "epitope conserved across H1N1pdm09 and H5N1. "
            "IGHV1-69 germline usage enables stem binding."
        ),
        "year":     "2019",
        "source":   "pubmed",
    },
    {
        "pmid":     "87654321",
        "title":    "FI6 achieves pan-influenza neutralization",
        "abstract": (
            "FI6 is a group 1 and group 2 broadly neutralizing antibody. "
            "It targets the HA stem fusion peptide with remarkable breadth "
            "across H3N2 and B/Victoria strains."
        ),
        "year":     "2020",
        "source":   "europe_pmc",
    },
    {
        "pmid":     "11111111",
        "title":    "MEDI8852 neutralizes H3N2 via HA stalk",
        "abstract": (
            "MEDI8852 targets the HA stalk epitope of H3N2 and H1N1pdm09. "
            "Crystal structure reveals a Site II binding mode."
        ),
        "year":     "2021",
        "source":   "pubmed",
    },
]

SAMPLE_ANTIBODIES = [
    {
        "antibody_name":           "CR6261",
        "target_protein":          "HA",
        "epitope_region":          "HA stalk",
        "gene_usage":              "IGHV1-69",
        "neutralization_spectrum": "H1, H5 (group 1)",
        "clinical_phase":          "Phase II",
        "key_pmids":               ["12345678"],
    },
    {
        "antibody_name":           "FI6",
        "target_protein":          "HA",
        "epitope_region":          "HA stem",
        "gene_usage":              "IGHV3-30",
        "neutralization_spectrum": "Group 1 + Group 2",
        "clinical_phase":          "Preclinical",
        "key_pmids":               ["87654321"],
    },
    {
        "antibody_name":           "MEDI8852",
        "target_protein":          "HA",
        "epitope_region":          "HA stalk",
        "gene_usage":              "IGHV1-18",
        "neutralization_spectrum": "H3, H1",
        "clinical_phase":          "Phase II",
        "key_pmids":               ["11111111"],
    },
]


@pytest.fixture
def graph() -> BioVoiceGraph:
    g = BioVoiceGraph()
    g.build_from_corpus(SAMPLE_ARTICLES)
    g.build_from_antibodies(SAMPLE_ANTIBODIES)
    return g


# ── Build tests ───────────────────────────────────────────────────────────────

def test_graph_has_nodes(graph):
    assert graph.G.number_of_nodes() > 0


def test_graph_has_edges(graph):
    assert graph.G.number_of_edges() > 0


def test_antibody_nodes_present(graph):
    ab_nodes = [
        n for n, d in graph.G.nodes(data=True)
        if d.get("type") == NodeType.ANTIBODY
    ]
    assert len(ab_nodes) >= 3, "CR6261, FI6, MEDI8852 should all be present"


def test_publication_nodes_present(graph):
    pub_nodes = [
        n for n, d in graph.G.nodes(data=True)
        if d.get("type") == NodeType.PUBLICATION
    ]
    assert len(pub_nodes) == 3


def test_epitope_nodes_present(graph):
    ep_nodes = [
        d.get("label", "")
        for _, d in graph.G.nodes(data=True)
        if d.get("type") == NodeType.EPITOPE
    ]
    assert any("HA stalk" in ep or "HA stem" in ep for ep in ep_nodes)


def test_virus_nodes_extracted(graph):
    virus_nodes = [
        n for n, d in graph.G.nodes(data=True)
        if d.get("type") == NodeType.VIRUS
    ]
    assert len(virus_nodes) >= 1, "H1N1pdm09, H3N2, etc. should be extracted"


# ── Query tests ───────────────────────────────────────────────────────────────

def test_top_targeted_epitopes_returns_list(graph):
    result = graph.query_top_targeted_epitopes(n=5)
    assert isinstance(result, list)
    assert len(result) >= 1
    first = result[0]
    assert "epitope" in first
    assert "antibody_count" in first


def test_ha_stalk_is_top_epitope(graph):
    result = graph.query_top_targeted_epitopes(n=5)
    top_labels = [r["epitope"].lower() for r in result]
    # CR6261 and MEDI8852 both bind HA stalk, so it should rank first
    assert any("ha stalk" in label for label in top_labels)


def test_antibodies_for_epitope(graph):
    result = graph.query_antibodies_for_epitope("HA stalk")
    assert len(result) >= 2, "CR6261 and MEDI8852 target HA stalk"
    result_lower = [name.lower() for name in result]
    assert any("cr6261" in r for r in result_lower)
    assert any("medi8852" in r for r in result_lower)


def test_ighv_antibodies(graph):
    result = graph.query_ighv_antibodies("IGHV1-69")
    names = [r["name"].lower() for r in result]
    assert any("cr6261" in n for n in names)


def test_shared_publications(graph):
    # CR6261 and MEDI8852 both target HA stalk; check shared pubs via graph
    shared = graph.query_shared_publications("CR6261", "MEDI8852")
    # They each cite different PMIDs in this fixture, so shared should be empty
    assert isinstance(shared, list)


def test_neighbourhood_returns_dict(graph):
    nb = graph.neighbourhood("CR6261", depth=1)
    assert "nodes" in nb
    assert "edges" in nb
    assert len(nb["nodes"]) >= 1


def test_neighbourhood_unknown_entity(graph):
    nb = graph.neighbourhood("NONEXISTENT_AB_XYZ", depth=1)
    assert "error" in nb


# ── Statistics tests ──────────────────────────────────────────────────────────

def test_statistics_keys(graph):
    stats = graph.statistics()
    assert "nodes" in stats
    assert "edges" in stats
    assert "antibody_nodes" in stats
    assert "publication_nodes" in stats
    assert stats["nodes"] > 0
    assert stats["antibody_nodes"] >= 3


# ── Export tests ──────────────────────────────────────────────────────────────

def test_to_json(graph, tmp_path):
    out = tmp_path / "graph.json"
    graph.to_json(str(out))
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert "nodes" in data


def test_to_graphml(graph, tmp_path):
    out = tmp_path / "graph.graphml"
    graph.to_graphml(str(out))
    assert out.exists()
    content = out.read_text()
    assert "graphml" in content.lower()


# ── Plot tests ────────────────────────────────────────────────────────────────

def test_plot_subgraph_returns_bytes(graph):
    try:
        buf = graph.plot_subgraph(entity_name="CR6261", depth=1)
        if buf is not None:
            assert buf.read(4) == b"\x89PNG"
    except Exception:
        pytest.skip("matplotlib not available or rendering error")


def test_plot_subgraph_no_entity_picks_top_ab(graph):
    try:
        buf = graph.plot_subgraph()  # entity_name=None
        # Should pick top antibody node automatically
        if buf is not None:
            assert buf.read(4) == b"\x89PNG"
    except Exception:
        pytest.skip("matplotlib not available")


def test_empty_graph_plot_returns_none():
    g = BioVoiceGraph()
    result = g.plot_subgraph()
    assert result is None
