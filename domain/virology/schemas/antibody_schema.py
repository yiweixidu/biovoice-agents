"""
domain/virology/schemas/antibody_schema.py
JSON schema dict used to guide LLM extraction of broadly neutralizing
antibody (bnAb) data from virology literature.

Imported by biovoice/core/orchestrator.py:
    from domain.virology.schemas.antibody_schema import antibody_schema
"""

antibody_schema = {
    "antibody_name": "string — common name or clone ID, e.g. 'CR6261', 'MEDI8852'",
    "target_protein": "string — antigen targeted, e.g. 'hemagglutinin HA stalk', 'neuraminidase'",
    "epitope_region": "string — binding site descriptor, e.g. 'HA stem', 'receptor binding site', 'fusion peptide'",
    "gene_usage": "string — germline IGHV gene, e.g. 'IGHV1-69', 'IGHV6-1'",
    "neutralization_spectrum": "string — breadth summary, e.g. 'group 1 influenza A', 'broad H1/H3', 'pan-influenza A/B'",
    "ic50_range": "string — IC50 or IC80 value(s) with units, e.g. '0.01-0.5 μg/mL'",
    "effector_functions": "string — Fc-mediated activities observed, e.g. 'ADCC, ADCP', 'complement activation', 'none reported'",
    "clinical_phase": "string — highest development stage, e.g. 'preclinical', 'Phase I', 'Phase II', 'approved'",
    "structure_pdb": "string — PDB ID(s) if crystal/cryo-EM structure available, e.g. '3GBN', '' if none",
    "key_pmids": "list[string] — PMIDs of primary discovery and characterization papers",
    "notes": "string — any notable caveats, resistance mutations, or cross-reactivity observations",
}
