#!/usr/bin/env python
"""Prepare PerturbGRN cell-order JSON from a preprocessed AnnData object.

The generated JSON is a flat dictionary:

    {
      "CTRL": [ordered row indices for control cells],
      "PERT1": [ordered row indices for PERT1 cells],
      ...
    }

Control cells are ordered by DPT pseudo-time. Perturbation groups are ordered by
distance to the control mean expression profile, matching the lightweight
preprocessing used in the original experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist


DEFAULT_MARKERS = {
    "T": [
        "CD8A",
        "CD8B",
        "CCR7",
        "SELL",
        "LEF1",
        "TCF7",
        "IL7R",
        "CD27",
        "CD28",
        "KLF2",
        "BACH2",
    ],
    "ESC": [
        "POU5F1",
        "SOX2",
        "NANOG",
        "KLF4",
        "KLF2",
        "ESRRB",
        "TFCP2L1",
        "PRDM14",
        "GBX2",
        "TBX3",
        "STAT3",
        "LIFR",
        "DPPA2",
        "DPPA4",
        "DPPA5A",
        "UTF1",
    ],
    "fibroblast": [
        "COL1A1",
        "COL1A2",
        "DCN",
        "VIM",
        "FAP",
    ],
}


def as_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def read_marker_file(path: str | Path) -> list[str]:
    markers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                markers.append(gene)
    if not markers:
        raise ValueError(f"No marker genes found in {path}")
    return markers


def choose_markers(args) -> list[str]:
    if args.marker_file:
        return read_marker_file(args.marker_file)
    if args.marker_preset not in DEFAULT_MARKERS:
        choices = ", ".join(sorted(DEFAULT_MARKERS))
        raise ValueError(f"Unknown marker preset {args.marker_preset!r}. Choose from: {choices}")
    return DEFAULT_MARKERS[args.marker_preset]


def ensure_gene_column(adata):
    if "gene" not in adata.var.columns:
        if "gene_name" in adata.var.columns:
            adata.var["gene"] = adata.var["gene_name"].astype(str)
        else:
            adata.var["gene"] = adata.var_names.astype(str)
    return adata


def filter_present_markers(adata, markers: list[str]) -> list[str]:
    genes = set(adata.var["gene"].astype(str))
    present = [gene for gene in markers if gene in genes]
    missing = [gene for gene in markers if gene not in genes]
    if missing:
        print(f"Warning: {len(missing)} marker genes were not found and will be ignored: {missing}")
    if not present:
        raise ValueError("None of the selected marker genes were found in adata.var['gene'].")
    return present


def build_cell_order(
    adata,
    markers: list[str],
    control_name: str,
    score_name: str,
    n_neighbors: int,
    n_pcs: int,
    n_dcs: int,
    distance_metric: str,
) -> dict[str, list[int]]:
    if "pert" not in adata.obs.columns:
        raise ValueError("Input AnnData must contain adata.obs['pert'].")

    adata.obs["pert"] = adata.obs["pert"].astype(str)
    ctrl_adata = adata[adata.obs["pert"] == control_name].copy()
    if ctrl_adata.n_obs == 0:
        raise ValueError(f"No control cells found with adata.obs['pert'] == {control_name!r}.")

    ctrl_adata.var_names = ctrl_adata.var["gene"].astype(str)
    ctrl_adata.var_names_make_unique()
    sc.tl.score_genes(ctrl_adata, gene_list=markers, score_name=score_name)
    root_cell = ctrl_adata.obs[score_name].idxmax()
    ctrl_adata.uns["iroot"] = list(ctrl_adata.obs_names).index(root_cell)

    sc.pp.neighbors(ctrl_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    effective_n_dcs = min(n_dcs, max(1, ctrl_adata.n_obs - 2))
    sc.tl.diffmap(ctrl_adata, n_comps=effective_n_dcs)
    sc.tl.dpt(ctrl_adata, n_dcs=effective_n_dcs)

    ctrl_order = list(ctrl_adata.obs["dpt_pseudotime"].sort_values().index)
    pert_dict = {control_name: [int(adata.obs_names.get_loc(x)) for x in ctrl_order]}

    ctrl_mean = as_dense(ctrl_adata.X).mean(axis=0)
    perturbations = sorted(set(adata.obs["pert"].astype(str)) - {control_name})
    for pert_name in perturbations:
        pert_adata = adata[adata.obs["pert"].astype(str) == pert_name].copy()
        x_pert = as_dense(pert_adata.X)
        dist = cdist(x_pert, ctrl_mean.reshape(1, -1), metric=distance_metric).ravel()
        pert_adata.obs["dist_to_ctrl"] = dist
        ordered = list(pert_adata.obs["dist_to_ctrl"].sort_values().index)
        pert_dict[pert_name] = [int(adata.obs_names.get_loc(x)) for x in ordered]

    return pert_dict


def write_marker_table(path: str | Path):
    rows = []
    marker_category = {
        "T": {
            "CD8A": "Lineage",
            "CD8B": "Lineage",
            "CCR7": "Homing",
            "SELL": "Homing",
            "LEF1": "Transcription factor",
            "TCF7": "Transcription factor",
            "IL7R": "Memory/survival",
            "CD27": "Co-stimulation",
            "CD28": "Co-stimulation",
            "KLF2": "Migration/quiescence",
            "BACH2": "Memory/quiescence",
        },
        "ESC": {
            "POU5F1": "Core pluripotency factor",
            "SOX2": "Core pluripotency factor",
            "NANOG": "Core pluripotency factor",
            "KLF4": "Pluripotency-associated TF",
            "KLF2": "Pluripotency-associated TF",
            "ESRRB": "Pluripotency-associated TF",
            "TFCP2L1": "Pluripotency-associated TF",
            "PRDM14": "Naive pluripotency",
            "GBX2": "Naive pluripotency",
            "TBX3": "Naive pluripotency",
            "STAT3": "Self-renewal signaling",
            "LIFR": "Self-renewal signaling",
            "DPPA2": "Chromatin/epigenetic regulator",
            "DPPA4": "Chromatin/epigenetic regulator",
            "DPPA5A": "Chromatin/epigenetic regulator",
            "UTF1": "Chromatin/epigenetic regulator",
        },
        "fibroblast": {
            "COL1A1": "Extracellular matrix",
            "COL1A2": "Extracellular matrix",
            "DCN": "Extracellular matrix",
            "VIM": "Mesenchymal marker",
            "FAP": "Activated fibroblast marker",
        },
    }
    marker_note = {"POU5F1": "also known as OCT4", "DPPA5A": "human DPPA5 family marker"}
    for preset, genes in DEFAULT_MARKERS.items():
        for gene in genes:
            rows.append(
                {
                    "preset": preset,
                    "gene": gene,
                    "category": marker_category.get(preset, {}).get(gene, ""),
                    "note": marker_note.get(gene, ""),
                }
            )

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["preset", "gene", "category", "note"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True, help="Input preprocessed .h5ad file.")
    parser.add_argument("--output", required=True, help="Output cell-order JSON path.")
    parser.add_argument(
        "--marker-preset",
        default="ESC",
        choices=sorted(DEFAULT_MARKERS),
        help="Built-in marker preset used to define the control-cell root.",
    )
    parser.add_argument(
        "--marker-file",
        default=None,
        help="Optional one-gene-per-line marker file. Overrides --marker-preset.",
    )
    parser.add_argument("--control-name", default="CTRL", help="Control label in adata.obs['pert'].")
    parser.add_argument("--score-name", default="marker_score", help="Column name for Scanpy marker score.")
    parser.add_argument("--n-neighbors", type=int, default=30, help="Number of neighbors for Scanpy.")
    parser.add_argument("--n-pcs", type=int, default=20, help="Number of PCs for Scanpy neighbors.")
    parser.add_argument("--n-dcs", type=int, default=10, help="Number of diffusion components for DPT.")
    parser.add_argument("--distance-metric", default="euclidean", help="Metric used by scipy.spatial.distance.cdist.")
    parser.add_argument(
        "--write-marker-table",
        default=None,
        help="Optional path to write the built-in marker table as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    markers = choose_markers(args)
    adata = ensure_gene_column(sc.read_h5ad(args.adata))
    markers = filter_present_markers(adata, markers)
    pert_dict = build_cell_order(
        adata,
        markers=markers,
        control_name=args.control_name,
        score_name=args.score_name,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        n_dcs=args.n_dcs,
        distance_metric=args.distance_metric,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pert_dict, f, indent=4)

    if args.write_marker_table:
        write_marker_table(args.write_marker_table)

    print(f"Wrote cell-order JSON to {output_path}")


if __name__ == "__main__":
    main()
