#!/usr/bin/env python3
"""Merge Biolord pseudo Perturb-seq output with target scRNA-seq for PerturbGRN.

This script is a cleaned, parameterized version of
`GRN/bayes_network/PerturbGRN/biolord_data_preprocess.py`.

It bridges the decoupled generation workflow and the PerturbGRN main loop:

1. Read generated/pseudo Perturb-seq AnnData from
   `main/train_biolord_pseudo_perturbseq.py`.
2. Convert perturbation labels to PerturbGRN conventions:
   `condition2 -> obs["pert"]`, `control -> CTRL`.
3. Read target-state scRNA-seq and assert that its genes are gene symbols before
   taking a gene intersection.
4. Mark target-state scRNA-seq cells as `obs["pert"] == "unknown"`.
5. Concatenate train/generated cells and test/target cells with
   `obs["label"] in {"train", "test"}`.
6. Write the merged PerturbGRN-ready h5ad and a flat cell-order JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial.distance import cdist


DEFAULT_T_MARKERS = [
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
]

ENSEMBL_GENE_PATTERN = re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE)


def read_h5ad_compat(path: Path) -> ad.AnnData:
    """Read h5ad files containing the newer nullable-value encoding.

    PerturbGRN uses Python 3.10 and anndata 0.11.4, while some target-state
    files were written by anndata 0.12 and may contain ``null`` values in
    ``uns``. Registering this reader preserves those values as ``None`` without
    requiring a third environment for the conversion step.
    """
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        is_missing_null_reader = (
            exc.__class__.__name__ == "IORegistryError"
            and "encoding_type='null'" in str(exc)
        )
        if not is_missing_null_reader:
            raise

        from anndata._io.specs.registry import IOSpec, _REGISTRY
        from anndata.compat import H5Array

        @_REGISTRY.register_read(H5Array, IOSpec("null", "0.1.0"))
        def read_null(_elem, _reader) -> None:
            return None

        print(
            "[read] enabled compatibility for h5ad null encoding 0.1.0",
            flush=True,
        )
        return sc.read_h5ad(path)


def as_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def read_marker_file(path: Path) -> list[str]:
    markers = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            gene = line.strip()
            if gene and not gene.startswith("#"):
                markers.append(gene)
    if not markers:
        raise ValueError(f"No marker genes found in {path}")
    return markers


def choose_markers(args: argparse.Namespace) -> list[str]:
    if args.marker_file is not None:
        return read_marker_file(args.marker_file)
    return DEFAULT_T_MARKERS


def assert_gene_symbols(genes: pd.Index, context: str) -> pd.Index:
    """Assert that the provided gene identifiers look like gene symbols.

    This intentionally fails on Ensembl-like IDs because taking intersections
    between generated gene-symbol data and target Ensembl IDs can silently drop
    most genes.
    """
    genes = pd.Index(genes.astype(str), dtype="object")
    assert len(genes) > 0, f"{context}: gene list is empty."
    assert not genes.isna().any(), f"{context}: gene list contains missing values."
    assert all(g.strip() for g in genes), f"{context}: gene list contains empty values."
    assert genes.is_unique, f"{context}: gene symbols must be unique before merging."

    ensembl_like = [g for g in genes if ENSEMBL_GENE_PATTERN.match(g)]
    assert not ensembl_like, (
        f"{context}: var identifiers must be gene symbols, but found Ensembl-like "
        f"IDs such as {ensembl_like[:5]}. Convert target adata.var/var_names to "
        "gene symbols before running this script."
    )
    return genes


def set_target_var_names_to_symbols(
    adata: ad.AnnData,
    gene_source: str,
) -> ad.AnnData:
    """Set target var names from a symbol source and assert symbol validity."""
    adata = adata.copy()
    if gene_source == "var_names":
        genes = pd.Index(adata.var_names.astype(str), dtype="object")
    else:
        if gene_source not in adata.var.columns:
            raise KeyError(f"Target adata.var does not contain {gene_source!r}.")
        genes = pd.Index(adata.var[gene_source].astype(str), dtype="object")

    genes = assert_gene_symbols(genes, context="target scRNA-seq")
    adata.var_names = genes
    adata.var["gene"] = adata.var_names.astype(str)
    return adata


def prepare_generated_perturbseq(
    path: Path,
    perturbation_key: str,
    control_label: str,
    include_groups: list[str] | None,
) -> ad.AnnData:
    """Load Biolord output and convert its obs/var to PerturbGRN conventions."""
    adata = read_h5ad_compat(path)

    if include_groups is not None:
        if "group" not in adata.obs.columns:
            raise KeyError("--include-groups was provided but generated adata has no obs['group'].")
        adata = adata[adata.obs["group"].astype(str).isin(include_groups)].copy()

    if perturbation_key not in adata.obs.columns:
        raise KeyError(f"Generated adata is missing obs[{perturbation_key!r}].")

    adata.obs["pert"] = adata.obs[perturbation_key].astype(str).replace(
        {control_label: "CTRL"}
    )
    if "CTRL" not in set(adata.obs["pert"].astype(str)):
        raise ValueError("Generated adata contains no CTRL cells after control-label conversion.")

    adata.var_names = pd.Index(adata.var_names.astype(str), dtype="object")
    adata.var_names_make_unique()
    adata.var["gene"] = adata.var_names.astype(str)
    return adata


def prepare_target_scrna(
    path: Path,
    gene_source: str,
    unknown_label: str,
) -> ad.AnnData:
    """Load target-state scRNA-seq and mark it as the unknown target state."""
    adata = read_h5ad_compat(path)
    adata = set_target_var_names_to_symbols(adata, gene_source=gene_source)
    adata.obs["pert"] = unknown_label
    return adata


def subset_to_common_genes(
    generated: ad.AnnData,
    target: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData, list[str]]:
    """Take an ordered gene-symbol intersection shared by generated and target data."""
    generated_genes = pd.Index(generated.var_names.astype(str))
    target_genes = pd.Index(target.var_names.astype(str))
    common = generated_genes.intersection(target_genes).tolist()
    if not common:
        raise ValueError("No common gene symbols found between generated and target AnnData.")
    return generated[:, common].copy(), target[:, common].copy(), common


def filter_valid_perturbations(
    adata: ad.AnnData,
    control_label: str,
    unknown_label: str,
) -> ad.AnnData:
    """Keep only perturbations compatible with PerturbGRN's gene-indexed updates."""
    valid = set(adata.var_names.astype(str)) | {control_label, unknown_label}
    mask = adata.obs["pert"].astype(str).isin(valid)
    dropped = int((~mask).sum())
    if dropped:
        print(f"[filter] dropped {dropped} cells with perturbation labels absent from var_names", flush=True)
    return adata[mask].copy()


def build_perturbgrn_adata(
    generated: ad.AnnData,
    target: ad.AnnData,
    control_label: str,
    unknown_label: str,
) -> ad.AnnData:
    """Concatenate generated train cells and target test cells."""
    generated, target, common_genes = subset_to_common_genes(generated, target)
    print(f"[merge] common genes: {len(common_genes)}", flush=True)

    merged = ad.concat(
        [generated, target],
        axis=0,
        join="inner",
        label="label",
        keys=["train", "test"],
        merge="same",
    )
    merged.var["gene"] = merged.var_names.astype(str)
    merged.obs["pert"] = merged.obs["pert"].astype(str)
    merged = filter_valid_perturbations(
        merged,
        control_label=control_label,
        unknown_label=unknown_label,
    )
    merged.obs_names_make_unique()
    return merged


def build_cell_order(
    adata: ad.AnnData,
    markers: list[str],
    control_label: str,
    score_name: str,
    n_neighbors: int,
    n_pcs: int,
    diffmap_n_comps: int,
    n_dcs: int,
    distance_metric: str,
) -> dict[str, list[int]]:
    """Build flat PerturbGRN cell-order JSON from the merged AnnData."""
    ctrl_adata = adata[adata.obs["pert"].astype(str) == control_label].copy()
    if ctrl_adata.n_obs == 0:
        raise ValueError(f"No control cells found with obs['pert'] == {control_label!r}.")

    present_markers = [gene for gene in markers if gene in set(adata.var_names.astype(str))]
    if not present_markers:
        raise ValueError("None of the marker genes are present in the merged AnnData.")
    missing_markers = sorted(set(markers) - set(present_markers))
    if missing_markers:
        print(f"[cell-order] ignored missing marker genes: {missing_markers}", flush=True)

    ctrl_adata.var_names = ctrl_adata.var_names.astype(str)
    sc.tl.score_genes(ctrl_adata, gene_list=present_markers, score_name=score_name)
    root_cell = ctrl_adata.obs[score_name].idxmax()
    ctrl_adata.uns["iroot"] = list(ctrl_adata.obs_names).index(root_cell)

    effective_n_neighbors = min(n_neighbors, max(2, ctrl_adata.n_obs - 1))
    sc.pp.neighbors(ctrl_adata, n_neighbors=effective_n_neighbors, n_pcs=n_pcs)
    # Match the historical Mode 3 workflow: diffmap uses 15 components while
    # DPT uses the first 10 diffusion components.
    effective_n_comps = max(3, min(diffmap_n_comps, max(3, ctrl_adata.n_obs - 2)))
    effective_n_dcs = max(1, min(n_dcs, effective_n_comps))
    sc.tl.diffmap(ctrl_adata, n_comps=effective_n_comps)
    sc.tl.dpt(ctrl_adata, n_dcs=effective_n_dcs)

    ctrl_order = list(ctrl_adata.obs["dpt_pseudotime"].sort_values().index)
    cell_order = {control_label: [int(adata.obs_names.get_loc(x)) for x in ctrl_order]}

    ctrl_mean = as_dense(ctrl_adata.X).mean(axis=0)
    perturbations = sorted(set(adata.obs["pert"].astype(str)) - {control_label})
    for pert in perturbations:
        pert_adata = adata[adata.obs["pert"].astype(str) == pert].copy()
        dist = cdist(as_dense(pert_adata.X), ctrl_mean.reshape(1, -1), metric=distance_metric).ravel()
        pert_adata.obs["dist_to_ctrl"] = dist
        ordered = list(pert_adata.obs["dist_to_ctrl"].sort_values().index)
        cell_order[pert] = [int(adata.obs_names.get_loc(x)) for x in ordered]

    return cell_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-adata",
        type=Path,
        default=Path("Data/decoupled_generation/biolord_outputs/T/result_biolord.h5ad"),
        help="Output h5ad from train_biolord_pseudo_perturbseq.py.",
    )
    parser.add_argument(
        "--target-scrna",
        type=Path,
        required=True,
        help="Target-state scRNA-seq h5ad to be labeled as the unknown target state.",
    )
    parser.add_argument(
        "--output-adata",
        type=Path,
        default=Path("Data/decoupled_generation/perturbgrn_input.h5ad"),
    )
    parser.add_argument(
        "--output-cell-order",
        type=Path,
        default=Path("Data/decoupled_generation/cell_order.json"),
    )
    parser.add_argument("--target-gene-source", default="var_names")
    parser.add_argument("--perturbation-key", default="condition2")
    parser.add_argument("--generated-control-label", default="control")
    parser.add_argument("--control-label", default="CTRL")
    parser.add_argument("--unknown-label", default="unknown")
    parser.add_argument(
        "--include-groups",
        default=None,
        help="Optional comma-separated generated obs['group'] values to keep, e.g. control,imputed.",
    )
    parser.add_argument(
        "--marker-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one root-state marker gene symbol per line. "
            "When omitted, the historical built-in T-cell markers are used."
        ),
    )
    parser.add_argument("--score-name", default="marker_score")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument(
        "--diffmap-n-comps",
        type=int,
        default=15,
        help="Diffusion-map components. The default matches the historical Mode 3 workflow.",
    )
    parser.add_argument("--n-dcs", type=int, default=10)
    parser.add_argument("--distance-metric", default="euclidean")
    parser.add_argument("--skip-cell-order", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = prepare_generated_perturbseq(
        args.generated_adata,
        perturbation_key=args.perturbation_key,
        control_label=args.generated_control_label,
        include_groups=parse_csv(args.include_groups),
    )
    target = prepare_target_scrna(
        args.target_scrna,
        gene_source=args.target_gene_source,
        unknown_label=args.unknown_label,
    )
    merged = build_perturbgrn_adata(
        generated,
        target,
        control_label=args.control_label,
        unknown_label=args.unknown_label,
    )

    print(f"[merge] final shape: {merged.shape}", flush=True)
    print("[merge] label counts:", flush=True)
    print(merged.obs["label"].value_counts().to_string(), flush=True)
    print("[merge] perturbation examples:", sorted(merged.obs["pert"].unique())[:20], flush=True)

    if not args.dry_run:
        args.output_adata.parent.mkdir(parents=True, exist_ok=True)
        print(f"[write] writing {args.output_adata}", flush=True)
        merged.write_h5ad(args.output_adata)

    if not args.skip_cell_order:
        markers = choose_markers(args)
        cell_order = build_cell_order(
            merged,
            markers=markers,
            control_label=args.control_label,
            score_name=args.score_name,
            n_neighbors=args.n_neighbors,
            n_pcs=args.n_pcs,
            diffmap_n_comps=args.diffmap_n_comps,
            n_dcs=args.n_dcs,
            distance_metric=args.distance_metric,
        )
        if not args.dry_run:
            args.output_cell_order.parent.mkdir(parents=True, exist_ok=True)
            print(f"[write] writing {args.output_cell_order}", flush=True)
            with args.output_cell_order.open("w", encoding="utf-8") as handle:
                json.dump(cell_order, handle, indent=2)

    if args.dry_run:
        print("[write] dry run requested; outputs not written", flush=True)


if __name__ == "__main__":
    main()
