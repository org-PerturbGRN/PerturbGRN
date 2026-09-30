#!/usr/bin/env python3
"""Train Biolord and generate pseudo Perturb-seq responses for target cells.

This script is a cleaned, parameterized version of
`GRN/bayes_network/test/Biolord/biolord-emCell.py`.

Inputs:
  1. A user-prepared merged AnnData satisfying the documented Mode 3 schema.
  2. A cell/source embedding dictionary whose keys match `obs["condition1"]`.

For each target source/cell type, perturbed target cells are held out as OOD.
Biolord is trained on all remaining cells and then predicts target-cell
perturbation responses from the target control cells.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import numpy as np
import scanpy as sc


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_embedding_dict(path: Path) -> dict[str, np.ndarray]:
    """Load a pickle/joblib embedding dictionary and convert values to 1D arrays."""
    try:
        raw = joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            raw = pickle.load(handle)

    if not isinstance(raw, dict):
        raise TypeError("Embedding file must contain a dictionary keyed by condition1.")

    embedding_dict: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Embedding for {key!r} is empty.")
        embedding_dict[str(key)] = arr
    return embedding_dict


def validate_input_adata(
    adata: ad.AnnData,
    source_key: str,
    perturbation_key: str,
    layer: str,
) -> None:
    """Check that the merged AnnData has the fields used by Biolord."""
    missing_obs = [key for key in [source_key, perturbation_key] if key not in adata.obs]
    if missing_obs:
        raise KeyError(f"AnnData is missing required obs columns: {missing_obs}")
    if layer not in adata.layers:
        raise KeyError(f"AnnData is missing required layer {layer!r}.")


def add_cell_type_embeddings(
    adata: ad.AnnData,
    embedding_dict: dict[str, np.ndarray],
    source_key: str,
    embedding_key: str,
) -> None:
    """Attach one source/cell-type embedding to every cell."""
    sources = adata.obs[source_key].astype(str)
    missing = sorted(set(sources) - set(embedding_dict))
    if missing:
        raise KeyError(
            "Embedding dictionary is missing these condition1 values: "
            + ", ".join(missing)
        )

    used_dims = {embedding_dict[key].shape[0] for key in set(sources)}
    if len(used_dims) != 1:
        dims = {key: value.shape[0] for key, value in embedding_dict.items()}
        raise ValueError(f"Embedding dimensions are inconsistent: {dims}")

    adata.obsm[embedding_key] = np.vstack([embedding_dict[key] for key in sources])


def assign_split(
    adata: ad.AnnData,
    target: str,
    source_key: str,
    perturbation_key: str,
    control_label: str,
    valid_fraction: float,
    seed: int,
) -> str:
    """Create train/valid/OOD split matching the original notebook logic."""
    rng = np.random.RandomState(seed)
    split_key = f"split_{target}"
    obs = adata.obs.copy()
    obs["ct_con"] = obs[source_key].astype(str) + "_" + obs[perturbation_key].astype(str)
    obs[split_key] = None

    is_target = obs[source_key].astype(str) == target
    is_perturbed = obs[perturbation_key].astype(str) != control_label
    obs.loc[is_target & is_perturbed, split_key] = "ood"

    train_valid_mask = obs[split_key] != "ood"
    obs.loc[train_valid_mask, split_key] = rng.choice(
        ["train", "valid"],
        size=int(train_valid_mask.sum()),
        replace=True,
        p=[1.0 - valid_fraction, valid_fraction],
    )
    adata.obs = obs
    return split_key


def module_params(args: argparse.Namespace) -> dict[str, Any]:
    """Biolord module parameters kept close to `biolord-emCell.py`."""
    return {
        "decoder_width": args.decoder_width,
        "decoder_depth": args.decoder_depth,
        "attribute_nn_width": args.attribute_nn_width,
        "attribute_nn_depth": args.attribute_nn_depth,
        "n_latent_attribute_categorical": args.n_latent_attribute_categorical,
        "gene_likelihood": args.gene_likelihood,
        "reconstruction_penalty": args.reconstruction_penalty,
        "unknown_attribute_penalty": args.unknown_attribute_penalty,
        "unknown_attribute_noise_param": args.unknown_attribute_noise_param,
        "attribute_dropout_rate": args.attribute_dropout_rate,
        "use_batch_norm": args.use_batch_norm,
        "use_layer_norm": args.use_layer_norm,
        "seed": args.model_seed,
    }


def trainer_params(args: argparse.Namespace) -> dict[str, Any]:
    """Training hyperparameters kept close to `biolord-emCell.py`."""
    return {
        "n_epochs_warmup": args.n_epochs_warmup,
        "latent_lr": args.latent_lr,
        "latent_wd": args.latent_wd,
        "decoder_lr": args.decoder_lr,
        "decoder_wd": args.decoder_wd,
        "attribute_nn_lr": args.attribute_nn_lr,
        "attribute_nn_wd": args.attribute_nn_wd,
        "step_size_lr": args.step_size_lr,
        "cosine_scheduler": args.cosine_scheduler,
        "scheduler_final_lr": args.scheduler_final_lr,
    }


def train_and_predict(
    base_adata: ad.AnnData,
    embedding_dict: dict[str, np.ndarray],
    target: str,
    args: argparse.Namespace,
) -> ad.AnnData:
    """Train one target-specific Biolord model and return result AnnData."""
    import biolord

    adata = base_adata.copy()
    adata.obs_names_make_unique()
    adata.X = adata.layers[args.layer].copy()

    add_cell_type_embeddings(adata, embedding_dict, args.source_key, args.embedding_key)
    split_key = assign_split(
        adata,
        target=target,
        source_key=args.source_key,
        perturbation_key=args.perturbation_key,
        control_label=args.control_label,
        valid_fraction=args.valid_fraction,
        seed=args.split_seed,
    )

    target_control_mask = (
        (adata.obs[args.source_key].astype(str) == target)
        & (adata.obs[args.perturbation_key].astype(str) == args.control_label)
    )
    if int(target_control_mask.sum()) == 0:
        raise ValueError(
            f"No target control cells found for {target!r}. Expected "
            f"{args.source_key} == {target!r} and "
            f"{args.perturbation_key} == {args.control_label!r}."
        )

    if int((adata.obs[split_key] == "ood").sum()) == 0:
        print(f"[{target}] warning: no target perturbed cells were marked as OOD.", flush=True)

    biolord.Biolord.setup_anndata(
        adata=adata,
        ordered_attributes_keys=[args.embedding_key],
        categorical_attributes_keys=[args.perturbation_key],
        layer=args.layer,
    )

    model = biolord.Biolord(
        adata=adata,
        n_latent=args.n_latent,
        model_name=target,
        module_params=module_params(args),
        train_classifiers=False,
        split_key=split_key,
        train_split="train",
        valid_split="valid",
        test_split="ood",
    )

    print(f"[{target}] model training start", flush=True)
    model.train(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        plan_kwargs=trainer_params(args),
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_checkpointing=args.enable_checkpointing,
    )

    source_adata = adata[target_control_mask].copy()
    pred_adata = model.compute_prediction_adata(
        adata,
        source_adata,
        target_attributes=[args.perturbation_key],
    )

    target_adata = adata[adata.obs[args.source_key].astype(str) == target].copy()
    ctrl_adata = target_adata[
        target_adata.obs[args.perturbation_key].astype(str) == args.control_label
    ].copy()
    stim_adata = target_adata[
        target_adata.obs[args.perturbation_key].astype(str) != args.control_label
    ].copy()
    pred_adata = pred_adata[
        pred_adata.obs[args.perturbation_key].astype(str) != args.control_label
    ].copy()

    ctrl_adata.obs["group"] = "control"
    stim_adata.obs["group"] = "stimulated"
    pred_adata.obs["group"] = "imputed"
    return ad.concat([ctrl_adata, stim_adata, pred_adata], axis=0, join="outer")


def write_per_perturbation_outputs(
    result: ad.AnnData,
    output_dir: Path,
    perturbation_key: str,
    control_label: str,
) -> None:
    """Optionally save one h5ad per perturbation for downstream evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    perturbations = sorted(set(result.obs[perturbation_key].astype(str)) - {control_label})
    for perturbation in perturbations:
        subset = result[
            (result.obs[perturbation_key].astype(str) == perturbation)
            | (result.obs[perturbation_key].astype(str) == control_label)
        ].copy()
        subset.write_h5ad(output_dir / f"{perturbation}_imputed.h5ad")


def output_path_for_target(output_dir: Path, output_template: str, target: str) -> Path:
    relative = Path(output_template.format(target=target))
    if relative.is_absolute():
        return relative
    return output_dir / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adata",
        type=Path,
        default=Path("Data/decoupled_generation/adata_decoupled_generation_input.h5ad"),
        help="User-prepared Biolord-ready merged AnnData.",
    )
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path("Data/decoupled_generation/CellTypeEmbedding.pkl"),
        help="Pickle/joblib dictionary keyed by adata.obs['condition1'].",
    )
    parser.add_argument("--targets", default="T", help="Comma-separated target condition1 values.")
    parser.add_argument("--output-dir", type=Path, default=Path("Data/decoupled_generation/biolord_outputs"))
    parser.add_argument("--output-template", default="{target}/result_biolord.h5ad")
    parser.add_argument("--write-per-perturbation", action="store_true")
    parser.add_argument("--source-key", default="condition1")
    parser.add_argument("--perturbation-key", default="condition2")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--layer", default="logNor")
    parser.add_argument("--embedding-key", default="CellTypeEmbedding")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=1116)
    parser.add_argument("--cuda-visible-devices", default=None)

    parser.add_argument("--n-latent", type=int, default=256)
    parser.add_argument("--decoder-width", type=int, default=1024)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--attribute-nn-width", type=int, default=512)
    parser.add_argument("--attribute-nn-depth", type=int, default=2)
    parser.add_argument("--n-latent-attribute-categorical", type=int, default=4)
    parser.add_argument("--gene-likelihood", default="normal")
    parser.add_argument("--reconstruction-penalty", type=float, default=1e2)
    parser.add_argument("--unknown-attribute-penalty", type=float, default=1e1)
    parser.add_argument("--unknown-attribute-noise-param", type=float, default=1e-1)
    parser.add_argument("--attribute-dropout-rate", type=float, default=0.1)
    parser.add_argument("--use-batch-norm", action="store_true")
    parser.add_argument("--use-layer-norm", action="store_true")
    parser.add_argument("--model-seed", type=int, default=42)

    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs-warmup", type=int, default=0)
    parser.add_argument("--latent-lr", type=float, default=1e-4)
    parser.add_argument("--latent-wd", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=1e-4)
    parser.add_argument("--decoder-wd", type=float, default=1e-4)
    parser.add_argument("--attribute-nn-lr", type=float, default=1e-2)
    parser.add_argument("--attribute-nn-wd", type=float, default=4e-8)
    parser.add_argument("--step-size-lr", type=int, default=45)
    parser.add_argument("--cosine-scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scheduler-final-lr", type=float, default=1e-5)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=5)
    parser.add_argument("--enable-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    adata = sc.read_h5ad(args.adata)
    validate_input_adata(adata, args.source_key, args.perturbation_key, args.layer)
    embedding_dict = load_embedding_dict(args.embedding)

    print("adata sources:", sorted(adata.obs[args.source_key].astype(str).unique()), flush=True)
    print("embedding keys:", sorted(embedding_dict), flush=True)

    for target in parse_csv(args.targets):
        result = train_and_predict(adata, embedding_dict, target, args)
        out_path = output_path_for_target(args.output_dir, args.output_template, target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{target}] writing {out_path}", flush=True)
        result.write_h5ad(out_path)

        if args.write_per_perturbation:
            write_per_perturbation_outputs(
                result,
                output_dir=out_path.parent / "per_perturbation",
                perturbation_key=args.perturbation_key,
                control_label=args.control_label,
            )


if __name__ == "__main__":
    main()
