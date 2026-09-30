import csv
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scanpy as sc
from tqdm import tqdm

try:
    import config as cfg
    from utils import (
        BufferedCSVLogger,
        as_dense,
        build_responsibility_matrix,
        caculate_cor_matrix,
        compute_percentile_matrix,
        compute_percentile_matrix_old_effect,
        compute_prior_improvement,
        downsample_by_mask,
        ensure_gene_column,
        ensure_pert_column,
        evaluate_network_reconstruction,
        get_DE_old_effect,
        get_intervene_index,
        get_regulator_list,
        get_score_improvement_EM,
        get_target_index,
        log_likelihood_0,
        masked_standardize,
        m_step_fast,
        minmax_normalize,
        normaliz,
        ols_linear_fit,
        reorder_array,
        split_dict,
        split_number,
        to_row_iter_value,
        true_adj,
        update_ll0_parallel,
    )
except ImportError:
    from . import config as cfg
    from .utils import (
        BufferedCSVLogger,
        as_dense,
        build_responsibility_matrix,
        caculate_cor_matrix,
        compute_percentile_matrix,
        compute_percentile_matrix_old_effect,
        compute_prior_improvement,
        downsample_by_mask,
        ensure_gene_column,
        ensure_pert_column,
        evaluate_network_reconstruction,
        get_DE_old_effect,
        get_intervene_index,
        get_regulator_list,
        get_score_improvement_EM,
        get_target_index,
        log_likelihood_0,
        masked_standardize,
        m_step_fast,
        minmax_normalize,
        normaliz,
        ols_linear_fit,
        reorder_array,
        split_dict,
        split_number,
        to_row_iter_value,
        true_adj,
        update_ll0_parallel,
    )


GLOBAL = {}

DEFAULT_TEST_PERT_SITES = ["unknown"]
DEFAULT_GRN_CONTROL_NAMES = ["CTRL", "test1", "non-targeting"]
POSTERIOR_DELTA_THRESHOLD = 5e-5
TAU_MULTIPLIER = 3.0
POSTERIOR_PERT_WEIGHT = 0.3
POSTERIOR_NONPERT_WEIGHT = 0.7
DEFAULT_DRIVER_EARLY_STOP = True
DRIVER_STOP_MIN_ITER = 0
GRN_LIKELIHOOD_TYPE = "normal"
PRIOR_WEIGHT = 0.6
LIKELIHOOD_WEIGHT = 0.4
PT_SCORE_LOG_INTERVAL = 1000

GENERAL_GRN_PROFILE = {
    "GRN_USE_ALL_GENES_AS_REGULATORS": False,
    "GRN_PRIOR_MODE": "de_cdf",
    "GRN_PRIOR_TRANSFORM": "hybrid_minmax",
    "GRN_LL0_MODE": "current",
    "GRN_SCORE_MODEL": "ols",
    "GRN_COMBINE_MODE": "new_weighted",
    "GRN_INVALID_MODE": "new_masked",
    "GRN_REGULATOR_ORDER": "union1d",
}

BEELINE_GRN_PROFILE = {
    "GRN_USE_ALL_GENES_AS_REGULATORS": True,
    "GRN_PRIOR_MODE": "de_cdf",
    "GRN_PRIOR_TRANSFORM": "old_effect_percentile",
    "GRN_LL0_MODE": "old_effect",
    "GRN_SCORE_MODEL": "old_heteroscedastic",
    "GRN_COMBINE_MODE": "old_sum",
    "GRN_INVALID_MODE": "old_zero",
    "GRN_REGULATOR_ORDER": "auto",
}
BEELINE_SCENARIOS = {"GSD", "VSC", "MCAD", "HSC"}


def configure_algorithm_profile():
    """Select the internal GRN settings from the public scenario name."""
    cdf_mode = getattr(cfg, "GRN_CDF_MODE", "abs")
    if cdf_mode not in {"abs", "sign"}:
        raise ValueError("config.GRN_CDF_MODE must be 'abs' or 'sign'")

    scenario_prefix = str(cfg.SCENARIO).split("_", 1)[0].upper()
    is_beeline = cfg.MODE == "grn" and scenario_prefix in BEELINE_SCENARIOS
    profile = BEELINE_GRN_PROFILE if is_beeline else GENERAL_GRN_PROFILE
    for key, value in profile.items():
        setattr(cfg, key, value)
    cfg.GRN_CDF_MODE = cdf_mode
    profile_name = "beeline_legacy" if is_beeline else "general"
    print(f"algorithm_profile: {profile_name}, cdf_mode: {cdf_mode}")
    return profile_name


def z_posterior_Estep_with_pi(
    all_data,
    score_arrays,
    adj_matrix,
    tau,
    regulator_list,
    pi,
    pert_weight=0.3,
    nonpert_weight=0.7,
):
    """Driver-mode posterior E-step, matching the original experiment script.

    The default score is 0.3 * perturbation likelihood - 0.7 * non-perturbation
    likelihood, followed by the same dynamic scaling used in
    t2tcl_trans_linshi_scGPT.py.
    """

    def get_target_regulator(all_data, adj_matrix, gene_index, regulator_list):
        if len(np.where(adj_matrix[:, gene_index] == 1)[0]) > 0:
            regulating_tfs = regulator_list[np.where(adj_matrix[:, gene_index] == 1)[0]]
            regulator = all_data[:-1, regulating_tfs]
        else:
            regulator = None
        target = all_data[1:, gene_index]
        return target, regulator

    def logsumexp(a):
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    _, gene_num = all_data.shape
    lls = np.zeros([gene_num, 2])
    for i in range(gene_num):
        target, regulator = get_target_regulator(all_data, adj_matrix, i, regulator_list)
        if score_arrays["params_all"][i] is None:
            raise ValueError(f"no perturbation in gene {i}")

        param0 = score_arrays["params_all"][i][0]
        if isinstance(param0, np.ndarray):
            _, ll_pert = ols_linear_fit(regulator, target, parms=param0)
        else:
            if i in regulator_list:
                ll_pert = -np.inf
            else:
                ll_pert = np.nan

        parms_unpert = score_arrays["params_all"][i][1]
        _, ll_unperturb = ols_linear_fit(regulator, target, parms=parms_unpert)
        lls[i] = np.array([ll_pert, ll_unperturb])

    pert_num = adj_matrix.shape[0]
    log_ll_dataset = np.zeros(pert_num)
    assert pi.shape[1] == pert_num
    pi1 = pi.flatten()
    regulator_lls = lls[regulator_list]
    for pert_site in range(pert_num):
        log_prior = np.log(pi1[pert_site] + 1e-12)
        weighted_ll = (
            pert_weight * regulator_lls[pert_site, 0]
            - nonpert_weight * regulator_lls[pert_site, 1]
        )
        log_ll_dataset[pert_site] = log_prior + weighted_ll / tau

    delta = np.max(log_ll_dataset) - np.median(log_ll_dataset)
    dynamic_tau = delta * 0.3
    log_ll_dataset_scaled = log_ll_dataset / dynamic_tau
    log_norm = logsumexp(log_ll_dataset_scaled)
    posterior = np.exp(log_ll_dataset_scaled - log_norm)
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))
    norm_entropy = entropy / np.log(len(posterior))
    return posterior, lls, norm_entropy


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_tf_names(path, regulator_names):
    """Load and validate the optional TF subset used for GRN evaluation."""
    selected_tf = read_json(path)
    if not isinstance(selected_tf, list) or not selected_tf:
        raise ValueError("SELECTED_TF_JSON must contain a non-empty JSON list of TF symbols")
    if not all(isinstance(name, str) and name.strip() for name in selected_tf):
        raise ValueError("SELECTED_TF_JSON entries must be non-empty strings")
    if len(selected_tf) != len(set(selected_tf)):
        raise ValueError("SELECTED_TF_JSON must not contain duplicate TF symbols")

    missing = [name for name in selected_tf if name not in regulator_names]
    if missing:
        raise ValueError(
            "SELECTED_TF_JSON contains TFs that are absent from the regulator list: "
            + ", ".join(missing)
        )
    return selected_tf


def load_grn_evaluation_inputs(adata, regulator_names, selected_tf_json, ground_truth_path):
    """Load an aligned GRN ground truth and optional regulator-row subset."""
    if selected_tf_json is not None and ground_truth_path is None:
        raise ValueError("SELECTED_TF_JSON requires GROUND_TRUTH_PATH")
    if ground_truth_path is None:
        return None, None

    selected_tf = None
    selected_tf_index = None
    evaluation_regulators = list(regulator_names)
    if selected_tf_json is not None:
        selected_tf = load_selected_tf_names(selected_tf_json, regulator_names)
        selected_tf_index = [regulator_names.index(name) for name in selected_tf]
        evaluation_regulators = selected_tf

    ground_truth_path = str(ground_truth_path)
    suffix = Path(ground_truth_path).suffix.lower()
    if suffix == ".csv":
        ground_truth, _ = true_adj(adata, ground_truth_path, evaluation_regulators)
    elif suffix == ".npy":
        ground_truth = np.load(ground_truth_path)
    else:
        raise ValueError("GROUND_TRUTH_PATH must point to a .csv or .npy file")

    expected_shape = (len(evaluation_regulators), adata.n_vars)
    if ground_truth.ndim != 2 or ground_truth.shape != expected_shape:
        subset_note = " selected by SELECTED_TF_JSON" if selected_tf is not None else ""
        raise ValueError(
            "GROUND_TRUTH_PATH has shape "
            f"{ground_truth.shape}; expected {expected_shape} for regulator rows{subset_note} "
            "and AnnData gene columns"
        )
    return selected_tf_index, ground_truth


def load_cell_order_indices(path):
    """Load flat cell-order JSON and split it into regulator/target indices.

    If a sibling ``cell_order_meta.json`` exists, trajectory lengths are used to
    avoid creating artificial pairs across independent pseudo-time trajectories.
    Without metadata this falls back to the plain ``value[:-1]``/``value[1:]``
    split used by the original JSON path.
    """
    cell_order = read_json(path)
    meta_path = Path(path).with_name("cell_order_meta.json")
    if not meta_path.exists():
        return cell_order, *split_dict(cell_order)

    meta = read_json(meta_path)
    trajectory_lengths = meta.get("trajectory_lengths", {})
    all_regulator_index = {}
    all_target_index = {}
    for pert_name, ordered_indices in cell_order.items():
        lengths = trajectory_lengths.get(pert_name)
        if not lengths:
            all_regulator_index[pert_name] = ordered_indices[:-1]
            all_target_index[pert_name] = ordered_indices[1:]
            continue

        regulators = []
        targets = []
        start = 0
        for length in lengths:
            end = start + int(length)
            segment = ordered_indices[start:end]
            regulators.extend(segment[:-1])
            targets.extend(segment[1:])
            start = end
        if start != len(ordered_indices):
            raise ValueError(
                f"Trajectory lengths for {pert_name!r} in {meta_path} do not "
                f"sum to the number of indices in {path}."
            )
        all_regulator_index[pert_name] = regulators
        all_target_index[pert_name] = targets
    return cell_order, all_regulator_index, all_target_index


def build_grn_prior_matrix(adata, gene_name, regulator_list_name):
    if cfg.MODE != "grn":
        return np.ones((len(regulator_list_name), len(gene_name)))

    if cfg.GRN_PRIOR_MODE == "uniform":
        return np.ones((len(regulator_list_name), len(gene_name)))

    if cfg.GRN_PRIOR_MODE != "de_cdf":
        raise ValueError(f"Unknown GRN_PRIOR_MODE: {cfg.GRN_PRIOR_MODE}")

    _de_gene, _pert_influence, cdf_diff = get_DE_old_effect(
        adata,
        regulator_list_name,
        cdf_mode=cfg.GRN_CDF_MODE,
        control_names=DEFAULT_GRN_CONTROL_NAMES,
    )
    if cdf_diff.shape[0] == cdf_diff.shape[1]:
        np.fill_diagonal(cdf_diff, 0)

    prior_transform = getattr(cfg, "GRN_PRIOR_TRANSFORM", "old_effect_percentile")
    if prior_transform == "old_effect_percentile":
        return compute_percentile_matrix_old_effect(cdf_diff)
    if prior_transform == "current_percentile":
        return compute_percentile_matrix(cdf_diff)
    if prior_transform == "hybrid_minmax":
        zero_rows = np.all(cdf_diff == 0, axis=1)
        prior_matrix = np.zeros_like(cdf_diff, dtype=float)
        if np.any(~zero_rows):
            prior_matrix[~zero_rows, :] = minmax_normalize(cdf_diff[~zero_rows, :])
        if np.any(zero_rows):
            cor_adj = np.abs(caculate_cor_matrix(regulator_list_name, gene_name, adata, metric="cor"))
            if cor_adj.shape != cdf_diff.shape:
                raise ValueError(
                    "hybrid_minmax prior expects cor_adj and cdf_diff to have the same shape, "
                    f"got {cor_adj.shape} and {cdf_diff.shape}."
                )
            prior_matrix[zero_rows, :] = minmax_normalize(cor_adj)[zero_rows, :]
        return minmax_normalize(prior_matrix)

    raise ValueError(f"Unknown GRN_PRIOR_TRANSFORM: {prior_transform}")


def find_stop_point_relative(delta_lls, threshold=0.005, window=3):
    """Return the first low-gain index using the notebook stopping rule."""
    deltas = np.asarray(delta_lls, dtype=float).flatten()
    if len(deltas) == 0:
        return 0
    if window <= 0:
        raise ValueError("window must be positive")

    initial_gain = deltas[0]
    if initial_gain == 0:
        return len(deltas) - 1

    consecutive_low_gain = 0
    for i, delta in enumerate(deltas):
        ratio = delta / initial_gain
        if ratio < threshold:
            consecutive_low_gain += 1
        else:
            consecutive_low_gain = 0
        if consecutive_low_gain >= window:
            return max(0, i - window + 1)
    return len(deltas) - 1


def driver_stop_summary(delta_ll_history, threshold, window, current_iter):
    """Return stop metadata when the relative low-gain window is active."""
    if len(delta_ll_history) < window:
        return None
    stop_idx = find_stop_point_relative(delta_ll_history, threshold=threshold, window=window)
    if stop_idx == len(delta_ll_history) - 1:
        return None

    ratios = (np.asarray(delta_ll_history, dtype=float) / delta_ll_history[0]).tolist()
    return {
        "stopped": True,
        "trigger_iter": int(current_iter),
        "stop_idx": int(stop_idx),
        "threshold": float(threshold),
        "window": int(window),
        "initial_delta_ll": float(delta_ll_history[0]),
        "delta_ll_history": [float(x) for x in delta_ll_history],
        "relative_delta_ll_history": [float(x) for x in ratios],
    }


def log_pt_score_snapshot(logger, iter_idx, score, last_logged_iter, force=False):
    """Log the driver posterior every 1,000 iterations and at termination."""
    if iter_idx is None or iter_idx == last_logged_iter:
        return last_logged_iter
    if not force and (iter_idx + 1) % PT_SCORE_LOG_INTERVAL != 0:
        return last_logged_iter
    logger.add(to_row_iter_value(iter_idx, "score", score))
    logger.flush()
    return iter_idx


def write_final_tf_ranking(path, regulator_names, score):
    """Write TF names ordered by the final posterior score, highest first."""
    values = np.asarray(score, dtype=float).reshape(-1)
    if len(values) != len(regulator_names):
        raise ValueError(
            "Final posterior length does not match the regulator list: "
            f"{len(values)} != {len(regulator_names)}"
        )
    sortable_values = np.nan_to_num(values, nan=-np.inf)
    order = np.argsort(-sortable_values, kind="stable")
    ranking = [str(regulator_names[index]) for index in order]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ranking, handle, indent=2)
        handle.write("\n")
    return ranking


def init_worker(context):
    GLOBAL.clear()
    GLOBAL.update(context)


def invalid_result(i, j):
    if GLOBAL.get("invalid_mode") == "old_zero":
        return i, j, np.array([0.0, 0.0]), 0.0, 0.0, None
    return i, j, np.array([-99999.0, -99999.0]), -99999.0, -99999.0, None


def normalize_param_ll(params, log_likelihood):
    if len(params) == 1:
        return (np.nan, params[0]), (np.nan, log_likelihood[0])
    if len(params) != 2:
        raise ValueError("params length is wrong")

    param0 = params[0]
    ll0 = log_likelihood[0]
    param0_missing = param0 is None or (isinstance(param0, float) and np.isnan(param0))
    ll0_missing = np.isnan(ll0)
    if param0_missing or ll0_missing:
        return (np.nan, params[1]), (np.nan, log_likelihood[1])
    return (params[0], params[1]), (log_likelihood[0], log_likelihood[1])


def process_pair(args):
    """Score one candidate edge regulator j -> target i."""
    i, j = args
    data = GLOBAL["data"]
    adj_matrix = GLOBAL["adj_matrix"]
    weight = GLOBAL["weight"]
    ll0 = GLOBAL["ll0"]
    gene_name = GLOBAL["gene_name"]
    regulator_list = GLOBAL["regulator_list"]
    all_pert_sites = GLOBAL["all_pert_sites"]
    all_regulator_index = GLOBAL["all_regulator_index"]
    all_target_index = GLOBAL["all_target_index"]
    prior_matrix = GLOBAL["prior_matrix"]
    valid_mask = GLOBAL["valid_mask"]

    target_name = gene_name[i]
    regulator_gene_index = int(regulator_list[j])
    if i == regulator_gene_index or adj_matrix[j, i] == 1:
        return invalid_result(i, j)

    if target_name in all_pert_sites:
        target2regulator_index = all_pert_sites.index(target_name)
        groups = split_number(len(all_pert_sites) + 1, target2regulator_index + 1)
    else:
        groups = [list(range(len(all_pert_sites) + 1))]

    target = data[:, i]
    old_tf_indices = np.where(adj_matrix[:, i] == 1)[0]
    regulator_order = GLOBAL.get("regulator_order", "union1d")
    if regulator_order == "auto":
        old_compat = (
            GLOBAL.get("score_model") == "old_heteroscedastic"
            and GLOBAL.get("combine_mode") == "old_sum"
        )
        regulator_order = "candidate_first" if old_compat else "union1d"

    if regulator_order == "candidate_first":
        regulator = data[:, regulator_gene_index].reshape(-1, 1)
        if len(old_tf_indices) > 0:
            old_regulator_gene_indices = regulator_list[old_tf_indices].astype(int)
            regulator = np.concatenate([regulator, data[:, old_regulator_gene_indices]], axis=1)
    elif regulator_order == "union1d":
        tf_indices = np.union1d(old_tf_indices, [j])
        regulator_gene_indices = regulator_list[tf_indices].astype(int)
        regulator = data[:, regulator_gene_indices]
        if regulator.ndim == 1:
            regulator = regulator.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown regulator_order: {regulator_order}")

    target = reorder_array(target, all_target_index)
    regulator = reorder_array(regulator, all_regulator_index)
    regulator, target, weight_down = downsample_by_mask(
        regulator,
        target,
        weight,
        cfg.DOWN_SAMPLE_SIZE,
        random_state=cfg.RANDOM_STATE,
    )

    params, log_likelihood = m_step_fast(
        regulator,
        target,
        weight_down,
        groups,
        model=GLOBAL.get("score_model", "ols"),
    )
    if GLOBAL.get("mode") == "driver":
        if len(params) == 1:
            param = (np.nan, params[0])
            ll = (np.nan, log_likelihood[0])
        elif len(params) == 2:
            if params[0].sum() == 0 and log_likelihood[0].sum() == 0:
                param = (np.nan, params[1])
                ll = (np.nan, log_likelihood[1])
            else:
                param = (params[0], params[1])
                ll = (log_likelihood[0], log_likelihood[1])
        else:
            raise ValueError("params length is wrong")
    else:
        param, ll = normalize_param_ll(params, log_likelihood)

    score_improve, ll_val = get_score_improvement_EM(ll, ll0[i], use_intervene=1)
    if valid_mask[j, i]:
        prior_improve = compute_prior_improvement(prior_matrix[j, i], epsilon=1e-5)
    else:
        prior_improve = -99999
    return i, j, ll_val, score_improve, prior_improve, param


def initialize_ll0(data, gene_name, all_pert_sites, all_target_index, weight_init):
    """Compute baseline likelihood/parameters from scratch for both modes."""
    gene_num = data.shape[1]
    ll0 = np.full((gene_num, 2), np.nan)
    params_all = np.empty(gene_num, dtype=object)

    for i, variable in enumerate(tqdm(data.T, desc="initializing ll0")):
        target_name = gene_name[i]
        if target_name in all_pert_sites:
            target2regulator_index = all_pert_sites.index(target_name)
            groups = split_number(len(all_pert_sites) + 1, target2regulator_index + 1)
        else:
            groups = [list(range(len(all_pert_sites) + 1))]

        variable = reorder_array(variable, all_target_index)
        params, log_likelihood = m_step_fast(None, variable, weight_init, groups)
        if len(params) == 1:
            params_all[i] = (np.nan, params[0])
            ll0[i, 1] = log_likelihood[0]
        elif len(params) == 2:
            params_all[i] = (params[0], params[1])
            ll0[i, 0] = log_likelihood[0]
            ll0[i, 1] = log_likelihood[1]
    return ll0, params_all


def initialize_grn_ll0(data, adata, gene_name, all_regulator_index, all_target_index):
    gene_num = data.shape[1]
    all_index = np.arange(data.shape[0])
    ll0 = np.full((gene_num, 2), np.nan)
    params_all = np.empty(gene_num, dtype=object)

    for i in tqdm(range(gene_num), desc="initializing grn ll0"):
        target_name = gene_name[i]
        intervene_index = np.where(adata.obs["pert"] == target_name)[0]
        un_index = all_index[~np.isin(all_index, intervene_index)]
        pseudo_intervene, pseudo_un = get_intervene_index(all_regulator_index, all_target_index, target_name)
        intervene_denominator = len(pseudo_intervene[0]) if pseudo_intervene[0] is not None else len(intervene_index)
        un_denominator = len(pseudo_un[0]) if pseudo_un[0] else len(un_index)

        if len(intervene_index) != 0:
            ll_intervene = -log_likelihood_0(data[intervene_index, i], type=GRN_LIKELIHOOD_TYPE)
            if cfg.GRN_SCORE_MODEL in {"ols", "old_heteroscedastic"} and intervene_denominator:
                ll_intervene /= intervene_denominator
            ll0[i, 0] = ll_intervene
        elif cfg.GRN_SCORE_MODEL not in {"ols", "old_heteroscedastic"}:
            ll0[i, 0] = 0.0

        ll_unintervene = -log_likelihood_0(data[un_index, i], type=GRN_LIKELIHOOD_TYPE)
        if cfg.GRN_SCORE_MODEL in {"ols", "old_heteroscedastic"} and un_denominator:
            ll_unintervene /= un_denominator
        ll0[i, 1] = ll_unintervene
        params_all[i] = None
    return ll0, params_all


def score_candidate_edges(tasks, context):
    if cfg.POOL_SIZE == 1:
        init_worker(context)
        return [process_pair(task) for task in tqdm(tasks, total=len(tasks), desc="scoring")]
    with Pool(processes=cfg.POOL_SIZE, initializer=init_worker, initargs=(context,)) as pool:
        return list(tqdm(pool.imap_unordered(process_pair, tasks, chunksize=10), total=len(tasks)))


def update_score_arrays(results, score_arrays):
    for i, j, ll_val, score_improve, prior_improve, param in results:
        score_arrays["ll_val_normalized"][i, j] = ll_val
        score_arrays["score_improvement"][i, j] = score_improve
        score_arrays["prior_improvement"][i, j] = prior_improve
        score_arrays["tem_param"][i, j] = param


def combine_scores(score_arrays, perturbation_mask):
    if cfg.MODE == "grn":
        if cfg.GRN_COMBINE_MODE == "old_sum":
            return normaliz(score_arrays["prior_improvement"]) + normaliz(score_arrays["score_improvement"])
        if cfg.GRN_COMBINE_MODE != "new_weighted":
            raise ValueError(f"Unknown GRN_COMBINE_MODE: {cfg.GRN_COMBINE_MODE}")
        prior_n = masked_standardize(score_arrays["prior_improvement"])
        likelihood_n = masked_standardize(score_arrays["score_improvement"])
        return np.where(
            perturbation_mask,
            PRIOR_WEIGHT * prior_n + LIKELIHOOD_WEIGHT * likelihood_n,
            likelihood_n,
        )
    if cfg.MODE == "driver":
        return score_arrays["score_improvement"]
    raise ValueError(f"Unknown MODE: {cfg.MODE}")


def select_edges(total_score_improvement):
    return get_target_index(total_score_improvement, cfg.K)


def build_metrics_logger(path):
    if cfg.MODE != "grn" or getattr(cfg, "GROUND_TRUTH_PATH", None) is None:
        return None
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Iteration",
                "regulator",
                "target",
                "Precision",
                "Recall",
                "F1",
            ]
        )
    return path


def append_metrics(path, iter_idx, adj_matrix, regulator_index, target_index, selected_tf_index, ground_truth):
    if path is None:
        return

    metrics = (np.nan, np.nan, np.nan)
    if selected_tf_index is not None and ground_truth is not None:
        metrics = evaluate_network_reconstruction(
            adj_matrix[selected_tf_index, :],
            np.abs(ground_truth),
        )
    elif ground_truth is not None:
        metrics = evaluate_network_reconstruction(adj_matrix, np.abs(ground_truth))

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([iter_idx, regulator_index, target_index, *metrics])


def main():
    if cfg.MODE not in {"grn", "driver"}:
        raise ValueError("config.MODE must be 'grn' or 'driver'")
    configure_algorithm_profile()
    driver_early_stop = bool(
        getattr(cfg, "DRIVER_EARLY_STOP", DEFAULT_DRIVER_EARLY_STOP)
    )

    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ensure_pert_column(ensure_gene_column(sc.read_h5ad(cfg.ADATA_PATH)))
    data = as_dense(adata.X)
    gene_name = [str(x) for x in adata.var["gene"]]
    gene_num = len(gene_name)

    if cfg.MODE == "grn" and cfg.GRN_USE_ALL_GENES_AS_REGULATORS:
        regulator_list = np.arange(gene_num)
        regulator_list_name = gene_name
    else:
        regulator_list, regulator_list_name = get_regulator_list(adata)
        regulator_list_name = [str(x) for x in regulator_list_name]
    tf_num = len(regulator_list)
    print("tf_num:", tf_num)

    pert_dict, all_regulator_index, all_target_index = load_cell_order_indices(cfg.CELLORDER_FILE)
    test_pert_sites = list(DEFAULT_TEST_PERT_SITES)
    if cfg.MODE == "driver":
        target_pert_label = getattr(cfg, "TARGET_PERT_LABEL", DEFAULT_TEST_PERT_SITES[0])
        if not isinstance(target_pert_label, str) or not target_pert_label:
            raise ValueError("config.TARGET_PERT_LABEL must be a non-empty string")
        if target_pert_label not in pert_dict:
            raise ValueError(
                f"TARGET_PERT_LABEL {target_pert_label!r} is absent from cell_order.json"
            )
        observed_pert = set(adata.obs["pert"].astype(str))
        if target_pert_label not in observed_pert:
            raise ValueError(
                f"TARGET_PERT_LABEL {target_pert_label!r} is absent from adata.obs['pert']"
            )
        if len(pert_dict[target_pert_label]) < 2:
            raise ValueError(
                f"TARGET_PERT_LABEL {target_pert_label!r} requires at least two ordered cells"
            )
        test_pert_sites = [target_pert_label]
        print(
            f"target_pert_label: {target_pert_label}, "
            f"driver_early_stop: {driver_early_stop}"
        )

    all_pert_sites = sorted(
        (set(all_regulator_index.keys()) | set(regulator_list_name))
        - set(test_pert_sites)
        - set(DEFAULT_GRN_CONTROL_NAMES if cfg.MODE == "grn" else ["CTRL"])
    )

    prior_matrix = build_grn_prior_matrix(adata, gene_name, regulator_list_name)
    valid_mask = np.ones_like(prior_matrix, dtype=bool)
    perturbation_mask = valid_mask.T

    pt_score = np.full(tf_num, 1 / tf_num).reshape(1, -1)
    new_pt_score = pt_score.copy()
    weight = build_responsibility_matrix(
        all_regulator_index,
        regulator_list_name,
        test_pert_sites,
        new_pt_score,
        control_names=DEFAULT_GRN_CONTROL_NAMES if cfg.MODE == "grn" else None,
    )

    if cfg.MODE == "grn" and cfg.GRN_LL0_MODE == "old_effect":
        ll0, params_all = initialize_grn_ll0(data, adata, gene_name, all_regulator_index, all_target_index)
    else:
        ll0, params_all = initialize_ll0(data, gene_name, all_pert_sites, all_target_index, weight)

    adj_matrix = np.zeros((tf_num, gene_num), dtype=int)
    adj_matrix_auc = np.zeros((tf_num, gene_num))
    score_arrays = {
        "ll_val_normalized": np.zeros((gene_num, tf_num, 2)),
        "score_improvement": np.zeros((gene_num, tf_num)),
        "prior_improvement": np.zeros((gene_num, tf_num)),
        "tem_param": np.empty((gene_num, tf_num), dtype=object),
        "ll0": ll0,
        "params_all": params_all,
    }

    target_index = None
    new_edge_target_index = None

    ll_logger = BufferedCSVLogger(output_dir / "ll.csv", flush_every=cfg.FLUSH_EVERY)
    pts_logger = BufferedCSVLogger(output_dir / "pt_score.csv", flush_every=1) if cfg.MODE == "driver" else None
    metrics_logger_path = build_metrics_logger(output_dir / "metrics_log.csv")

    selected_tf_json = getattr(cfg, "SELECTED_TF_JSON", None)
    ground_truth_path = getattr(cfg, "GROUND_TRUTH_PATH", None)
    if cfg.MODE != "grn" and (selected_tf_json is not None or ground_truth_path is not None):
        raise ValueError("SELECTED_TF_JSON and GROUND_TRUTH_PATH are supported only in MODE='grn'")
    selected_tf_index, ground_truth = load_grn_evaluation_inputs(
        adata,
        regulator_list_name,
        selected_tf_json,
        ground_truth_path,
    )

    effective_max_iter = cfg.MAX_ITER // cfg.K

    print("begin!")
    delta_ll_history = []
    last_completed_iter = None
    last_driver_iter = None
    last_pt_score_iter = None
    for iter_idx in range(effective_max_iter):
        if cfg.MODE == "driver":
            last_driver_iter = iter_idx
            if iter_idx != 0:
                test_pert_data = adata[pert_dict[test_pert_sites[0]], :].X
                sample_num = test_pert_data.shape[0] - 1
                tau = sample_num * TAU_MULTIPLIER
                pt_score, _, _ = z_posterior_Estep_with_pi(
                    test_pert_data,
                    score_arrays,
                    adj_matrix,
                    tau,
                    regulator_list,
                    new_pt_score,
                    POSTERIOR_PERT_WEIGHT,
                    POSTERIOR_NONPERT_WEIGHT,
                )
                pt_score = pt_score.reshape(1, -1)
                delta_pt = np.abs(pt_score.flatten() - new_pt_score.flatten())
                changed = np.where(delta_pt > POSTERIOR_DELTA_THRESHOLD)[0]
                new_edge_target_index = list(set(target_index) | set(regulator_list[changed].tolist()))
                new_pt_score = pt_score

            weight = build_responsibility_matrix(
                all_regulator_index,
                regulator_list_name,
                test_pert_sites,
                new_pt_score,
            )
            score_arrays = update_ll0_parallel(
                score_arrays,
                weight,
                data,
                adj_matrix,
                regulator_list,
                gene_name,
                all_pert_sites,
                all_regulator_index,
                all_target_index,
                cfg.DOWN_SAMPLE_SIZE,
            )
            ll0 = score_arrays["ll0"]
            params_all = score_arrays["params_all"]

        elif cfg.MODE == "grn":
            # GRN mode keeps posterior/weight fixed and does not update ll0 each iteration.
            new_edge_target_index = target_index

        if new_edge_target_index is None:
            tasks = [(i, j) for i in range(gene_num) for j in range(tf_num)]
        else:
            tasks = [(int(i), j) for i in new_edge_target_index for j in range(tf_num)]

        context = {
            "data": data,
            "adj_matrix": adj_matrix,
            "weight": weight,
            "ll0": ll0,
            "gene_name": gene_name,
            "regulator_list": regulator_list,
            "all_pert_sites": all_pert_sites,
            "all_regulator_index": all_regulator_index,
            "all_target_index": all_target_index,
            "prior_matrix": prior_matrix,
            "valid_mask": valid_mask,
            "mode": cfg.MODE,
            "score_model": cfg.GRN_SCORE_MODEL if cfg.MODE == "grn" else "ols",
            "invalid_mode": cfg.GRN_INVALID_MODE if cfg.MODE == "grn" else "new_masked",
            "combine_mode": cfg.GRN_COMBINE_MODE if cfg.MODE == "grn" else "new_weighted",
            "regulator_order": cfg.GRN_REGULATOR_ORDER if cfg.MODE == "grn" else "union1d",
        }
        results = score_candidate_edges(tasks, context)
        update_score_arrays(results, score_arrays)

        total_score_improvement = combine_scores(score_arrays, perturbation_mask)
        target_index, regulator_index, delta_ll = select_edges(total_score_improvement)
        if len(target_index) == 0:
            if cfg.MODE == "driver":
                last_pt_score_iter = log_pt_score_snapshot(
                    pts_logger,
                    iter_idx,
                    new_pt_score,
                    last_pt_score_iter,
                    force=True,
                )
            print(f"iteration={iter_idx}, no selectable edge remains; stop.")
            break
        delta_ll_values = np.asarray(delta_ll, dtype=float).flatten().tolist()
        delta_ll_history.extend(delta_ll_values)
        print(f"iteration={iter_idx}, target={target_index}, regulator={regulator_index}, delta={delta_ll}")

        for target_id, regulator_id in zip(target_index, regulator_index):
            ll0[target_id] = score_arrays["ll_val_normalized"][target_id, regulator_id]
            adj_matrix[regulator_id, target_id] = 1
            adj_matrix_auc[regulator_id, target_id] = iter_idx + 1
            params_all[target_id] = score_arrays["tem_param"][target_id, regulator_id]

        score_arrays["ll0"] = ll0
        score_arrays["params_all"] = params_all
        last_completed_iter = iter_idx

        if iter_idx % cfg.SAVE_EVERY == 0 or iter_idx == 1:
            np.save(output_dir / f"cp_adj_{iter_idx}.npy", adj_matrix_auc)
            with open(output_dir / f"score_arrays_{iter_idx}.pkl", "wb") as f:
                pickle.dump(score_arrays, f)

        ll_logger.add(to_row_iter_value(iter_idx, "ll", delta_ll))
        ll_logger.maybe_flush(iter_idx)

        if cfg.MODE == "driver":
            last_pt_score_iter = log_pt_score_snapshot(
                pts_logger,
                iter_idx,
                new_pt_score,
                last_pt_score_iter,
            )

            if driver_early_stop:
                summary = driver_stop_summary(
                    delta_ll_history,
                    threshold=cfg.DRIVER_STOP_THRESHOLD,
                    window=cfg.DRIVER_STOP_WINDOW,
                    current_iter=iter_idx,
                )
                if summary is not None and iter_idx >= DRIVER_STOP_MIN_ITER:
                    last_pt_score_iter = log_pt_score_snapshot(
                        pts_logger,
                        iter_idx,
                        new_pt_score,
                        last_pt_score_iter,
                        force=True,
                    )
                    summary_path = output_dir / "driver_stop_summary.json"
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)
                    print(
                        "driver early stop: "
                        f"trigger_iter={summary['trigger_iter']}, "
                        f"stop_idx={summary['stop_idx']}, "
                        f"threshold={summary['threshold']}, "
                        f"window={summary['window']}"
                    )
                    break

        if cfg.MODE == "grn":
            append_metrics(
                metrics_logger_path,
                iter_idx,
                adj_matrix,
                regulator_index,
                target_index,
                selected_tf_index,
                ground_truth,
            )

    ll_logger.flush()
    if cfg.MODE == "driver":
        last_pt_score_iter = log_pt_score_snapshot(
            pts_logger,
            last_driver_iter,
            new_pt_score,
            last_pt_score_iter,
            force=True,
        )
        pts_logger.flush()
        write_final_tf_ranking(
            output_dir / "final_tf_ranking.json",
            regulator_list_name,
            new_pt_score,
        )

    if last_completed_iter is not None:
        final_adj_path = output_dir / f"cp_adj_{last_completed_iter}.npy"
        final_score_path = output_dir / f"score_arrays_{last_completed_iter}.pkl"
        if not final_adj_path.exists():
            np.save(final_adj_path, adj_matrix_auc)
        if not final_score_path.exists():
            with open(final_score_path, "wb") as f:
                pickle.dump(score_arrays, f)


if __name__ == "__main__":
    main()
