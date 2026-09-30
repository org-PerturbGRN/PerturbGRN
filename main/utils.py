import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import ks_2samp
from sympy.abc import sigma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma  # 导入 scipy.special.gamma
from sympy.matrices.expressions.blockmatrix import bounds
import pandas as pd
import json
from scipy.stats import mannwhitneyu
from scipy.stats import wasserstein_distance
import numpy as np
import os
import pickle
import re
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Pipe
import csv
from scipy.stats import rankdata
import numpy as np
from joblib import Parallel, delayed
import scanpy as sc


def as_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def ensure_gene_column(adata):
    if "gene" not in adata.var.columns:
        if "gene_name" in adata.var.columns:
            adata.var["gene"] = adata.var["gene_name"]
        else:
            adata.var["gene"] = adata.var_names.astype(str)
    return adata


def ensure_pert_column(adata):
    if "pert" not in adata.obs.columns and "gene" in adata.obs.columns:
        adata.obs["pert"] = adata.obs["gene"]
    return adata


def true_adj(adata, refnet_path, tf_list=None):
    df_refnet = pd.read_csv(refnet_path)
    genes = list(adata.var["gene"])
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}

    if tf_list is None:
        tf_to_index = gene_to_index
        row_num = len(genes)
        index = genes
    else:
        index = list(tf_list)
        tf_to_index = {tf: idx for idx, tf in enumerate(index)}
        row_num = len(index)

    adj_matrix = np.zeros((row_num, len(genes)), dtype=int)
    for _, row in df_refnet.iterrows():
        gene1, gene2, edge_type = row["Gene1"], row["Gene2"], row["Type"]
        if gene1 in tf_to_index and gene2 in gene_to_index:
            adj_matrix[tf_to_index[gene1], gene_to_index[gene2]] = 1 if edge_type == "+" else -1

    adj_df = pd.DataFrame(adj_matrix, index=index, columns=genes)
    return adj_matrix, adj_df


DEFAULT_TF_LIST_PATH = Path(__file__).resolve().parents[1] / "Data/reference/Homo_sapiens_TF.txt"


def get_regulator_list(adata, filter_gene=None, tf_list_path=None):
   """
    输入adata，输出这个adata中的gene是regulator的index集合
    注意：一定要保证通过adata.var["gene_name"]得到基因的名字集合。
    :param adata: adata
    :return: indexs. example: array[1,2,3,4,5]
   """
   gene_list = np.array(list(adata.var["gene"]))
   if filter_gene is None:
     path = Path(tf_list_path) if tf_list_path is not None else DEFAULT_TF_LIST_PATH
     if not path.exists():
       raise FileNotFoundError(
         f"Bundled TF list not found: {path}. Restore Data/reference/Homo_sapiens_TF.txt."
       )
     df = pd.read_csv(path, sep="\t")
     # 提取 Symbol 列为列表
     symbol_list = df["Symbol"].tolist()
     mask = np.isin(gene_list, symbol_list)  # 生成布尔掩码
     indices = np.where(mask)[0]  # 获取符合条件的索引
     TF_list = list(gene_list[indices]) # 输出TF的名字
   else:
     if isinstance(filter_gene, (str, os.PathLike)):
       with open(filter_gene, encoding="utf-8") as user_file:
         TF_list = json.load(user_file)
     else:
       TF_list = list(filter_gene)
     mask = np.isin(gene_list, TF_list)
     indices = np.where(mask)[0]
     TF_list = list(gene_list[indices])
   return indices,TF_list


def get_DE(adata,unknown_target, TF_name_list = None):
    """
    DE_gene[i,j]=1表示第i个基因的扰动，会导致第j个基因的分布发生显著的变化
    pert_influence[i,j]表示，第i个基因的扰动，第j个基因到底会变化多少
    """
    data_ctrl = adata[(adata.obs["pert"]=="test1")|(adata.obs["pert"]=="CTRL")|(adata.obs["pert"]=="non-targeting")].X


    if unknown_target ==1:  
    # 假如说这个时候并不清楚每一套数据的靶点具体是什么（adata中都没有标记好），那么有多少套数据就有多少行；
      dataset_num = len(set(list(adata.obs["pert"])))-1
      DE_gene = np.zeros([dataset_num,data_ctrl.shape[1]])
      pert_influence = np.zeros([dataset_num,data_ctrl.shape[1]])
      cdf_diff = np.zeros([dataset_num, data_ctrl.shape[1]])      
      exclude = {"test1", "CTRL", "non-targeting"}    
      valid_perts = sorted([p for p in set(list(adata.obs["pert"])) if p not in exclude])
      for index, i in enumerate(valid_perts):      
          data_pert = adata[adata.obs["pert"]==i].X
          for j in range(data_ctrl.shape[1]):
              pert_influence[index,j] = wasserstein_distance(data_ctrl[:, j], data_pert[:, j])
              ##################################
              # ctrl = data_ctrl[:, j][data_ctrl[:, j] != 0]
              # pert = data_pert[:, j][data_pert[:, j] != 0]
              col_ctrl = data_ctrl[:, j]
              col_pert = data_pert[:, j]

              # 对 ctrl 的独立逻辑
              if (col_ctrl == 0).all():
                  ctrl = col_ctrl
              else:
                  ctrl = col_ctrl[col_ctrl != 0]

              # 对 pert 的独立逻辑
              if (col_pert == 0).all():
                  pert = col_pert
              else:
                  pert = col_pert[col_pert != 0]


              wd = wasserstein_distance(ctrl, pert)
              mean_diff = pert.mean() - ctrl.mean()  # 方向
              cdf_diff[index,j]  = np.sign(mean_diff) * wd
              #########################
              #cdf_diff[index,j]  = wasserstein_distance((data_ctrl[:, j])[data_ctrl[:, j]!=0], (data_pert[:, j])[data_pert[:, j]!=0])
              ##########################
              _, p_value = mannwhitneyu(data_ctrl[:, j], data_pert[:, j]) # 非参数检验
  
              if p_value < 0.05:
                  DE_gene[index,j]=1         
    
    else:
      if TF_name_list is not None:
          TF_num = len(TF_name_list)
      else:
          TF_num = data_ctrl.shape[1]
      DE_gene = np.zeros([TF_num,data_ctrl.shape[1]])
      pert_influence = np.zeros([TF_num,data_ctrl.shape[1]])
      cdf_diff = np.zeros([TF_num, data_ctrl.shape[1]])
      for i in set(list(adata.obs["pert"])): # 对每一种
          # if i == "test1" or i == "CTRL" or i == "non-targeting" or i not in TF_name_list:
          #     continue
          if i in {"test1", "CTRL", "non-targeting"}:
              continue
          if TF_name_list is not None and i not in TF_name_list:
              continue  # 仅在 TF_list 存在时进行筛选
  
          if TF_name_list is not None:
              index = TF_name_list.index(i)
          else:
              df = adata.var
              index = int(df.index[df['gene'] == i].tolist()[0])
          data_pert = adata[adata.obs["pert"]==i].X
          for j in range(data_ctrl.shape[1]):
              pert_influence[index,j] = wasserstein_distance(data_ctrl[:, j], data_pert[:, j])
              ##################################
              # ctrl = data_ctrl[:, j][data_ctrl[:, j] != 0]
              # pert = data_pert[:, j][data_pert[:, j] != 0]
              col_ctrl = data_ctrl[:, j]
              col_pert = data_pert[:, j]

              # 对 ctrl 的独立逻辑
              if (col_ctrl == 0).all():
                  ctrl = col_ctrl
              else:
                  ctrl = col_ctrl[col_ctrl != 0]

              # 对 pert 的独立逻辑
              if (col_pert == 0).all():
                  pert = col_pert
              else:
                  pert = col_pert[col_pert != 0]
              wd = wasserstein_distance(ctrl, pert)
              mean_diff = pert.mean() - ctrl.mean()  # 方向
              cdf_diff[index,j]  = np.sign(mean_diff) * wd
              #########################
              #cdf_diff[index,j]  = wasserstein_distance((data_ctrl[:, j])[data_ctrl[:, j]!=0], (data_pert[:, j])[data_pert[:, j]!=0])
              ##########################
              _, p_value = mannwhitneyu(data_ctrl[:, j], data_pert[:, j]) # 非参数检验
  
              if p_value < 0.05:
                  DE_gene[index,j]=1

    return DE_gene,pert_influence,cdf_diff


def _nonzero_or_all(values):
    values = np.asarray(values)
    if np.all(values == 0):
        return values
    return values[values != 0]


def get_DE_old_effect(adata, TF_name_list=None, cdf_mode="abs", control_names=None):
    if cdf_mode not in {"abs", "sign"}:
        raise ValueError("cdf_mode must be 'abs' or 'sign'.")
    control_names = set(control_names or ["test1", "CTRL", "non-targeting"])
    data_ctrl = as_dense(adata[adata.obs["pert"].isin(control_names)].X)

    if TF_name_list is not None:
        tf_num = len(TF_name_list)
    else:
        tf_num = data_ctrl.shape[1]
    DE_gene = np.zeros([tf_num, data_ctrl.shape[1]])
    pert_influence = np.zeros([tf_num, data_ctrl.shape[1]])
    cdf_diff = np.zeros([tf_num, data_ctrl.shape[1]])

    for pert_name in set(list(adata.obs["pert"])):
        if pert_name in control_names:
            continue
        if TF_name_list is not None and pert_name not in TF_name_list:
            continue
        if TF_name_list is not None:
            index = TF_name_list.index(pert_name)
        else:
            index = int(adata.var.index[adata.var["gene"] == pert_name].tolist()[0])
        data_pert = as_dense(adata[adata.obs["pert"] == pert_name].X)
        for j in range(data_ctrl.shape[1]):
            pert_influence[index, j] = wasserstein_distance(data_ctrl[:, j], data_pert[:, j])
            if cdf_mode == "abs":
                ctrl = _nonzero_or_all(data_ctrl[:, j])
                pert = _nonzero_or_all(data_pert[:, j])
            else:
                ctrl = _nonzero_or_all(data_ctrl[:, j])
                pert = _nonzero_or_all(data_pert[:, j])
            wd = wasserstein_distance(ctrl, pert)
            cdf_diff[index, j] = wd if cdf_mode == "abs" else np.sign(pert.mean() - ctrl.mean()) * wd
            _, p_value = mannwhitneyu(data_ctrl[:, j], data_pert[:, j])
            if p_value < 0.05:
                DE_gene[index, j] = 1
    return DE_gene, pert_influence, cdf_diff

def find_indices(lst, target, return_all=True):
    """
    在列表中查找指定元素的索引位置。

    参数：
        lst (list): 要搜索的列表。
        target: 要查找的元素。
        return_all (bool): 如果为 True，返回所有匹配的索引；否则只返回最后一个。

    返回：
        list 或 int: 所有匹配的索引，或最后一个索引。
    """
    indices = [i for i, x in enumerate(lst) if x == target]

    if not indices:
        return None  # 没有匹配
    return indices if return_all else indices[-1]
def caculate_cor(tf_index, adata):
    # 选择第 0 列作为目标变量（target）
    X = adata.X
    target = X[:, tf_index]
    target_centered = target - target.mean()

    X_centered = X - X.mean(axis=0)
    cov = np.dot(X_centered.T, target_centered) / (len(target) - 1)

    std_target = np.std(target, ddof=1)
    std_all = np.std(X, axis=0, ddof=1)

    correlations = cov / (std_target * std_all)
    return correlations
def caculate_cor_matrix(regulator_list1, gene_name, adata_ctrl, metric="cor"):  # cor:计算相关性；mi:计算互信息
    adj = np.zeros((len(regulator_list1), len(gene_name)), dtype=float)
    for i, tf_name in enumerate(tqdm(regulator_list1, desc="Computing correlation/MI")):
        tf_index = find_indices(gene_name, tf_name, return_all=False)
        if metric == "cor":
            adj[i, :] = np.abs(caculate_cor(tf_index, adata_ctrl))
        elif metric == "mi":
            adj[i, :] = np.abs(calculate_mutual_info(tf_index, adata_ctrl))
        adj[i, tf_index] = 0  # 去掉自环
    return adj


def split_dict(input_dict):
    dict_first = {key: value[:-1] for key, value in input_dict.items()}
    dict_second = {key: value[1:] for key, value in input_dict.items()}
    return dict_first, dict_second


def log_likelihood_0(data, type="zi-normal"):
    """
    :param data:需要估计的数据
    :param type:
    :return: 负！！对数似然函数
    """
    # 计算零膨胀部分的概率
    if type == "zi-normal":
        pi, mu, sigma = estimate_zero_inflated_normal(data)
        zero_likelihood = pi + (1 - pi) * norm.pdf(0, loc=mu, scale=sigma)
        # 计算非零部分的概率
        nonzero_likelihood = (1 - pi) * norm.pdf(data, loc=mu, scale=sigma)
        # 合并零膨胀和非零部分的似然
        likelihood = np.where(data == 0, zero_likelihood, nonzero_likelihood)
        epsilon = 1e-10  # 一个非常小的常数，防止出现 log(0)
        neg_log_likelihood = -np.sum(np.log(np.maximum(likelihood, epsilon)))
        return neg_log_likelihood
    elif type == "normal":
        n = len(data)
        mu = np.mean(data)
        sigma = np.std(data, ddof=0)
        sigma = np.maximum(sigma, 1e-10)
        ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * np.sum((data - mu) ** 2) / (sigma ** 2)
        return -1 * ll
    elif type == "zi-gamma":
        alpha, beta, pi = estimate_zero_inflated_gamma(data)
        # zero_prob = np.sum(pi * (data==0))
        # non_zero_prob = np.sum((1 - pi) * gamma_pdf(data, alpha, beta))
        # # 总概率是零膨胀部分和 Gamma 部分的和
        # ll = zero_prob + non_zero_prob
        neg_ll = log_likelihood_gamma([alpha, beta, pi], data, 0)  # 返回的是负对数似然！
        return neg_ll



def log_likelihood_init(data):
    """
    :param data:需要估计的数据
    :param type:
    :return: 正常的对数似然函数
    """
    # 计算零膨胀部分的概率
    n = len(data)
    mu = np.mean(data)
    sigma = np.std(data, ddof=0)
    sigma = np.maximum(sigma, 1e-10)
    ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * np.sum((data - mu) ** 2) / (sigma ** 2)
    return ll,mu,sigma




def z_posterior_Estep1(adata, adj_matrix, tau,regulator_list, lls = None,target_index = None):
    """
    计算隐变量的后验分布（仅限于adata中有一种扰动数据的场景。如果需要拓展，需要再计算一下） #？？
    p(z|D,G) = p(D|z,G)p(z|G)/(sum_z p(D|z,G)p(z|G))
    1. 计算每一个targt在all data、ctrl data、pert data下所对应的对数似然（输入data、adj，输出lls ）get_target_reulator、ols_linear_fit
    2. 计算在每一种扰动的情况下，整个数据集的 平均 对数似然（对各个gene的对数似然求和即可，但是要考虑扰动的到底是谁）（输入lls、pert_site，输出ll_dataset） get_dataset_ll_from_per_ll
    3. 根据求出的p(D|z,G)，求p(z|D,G)，其中z代表的是latent。
    """

    def get_target_reulator(adata, adj_matrix, gene_index):
        all_data = adata.X
        ctrl_data = adata[adata.obs["pert"].isin(["CTRL","test1"])].X
        pert_data = adata[~adata.obs["pert"].isin(["CTRL","test1"])].X
        regulating_tfs = np.where(adj_matrix[:, gene_index] == 1)[0]

        target = all_data[:, gene_index]
        regulator = all_data[:, regulating_tfs]

        target_c = ctrl_data[:, gene_index]
        regulator_c = ctrl_data[:, regulating_tfs]

        target_p = pert_data[:, gene_index]
        regulator_p = pert_data[:, regulating_tfs]
        return (target, regulator), (target_p, regulator_p), (target_c, regulator_c)

    def get_dataset_ll_from_per_ll(arr, row_index,regulator_list = None):
        """
        arr: np.ndarray, shape (n, 3)
        row_index: int, 指定的行
        """
        if arr.shape[1] != 3:
            raise ValueError("输入数组必须是 n x 3 的形状")
        if regulator_list is None:
            # 创建布尔掩码：True 表示不是指定行
            mask = np.ones(arr.shape[0], dtype=bool)
            mask[row_index] = False

            # 其他行第0列的和
            sum_others = arr[mask, 0].sum()

            # 指定行的第1列 + 第2列
            sum_row = arr[row_index, 1] + arr[row_index, 2]
        else:
            TF_index = regulator_list[row_index]
            # 创建布尔掩码：True 表示不是指定行
            mask = np.ones(arr.shape[0], dtype=bool)
            mask[TF_index] = False

            # 其他行第0列的和
            sum_others = arr[mask, 0].sum()

            # 指定行的第1列 + 第2列
            sum_row = arr[TF_index, 1] + arr[TF_index, 2]
        return sum_others + sum_row

    sample_num, gene_num = adata.X.shape
    pert_num = adj_matrix.shape[0]  # ?? 有多少可能的靶点呢？想清楚。

    if target_index is None:
        lls = np.zeros([gene_num, 3])
        for i in range(gene_num):  # 遍历所有的gene作为target
            (target, regulator), (target_p, regulator_p), (target_c, regulator_c) = get_target_reulator(adata, adj_matrix,
                                                                                                        i)

            regulator_c_, _, _, mask = filter_by_range(regulator_c, regulator_p)
            regulator = np.concatenate([regulator_c_, regulator_p])
            target_c = target_c[mask]  # 好像是因为这个线性的问题所导致的。根据干预后的范围来对原始数据再做一个删减。

            target = np.concatenate([target_c, target_p])
            # regulator_c_ = regulator_c
            _, ll_all = ols_linear_fit(regulator, target)
            _, ll_pert = ols_linear_fit(regulator_p, target_p)
            _, ll_unperturb = ols_linear_fit(regulator_c_, target_c)

            # lls[i] = np.array([ll_all/regulator.shape[0], ll_pert/regulator_p.shape[0], ll_unperturb/regulator_c_.shape[0]])
            lls[i] = np.array([ll_all, ll_pert, ll_unperturb])
    else:
        for i in target_index:
            (target, regulator), (target_p, regulator_p), (target_c, regulator_c) = get_target_reulator(adata, adj_matrix,
                                                                                                        i)

            regulator_c_, _, _, mask = filter_by_range(regulator_c, regulator_p)
            regulator = np.concatenate([regulator_c_, regulator_p])
            target_c = target_c[mask]

            target = np.concatenate([target_c, target_p])
            # regulator_c_ = regulator_c
            _, ll_all = ols_linear_fit(regulator, target)
            _, ll_pert = ols_linear_fit(regulator_p, target_p)
            _, ll_unperturb = ols_linear_fit(regulator_c_, target_c)

            # lls[i] = np.array([ll_all/regulator.shape[0], ll_pert/regulator_p.shape[0], ll_unperturb/regulator_c_.shape[0]])
            lls[i] = np.array([ll_all, ll_pert, ll_unperturb])

    log_ll_dataset = np.zeros(pert_num)  # 有多少个靶点就对应了多少个数据集
    for pert_site in range(pert_num):  # ？？
        log_ll_dataset[pert_site] = get_dataset_ll_from_per_ll(lls, pert_site,regulator_list)  # 每一种扰动的情况下，整个数据的对数似然

    def normalize_ll_zscore(ll):
        ll = np.array(ll, dtype=float)
        mu = ll.mean()
        sd = ll.std(ddof=0)
        sd = max(sd, 1e-12)
        return (ll - mu) / sd

    log_ll_dataset = normalize_ll_zscore(log_ll_dataset)

    # posterior = ll_dataset[latent] / ll_dataset.sum()
    # return posterior
    # 下面做log-sum-exp归一化，获得概率
    def logsumexp(a):
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    tau = tau  # 试试10、50、100、500等
    log_ll_dataset_scaled = log_ll_dataset / tau
    log_norm = logsumexp(log_ll_dataset_scaled)
    posterior = np.exp(log_ll_dataset_scaled - log_norm)
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))
    norm_entropy = entropy / np.log(len(posterior))

    # 返回你需要的latent的后验概率
    return posterior, lls, norm_entropy

def filter_by_range(array, array1):
    """
    根据 array1 每个特征的取值范围，
    对 array 样本进行区间筛选。
    """

    # 保证是 numpy 数组
    array = np.asarray(array)
    array1 = np.asarray(array1)

    # 如果是一维向量，转为二维
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array1.ndim == 1:
        array1 = array1.reshape(-1, 1)

    # 检查维度
    if array.shape[1] != array1.shape[1]:
        raise ValueError(f"列数不匹配: array={array.shape[1]}, array1={array1.shape[1]}")

    # 计算 array1 每列的最小值和最大值
    min_vals = np.min(array1, axis=0)
    max_vals = np.max(array1, axis=0)

    # 对 array 的每一行进行筛选
    mask = np.all((array >= min_vals) & (array <= max_vals), axis=1)
    array_filtered = array[mask]

    # 拼接结果
    array_concat = np.vstack([array_filtered, array1])

    return array_filtered, array1, array_concat,mask

def ols_linear_fit(x, y,parms = None):
    y = as_dense(y).reshape(-1)
    if x is not None:
        x = as_dense(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    if x is None:
        X = np.ones(len(y)).reshape(-1,1)
    else:
        X = np.column_stack([np.ones(len(x)), x])
    if parms is None:
        # 设计矩阵 (包含截距)
        lambda_reg = 1e-3
        I = np.eye(X.shape[1])
        # OLS估计
        beta = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y  # shape (m+1,)

        # 预测值
        y_pred = X @ beta  # (n,)

        # 残差和噪声估计
        residuals = y - y_pred
        sigma_hat = np.sqrt(np.sum(residuals ** 2) / len(y))  # MLE估计
        sigma_hat = max(sigma_hat, 1e-8)
    else:
        beta = parms[:-1]
        sigma_hat = parms[-1]
        y_pred = X @ beta
    log_likelihood = np.sum(norm.logpdf(y, loc=y_pred, scale=sigma_hat))
    params = np.concatenate([beta, np.array([sigma_hat])])
    return params, log_likelihood

# weight的生成过程可能也是需要修改的：是的！
# weight的功能：每一个样本属于某一种干预类别的概率。
# 修改的逻辑：对于Train的部分，完全是不需要单独增加一个weight的，但是对于test的部分，就还是需要再增加一个weight。
def build_responsibility_matrix(data_dict, class_list, test_pert_site, pt_score=None, control_names=None):
    """
    根据字典和类别列表，构造 N x K 的 responsibility 矩阵 (0/1 one-hot)
    N：根据dict的values所计算出来的所有样本的数量
    #K：TF的数量+1. 因为当前的靶点基本上都是TF，然后还有CTRL的情况，就直接放到第一列了，第一列全部为1。其余只要是对照组的，第一列都是0
    K: pert的数量+1. 按照data_dict.keys()的顺序来做。
    
    Args:
        data_dict: dict, key 是类别名，value 是一个 list (长度表示样本数)。作用主要是帮助我们确定有多少个样本，以及CTRL的样本对应的是那些。
        class_list: list[str], 当前所有的TF。pt_score的列就是按照这个顺序来呈现的。
        pt_score: 除了CTRL的每一套数据集（行）的靶点相似度。

    Returns:
        resp: ndarray, shape (N, K).每一行是一个样本，每一列是一种TF（第一列是CTRL）
        行按照的顺序是data_dict，比如说第一个key有2000个样本，第二个key有3000个样本。。。
        列按照的顺序是CTRL + pert list
        其中的元素主要遵照pt_score，而pt_score的行的顺序是严格按照valid_perts来的，所以先在pt_score中找到这个key所对应的行的index，然后再把resp对应的行用pt_score[index]填充了，代表这个数据被扰动的靶点的概率。
    """
    N = sum(len(v) for v in data_dict.values())

    control_names = set(control_names or ["CTRL"])
    all_pert_sites = sorted((set(data_dict.keys()) | set(class_list)) - set(test_pert_site) - control_names) # weight所呈现的顺序（第0列是CTRL）
    #all_pert_sites = list((set(data_dict.keys()) | set(class_list)) - set(test_pert_site) - set(["CTRL"]))
    K = len(all_pert_sites) + 1 
    resp = np.zeros((N, K))

    

    # 先把TF的顺序映射到all_pert_sites上。
    target_indices = [all_pert_sites.index(cls)+1 for cls in class_list]

    
    if pt_score is not None:
        assert pt_score.shape[0] == 1
        row_start = 0
        pert_sites = data_dict.keys()

        for pert_site in pert_sites:
            n = len(data_dict[pert_site])
            if pert_site in control_names:
                col = 0
                resp[row_start:row_start + n, col] = 1
            elif pert_site in all_pert_sites: # 如果已经被干预了，那么就在对应位置（all_pert_sites）设置为1
                pt_score_row = all_pert_sites.index(pert_site) +1  #??
                resp[row_start:row_start + n, pt_score_row] = 1.0
            elif pert_site in test_pert_site:
                for i, tgt in enumerate(target_indices):
                    resp[row_start:row_start + n, tgt] = (pt_score)[0][i]
            row_start += n
    else:
        resp[:,0] = 1.0
    return resp

def batch_exec(f, args_batch, w):
    results = []
    for args in args_batch:
        try:
            ans = f(args)
            results.append(ans)
        except Exception:
            results.append(None)
        w.send(1)  # 发送更新信号
    return results


# 进度可视化的多进程执行
def multi_process_exec(f, args_mat, pool_size, desc=None):
    if len(args_mat) == 0:
        return []
    batch_size = max(1, int(len(args_mat) / 4 / pool_size))  # 设定批量大小
    results = []
    args_batches = [args_mat[i * batch_size:(i + 1) * batch_size] for i in
                    range((batch_size - 1 + len(args_mat)) // batch_size)]

    with tqdm(total=len(args_mat), desc=desc) as pbar:
        with Pool(processes=pool_size) as pool:
            r, w = Pipe(duplex=False)
            pool_rets = []

            for args_batch in args_batches:
                pool_rets.append(pool.apply_async(batch_exec, (f, args_batch, w)))

            cnt = 0
            while cnt < len(args_mat):
                try:
                    r.recv()  # 接收进度更新信号
                    pbar.update(1)
                    cnt += 1
                except EOFError:
                    break

            for ret in pool_rets:
                results.extend(ret.get())

    return results

def get_target_index(M: np.ndarray, n: int, one_based: bool = False, sum_topn: bool = True):
    """
    M: 2D numpy array
    返回 (rows, cols) 或 (rows, cols, S)
    
    逻辑说明：
    1. 找出全局最大的前 n 个元素（Top N）。
    2. 在这 n 个元素中，如果同一行出现了多次，只保留该行中值最大的那个（即第一次出现的那个，因为是降序排列）。
    3. 结果的数量可能小于 n。
    """
    if M.ndim != 2:
        raise ValueError("M must be a 2D array.")
    
    # 1. 获取全局 Top N 的索引
    # 使用 argpartition 找到前 n 大的索引（无序），或者直接 argsort（有序）
    # 为了逻辑清晰，先拉平并排序
    flat_indices = np.argsort(-M.ravel()) # 降序排列的全局索引
    
    # 只取前 n 个
    top_n_indices = flat_indices[:n]
    
    # 还原为二维坐标 (row, col)
    # 注意：这里的顺序是按值从大到小排列的
    top_rows, top_cols = np.unravel_index(top_n_indices, M.shape)
    
    selected_rows = []
    selected_cols = []
    seen_rows = set()
    
    # 2. 在这有限的 n 个候选中进行去重
    for r, c in zip(top_rows, top_cols):
        # 如果这一行在这个 Top N 集合里第一次出现，它肯定也是该行在这个集合里最大的
        # (因为我们是按值降序遍历的)
        if r not in seen_rows:
            seen_rows.add(r)
            selected_rows.append(r)
            selected_cols.append(c)
        # 如果 r 已经在 seen_rows 里，说明 Top N 里有该行更大的值，当前的被“吞掉”了
        # 此时不做任何操作，直接跳过，也不去后面补新的数据。

    # 转换为 numpy 数组
    final_rows = np.array(selected_rows)
    final_cols = np.array(selected_cols)
    
    # 获取对应的值
    if len(final_rows) > 0:
        final_vals = M[final_rows, final_cols]
    else:
        final_vals = np.array([])

    # 3. 最终排序输出 (按值降序)
    if len(final_rows) > 0:
        order = np.lexsort((final_cols, final_rows, -final_vals))
        final_rows = final_rows[order]
        final_cols = final_cols[order]
        final_vals = final_vals[order]

    # 4. 处理 one_based 索引
    if one_based:
        final_rows = final_rows + 1
        final_cols = final_cols + 1

    # 5. 返回结果
    if sum_topn:
        return final_rows.tolist(), final_cols.tolist(), float(np.sum(final_vals))
    else:
        return final_rows.tolist(), final_cols.tolist()







# def get_target_index(M: np.ndarray, n: int, one_based: bool = False, sum_topn: bool = True):
#     """
#     M: 2D numpy array (Rows: Targets, Cols: Regulators)
#     功能：从矩阵中选取前 n 个最大的元素，但限制 **每一行（Target）最多只能被选中一次**。
#     即：选取 n 个不同的行，使得这些行中最大值的总和尽可能大。
    
#     返回 (rows, cols) 或 (rows, cols, S)
#     S 为前n大元素的和
#     """
#     if M.ndim != 2:
#         raise ValueError("M must be a 2D array.")
    
#     n_rows, n_cols = M.shape
    
#     # 如果请求的 n 大于行数，最多只能返回所有行各一次
#     n = min(n, n_rows) 
    
#     # 1. 找到每一行（每个Target）的最大值及其所在的列索引（Regulator）
#     # axis=1 表示沿着列的方向操作，得到每个行的结果
#     row_max_vals = M.max(axis=1)      # 形状: (n_rows,)
#     row_argmax_cols = M.argmax(axis=1) # 形状: (n_rows,)
    
#     # 2. 找出 row_max_vals 中最大的 n 个值的索引（即选出哪几行）
#     # 使用 argpartition 进行快速选择（不完全排序），然后取前 n 个
#     if n < n_rows:
#         # 找出前 n 大的索引（无序）
#         top_row_indices = np.argpartition(-row_max_vals, n-1)[:n]
#         # 获取这些索引对应的值
#         top_vals = row_max_vals[top_row_indices]
#         # 对这 n 个进行排序，以便输出有序结果（从大到小）
#         # argsort 默认从小到大，所以对负值排序，或者[::-1]
#         sort_order = np.argsort(-top_vals)
        
#         selected_rows = top_row_indices[sort_order]
#     else:
#         # 如果 n >= n_rows，直接对所有行按最大值排序
#         selected_rows = np.argsort(-row_max_vals)

#     # 3. 根据选中的行，获取对应的列索引和值
#     selected_cols = row_argmax_cols[selected_rows]
#     selected_vals = row_max_vals[selected_rows]
    
#     # 4. 处理 one_based 索引（如果需要）
#     if one_based:
#         selected_rows = selected_rows + 1
#         selected_cols = selected_cols + 1
        
#     # 5. 返回结果
#     if sum_topn:
#         return selected_rows.tolist(), selected_cols.tolist(), float(np.sum(selected_vals))
#     else:
#         return selected_rows.tolist(), selected_cols.tolist()


# def get_target_index(M: np.ndarray, n: int, one_based: bool = False, sum_topn: bool = True):
#     """
#     M: 2D numpy array
#     返回 (rows, cols) 或 (rows, cols, S)
#     S 为前n大元素的和
#     """
#     if M.ndim != 2:
#         raise ValueError("M must be a 2D array.")
#     flat = M.ravel()
#     n = min(n, flat.size)
#     idx = np.argpartition(-flat, n-1)[:n]
#     rows, cols = np.unravel_index(idx, M.shape)
#     vals = flat[idx]
#     order = np.lexsort((cols, rows, -vals))
#     rows = rows[order]
#     cols = cols[order]
#     vals = vals[order]
#     if one_based:
#         rows = rows + 1
#         cols = cols + 1
#     if sum_topn:
#         return rows.tolist(), cols.tolist(), float(np.sum(vals))
#     else:
#         return rows.tolist(), cols.tolist()
    

class ConvergenceChecker:
    def __init__(self, window_size=30, tol=1e-4, init_ll=0.0):
        """
        参数:
            window_size: 滑动窗口大小（几轮内波动小则认为收敛）
            tol: 收敛阈值，波动小于这个值则认为收敛
            init_ll: 初始对数似然值（默认0即可）
        """
        self.window_size = window_size
        self.tol = tol
        self.init_ll = init_ll

        self.delta_lls = []  # 保存每一轮增量
        self.ll_values = [init_ll]  # 重建后的LL曲线

    def update(self, delta_ll):
        """
        更新新的对数似然增量，并判断是否收敛
        """
        self.delta_lls.append(delta_ll)
        new_ll = self.ll_values[-1] + delta_ll
        self.ll_values.append(new_ll)

        # 只有当长度足够时才判断
        if len(self.ll_values) < self.window_size:
            return False  # 还不能判断收敛

        window = self.ll_values[-self.window_size:]
        fluctuation = max(window) - min(window)
        converged = fluctuation < self.tol

        return converged

    def get_history(self):
        """
        返回历史LL和deltaLL，可用于绘图或分析
        """
        return np.array(self.ll_values[1:]), np.array(self.delta_lls)


def top5_per_row(matrix, k=5):
    """
    对每一行保留前5大元素，其他位置设为0。
    """
    matrix = np.array(matrix)
    result = np.zeros_like(matrix)

    # 遍历每一行
    for i in range(matrix.shape[0]):
        row = matrix[i]
        # 获取前5大元素的索引
        top5_idx = np.argsort(row)[-k:]
        # 保留对应位置的值
        result[i, top5_idx] = row[top5_idx]

    return result




def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _append_rows_csv(csv_path, rows):
    if not rows:
        return
    _ensure_dir(csv_path)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    # 确定列名：用第一行的键
    fieldnames = list(rows[0].keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

class BufferedCSVLogger:
    def __init__(self, csv_path, flush_every=100):
        """
        csv_path: 输出文件
        flush_every: 每多少轮落盘一次
        """
        self.csv_path = csv_path
        self.flush_every = int(flush_every)
        self.buffer = []
        self._fieldnames = None

    def add(self, row_dict):
        # 首次统一列名（按第一次的键集合）
        if self._fieldnames is None:
            self._fieldnames = list(row_dict.keys())
        else:
            # 仅保留首次确定的列，避免列漂移（可按需放宽）
            row_dict = {k: row_dict.get(k, None) for k in self._fieldnames}
        self.buffer.append(row_dict)

    def maybe_flush(self, iter_idx):
        if (iter_idx + 1) % self.flush_every == 0:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        _append_rows_csv(self.csv_path, self.buffer)
        self.buffer = []

    def close(self):
        self.flush()

# 简单的三类包装：rank / ll / new_pt_score
def to_row_iter_value(iter_idx, key_prefix, value, extra=None):
    """
    将任意标量/数组展平为一行。key_prefix 用作列名前缀。
    """
    row = {"iter": int(iter_idx)}
    if np.isscalar(value):
        row[f"{key_prefix}"] = float(value)
    else:
        arr = np.asarray(value)
        flat = arr.reshape(-1)
        for i, v in enumerate(flat):
            row[f"{key_prefix}_{i}"] = float(v)
    if extra:
        row.update(extra)
    return row


def batch_exec(f, args_batch, w):
    results = []
    for args in args_batch:
        try:
            ans = f(args)
            results.append(ans)
        except Exception:
            results.append(None)
        w.send(1)  # 发送更新信号
    return results


def get_rank(arr, indices):
    arr = np.array(arr).flatten()  # 展平成一维数组
    ranks = []

    for index in indices:
        value = arr[index]
        rank = (arr > value).sum() + 1  # 从大到小排名
        ranks.append(rank)

    return ranks


def reorder_array(arr, order_dict):
    """
    按照字典 values（list 拼接起来的顺序）重新排列数组
    :param arr: numpy array 或 list
    :param order_dict: dict，values 是 list，拼接后给出新顺序索引
    :return: numpy array，重新排列后的结果
    """
    arr = np.array(arr)
    # 拼接所有的 values (保持原有顺序)
    order = []
    for v in order_dict.values():
        order.extend(v)
    # 根据拼接后的顺序重排
    return arr[order]

def downsample_by_mask(X, data, weight, k, random_state=0):
    """
    根据 weight 的规则做下采样，返回新的 X, data, weight。
    
    参数:
        X, data, weight : np.ndarray, shape=(n, m)
        k : int 或 None，下采样数量；如果 None，用候选行数 // 3
        random_state : int 或 None，随机种子
    """
    rng = np.random.default_rng(random_state)
    n, m = weight.shape

    # 确保每一行的和为 1
    assert np.allclose(weight.sum(axis=1), 1), "每一行的权重和必须为1"

    # 条件1: 第一列为 1
    cond1 = (weight[:, 0] == 1)
    # 条件2: 其他列为 0
    cond2 = np.all(weight[:, 1:] == 0, axis=1)
    # 候选行索引
    candidates = np.where(cond1 & cond2)[0]

    # 下采样数量
    if k != 0:
        k = max(1, len(candidates) // k)  # 至少取 1 行，避免空
    else:
        k = len(candidates)
    k = min(k, len(candidates))  # 不能超过候选数

    # 随机抽取
    chosen = rng.choice(candidates, size=k, replace=False)

    # 其他行：第一列为 0 的
    others = np.where(weight[:, 0] == 0)[0]

    # 拼接索引并去重
    selected_idx = np.unique(np.concatenate([chosen, others]))
    if X is None:
        return None, data[selected_idx], weight[selected_idx]
    else:
        return X[selected_idx], data[selected_idx], weight[selected_idx]


def m_step_groups_simple(X, y, responsibilities, groups, lambda_reg=1e-5, min_sigma=1e-8):
    """
    相比于ols_linear_fit，本函数多了权重responsibilities作为输入。为了进行MLE以估计出似然函数以及条件分布，本函数对扰动的群体和非扰动的群体做了加权最小二乘法
    输入：
    X：[n,m] 
    y: [n,]
    responsibilities: [n,K],K = len(tf_list)+1
    groups:[intervene_index,unintervene_index],如[[2],[0,1,3]]; 如果说都是非扰动的，那么就
    注意：我们讨论的是否是扰动的，是针对于当前的（i，j）pair讨论的。有可能i这个target，并不是TF，但是我们认为扰动的数据一定是扰动的TF，所以说不会存在扰动i的数据。
    输出：
    
    [[intervene_parameters],[unintervene_parameters]]
    [[intervene_ll],[unintervene_ll]]
    """
    
    
    n, m = X.shape
    p = m + 1
    X_design = np.column_stack([np.ones(n), X])
    reg_matrix = np.eye(p)
    reg_matrix[0, 0] = 0.0 

    group_params = []
    group_lls = []
   
    n_groups = len(groups)
    W_all = np.zeros((n, n_groups))
    for i, group in enumerate(groups):
        if len(group) > 0:
            W_all[:, i] = responsibilities[:, group].sum(axis=1)

    log_2pi = np.log(2 * np.pi)
    for g in range(n_groups): # 对于扰动组和非扰动组分开计算
        w_group = W_all[:, g] # 每一个样本所占的权重吧。然后在这种情况下去做加权最小二乘
        nonzero_count = np.count_nonzero(w_group) # 看看
        R_sum = w_group.sum()
        if R_sum <= 0: # 有可能并不存在对这个类型的扰动。
            # beta_g = np.zeros(p)
            # sigma_g = 1.0
            # raise ValueError(f"no samples for group{group}")
            # params, log_likelihood = ols_linear_fit(X, y)
            # return np.array(params),np.array(log_likelihood)
            continue
        else:
            # W = np.diag(w_group)
            # A = X_design.T @ W @ X_design + lambda_reg * np.eye(p)
            Xw = X_design * w_group[:, None]  # 按行加权
            A = X_design.T @ Xw + lambda_reg * reg_matrix
            b = X_design.T @ (w_group * y)
            beta_g = np.linalg.solve(A, b)
            y_pred = X_design @ beta_g
            resid = y - y_pred
            sigma2_g = np.sum(w_group * resid ** 2) / R_sum
            sigma_g = np.sqrt(max(sigma2_g, min_sigma ** 2))

        params = np.concatenate([beta_g, np.array([sigma_g])])
        group_params.append(params)
        
        # term1 = -0.5 * np.log(2 * np.pi)
        # term2 = -np.log(sigma_g)
        # term3 = -0.5 * (resid / sigma_g) ** 2
        # per_sample_ll = term1 + term2 + term3

        # log_likelihood = np.sum(w_group * per_sample_ll)
        log_likelihood = -0.5 * R_sum * (log_2pi + 2 * np.log(sigma_g) + 1)

        group_lls.append(log_likelihood/nonzero_count) # 对于样本的数量去计算出一个平均值

    return np.array(group_params), np.array(group_lls)

def get_score_improvement_EM_test(ll, ll0, use_intervene):
    if use_intervene == 0:
        new_ll = ll #/ data.shape[0]
        return new_ll - ll0, new_ll
    else:
        if ll[0] is None or np.isnan(ll[0]):
            new_ll = ll[1]
        else:
            new_ll =  (ll[0] + ll[1])/2
        return new_ll - ll0, new_ll


def get_score_improvement_EM(ll, ll0, use_intervene):
    assert len(ll0) == 2
    if ll[0] is None or np.isnan(ll[0]):
        #new_ll = ll[1]
        delta = ll[1] - ll0[1]
    else:
        #new_ll =  (ll[0] + ll[1])/2
        delta = (ll[0] - ll0[0])/2+ (ll[1] - ll0[1])/2
    return delta, ll

def split_number(N, k):
    """
    输入 N 和一个指定数字 k
    输出 [[k], [剩下的数字]]
    split_number(3, 0))  # [[0], [1, 2]] 第一个代表的是intervened的，其他的是没有被干预到的。，
    """
    if not (0 <= k < N):
        raise ValueError("k 必须在 [0, N-1] 之间")
    all_nums = list(range(N))
    rest = [x for x in all_nums if x != k]
    return [[k], rest]

def compute_percentile_matrix(matrix):
    zero_rows = np.all(matrix == 0, axis=1)

    # 2. 创建结果矩阵并初始化为-1
    percentile_matrix = np.full_like(matrix, -1.0, dtype=float)

    # 3. 创建统一的掩码，标记需要计算分位数的位置
    # 初始掩码：非全零行的所有位置
    valid_mask = np.zeros_like(matrix, dtype=bool)
    valid_mask[~zero_rows] = True

    # 如果是方阵，需要排除对角线
    if matrix.shape[0] == matrix.shape[1]:
        # 创建对角线掩码
        diag_mask = np.eye(matrix.shape[0], dtype=bool)
        # 从有效掩码中排除对角线位置
        valid_mask &= ~diag_mask
        # 在结果矩阵中显式设置非全零行的对角线为0
        percentile_matrix[diag_mask & ~zero_rows.reshape(-1, 1)] = -1

    # 4. 提取需要计算分位数的值
    values_to_rank = matrix[valid_mask]

    # 5. 计算分位数（如果存在有效值）
    if len(values_to_rank) > 0:
        # 计算最小排名分位数
        ranks = rankdata(values_to_rank, method='min')
        percentiles = ranks / len(ranks)

        # 6. 将分位数填充到结果矩阵
        percentile_matrix[valid_mask] = percentiles
    return percentile_matrix,valid_mask


def compute_prior_improvement(prior_matrix, epsilon=1e-7):
    """
    计算 prior_improvement 矩阵，同时规避 log(0) 的问题。

    参数：
    - prior_matrix: 输入的先验矩阵
    - epsilon: 裁剪的最小值，避免 log(0) 问题

    返回：
    - prior_improvement 矩阵
    """
    # 对 prior_matrix 进行裁剪，限制值在 [epsilon, 1-epsilon] 范围内
    clipped_prior_matrix = np.clip(prior_matrix, epsilon, 1 - epsilon)

    # 计算 prior_improvement
    prior_improvement = np.log(clipped_prior_matrix) - np.log(1 - clipped_prior_matrix)

    return prior_improvement


def compute_percentile_matrix_old_effect(matrix):
    if matrix.shape[0] == matrix.shape[1]:
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        non_diag_values = matrix[mask]
        non_diag_percentiles = rankdata(non_diag_values, method="min") / len(non_diag_values)
        percentile_matrix = np.zeros_like(matrix, dtype=float)
        percentile_matrix[mask] = non_diag_percentiles
    else:
        values = matrix.flatten()
        percentiles = rankdata(values, method="min") / len(values)
        percentile_matrix = percentiles.reshape(matrix.shape)
    return percentile_matrix


def normaliz(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean) / std if std != 0 else matrix - mean


def minmax_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)


def masked_standardize(matrix, invalid_val=-99999):
    # 确保数据为 float64 类型
    matrix = matrix.astype(np.float64)

    # 创建掩码
    mask = matrix != invalid_val

    # 提取有效值
    valid_values = matrix[mask]
    if valid_values.size == 0:
        raise ValueError("No valid values to standardize.")

    mean = np.mean(valid_values)
    std = np.std(valid_values)

    # 如果标准差为 0，说明所有有效值都相等
    if std == 0:
        # 方案1：返回全 0（推荐）
        standardized = matrix.copy()
        standardized[mask] = 0.0
        return standardized

        # 如果你想保留原始值而不是全 0，可以换成：
        # return matrix

    # 创建输出矩阵并标准化
    standardized = matrix.copy()
    standardized[mask] = (matrix[mask] - mean) / std

    return standardized



import numpy as np
from numba import njit

# 辅助函数：预计算权重矩阵，因为 Numba 不太好处理 list of lists
def prepare_weights(responsibilities, groups):
    n = responsibilities.shape[0]
    n_groups = len(groups)
    W_all = np.zeros((n, n_groups))
    for i, group in enumerate(groups):
        # group 是 list，这里在 Python 端处理好
        if len(group) > 0:
            # 假设 group 是索引列表
            for idx in group:
                W_all[:, i] += responsibilities[:, idx]
    return W_all

# @njit(fastmath=True,nogil=True, parallel=False)
# def m_step_numba_core(X_design, y, W_all, lambda_reg=1e-5, min_sigma=1e-8):
#     n, p = X_design.shape
#     n_groups = W_all.shape[1]
    
#     # 预分配结果数组
#     out_params = np.zeros((n_groups, p + 1))
#     out_lls = np.zeros(n_groups)
    
#     reg_matrix = np.eye(p) * lambda_reg
#     reg_matrix[0, 0] = 0.0
    
#     log_2pi = np.log(2 * np.pi)

#     for g in range(n_groups):
#         w_group = W_all[:, g]
#         R_sum = 0.0
#         nonzero_count = 0
        
#         # 手动计算 sum 和 nonzero，避免创建额外数组
#         for i in range(n):
#             val = w_group[i]
#             R_sum += val
#             if val > 0:
#                 nonzero_count += 1
        
#         if R_sum <= 1e-12:
#             continue

#         # 构建 A = X.T @ W @ X
#         # Numba 中手动写循环有时比矩阵乘法更便于控制内存，
#         # 但对于这种规模，直接用 numpy 语法在 numba 里也是很快的
#         # 技巧：Xw = X_design * w_group.reshape(-1, 1) 会产生临时数组
#         # 我们用更省内存的方式构建 A 和 b
        
#         A = np.zeros((p, p))
#         b = np.zeros(p)
        
#         # 这是一个 O(N * P^2) 的操作，但在 C 层面非常快
#         # 如果 N 很大，P 很小，这是最优解
#         for i in range(n):
#             weight = w_group[i]
#             if weight > 1e-12: # 稀疏性优化
#                 yi = y[i]
#                 row = X_design[i, :]
#                 # A += weight * outer(row, row)
#                 # b += weight * yi * row
#                 for j in range(p):
#                     weighted_row_j = weight * row[j]
#                     b[j] += weighted_row_j * yi
#                     for k in range(p):
#                         A[j, k] += weighted_row_j * row[k]
        
#         # 添加正则项
#         for j in range(p):
#             A[j, j] += reg_matrix[j, j] # 只有对角线有值

#         # Solve
#         beta_g = np.linalg.solve(A, b)
        
#         # 计算 Sigma 和 LL
#         resid_sq_sum = 0.0
#         for i in range(n):
#             if w_group[i] > 1e-12:
#                 # pred = dot(row, beta)
#                 pred = 0.0
#                 for j in range(p):
#                     pred += X_design[i, j] * beta_g[j]
#                 resid_sq_sum += w_group[i] * (y[i] - pred)**2
        
#         sigma2_g = resid_sq_sum / R_sum
#         sigma_g = np.sqrt(max(sigma2_g, min_sigma ** 2))
        
#         # 填入结果
#         out_params[g, :p] = beta_g
#         out_params[g, p] = sigma_g
        
#         # 简化版 LL 计算
#         weighted_ll_sum = -0.5 * R_sum * (log_2pi + 2 * np.log(sigma_g) + 1)
#         out_lls[g] = weighted_ll_sum / nonzero_count

#     return out_params, out_lls


@njit(fastmath=True, nogil=True, parallel=False)
def m_step_numba_core(X_design, y, W_all, lambda_reg=1e-5, min_sigma=1e-8):
    n, p = X_design.shape
    n_groups = W_all.shape[1]
    
    out_params = np.zeros((n_groups, p + 1))
    out_lls = np.zeros(n_groups)
    
    # 修改点：截距项也给极小正则化
    reg_matrix_diag = np.full(p, lambda_reg)
    reg_matrix_diag[0] = 1e-6  

    log_2pi = np.log(2 * np.pi)

    for g in range(n_groups):
        w_group = W_all[:, g]
        R_sum = 0.0
        nonzero_count = 0
        
        for i in range(n):
            val = w_group[i]
            R_sum += val
            if val > 0:
                nonzero_count += 1
        
        if R_sum <= 1e-12:
            continue

        A = np.zeros((p, p))
        b = np.zeros(p)
        
        for i in range(n):
            weight = w_group[i]
            if weight > 1e-12: 
                yi = y[i]
                row = X_design[i, :]
                for j in range(p):
                    weighted_row_j = weight * row[j]
                    b[j] += weighted_row_j * yi
                    for k in range(p):
                        A[j, k] += weighted_row_j * row[k]
        
        for j in range(p):
            A[j, j] += reg_matrix_diag[j]

        # 修改点：使用 lstsq 替代 solve
        # rcond=-1 让 numba 使用机器精度作为默认截断
        beta_g = np.linalg.lstsq(A, b, rcond=-1)[0]
        
        resid_sq_sum = 0.0
        for i in range(n):
            if w_group[i] > 1e-12:
                pred = 0.0
                for j in range(p):
                    pred += X_design[i, j] * beta_g[j]
                resid_sq_sum += w_group[i] * (y[i] - pred)**2
        
        sigma2_g = resid_sq_sum / R_sum
        sigma_g = np.sqrt(max(sigma2_g, min_sigma ** 2))
        
        out_params[g, :p] = beta_g
        out_params[g, p] = sigma_g
        
        weighted_ll_sum = -0.5 * R_sum * (log_2pi + 2 * np.log(sigma_g) + 1)
        
        if nonzero_count > 0:
            out_lls[g] = weighted_ll_sum / nonzero_count
        else:
            out_lls[g] = -np.inf

    return out_params, out_lls



@njit(fastmath=True, nogil=True, parallel=False)
def m_step_numba_core_sparse(X_design, y, W_all, lambda_reg=1e-3, min_sigma=1e-8):
    n, p = X_design.shape
    n_groups = W_all.shape[1]

    out_params = np.zeros((n_groups, p + 1))
    out_lls = np.zeros(n_groups)

    reg_matrix_diag = np.full(p, lambda_reg)
    reg_matrix_diag[0] = 1e-6

    log_2pi = np.log(2.0 * np.pi)

    for g in range(n_groups):
        w_group = W_all[:, g]

        nz_idx = np.where(w_group > 1e-8)[0]
        k = len(nz_idx)
        if k == 0:
            if n_groups == 1:
                raise ValueError("n_groups == 1 without weight")
            continue

        R_sum = 0.0
        for ii in range(k):
            R_sum += w_group[nz_idx[ii]]

        A = np.zeros((p, p))
        b = np.zeros(p)
        for ii in range(k):
            i = nz_idx[ii]
            weight = w_group[i]
            row = X_design[i]
            yi = y[i]
            for j in range(p):
                w_xj = weight * row[j]
                b[j] += w_xj * yi
                for k2 in range(p):
                    A[j, k2] += w_xj * row[k2]

        for j in range(p):
            A[j, j] += reg_matrix_diag[j]

        beta_g = np.linalg.lstsq(A, b, rcond=-1)[0]

        resid_sq_sum = 0.0
        for ii in range(k):
            i = nz_idx[ii]
            pred = X_design[i] @ beta_g
            resid_sq_sum += w_group[i] * (y[i] - pred) ** 2

        sigma2_g = resid_sq_sum / R_sum
        sigma_g = np.sqrt(max(sigma2_g, min_sigma * min_sigma))

        out_params[g, :p] = beta_g
        out_params[g, p] = sigma_g

        weighted_ll_sum = -0.5 * R_sum * (log_2pi + 2.0 * np.log(sigma_g) + 1.0)
        out_lls[g] = weighted_ll_sum / k

    return out_params, out_lls


@njit(fastmath=True, nogil=True, parallel=False)
def weighted_loglik_numba_core(
    X_design,
    y,
    W_all,
    params,
):
    n, p = X_design.shape
    n_groups = W_all.shape[1]

    out_lls = np.zeros(n_groups)

    log_2pi = np.log(2.0 * np.pi)

    for g in range(n_groups):
        beta_g = params[g, :p]
        sigma_g = params[g, p]

        if sigma_g <= 0.0:
            out_lls[g] = -np.inf
            continue

        inv_2sigma2 = 0.5 / (sigma_g * sigma_g)

        w_group = W_all[:, g]
        ll_sum = 0.0
        nonzero_count = 0

        for i in range(n):
            w = w_group[i]
            if w > 1e-12:
                # 预测值
                pred = 0.0
                for j in range(p):
                    pred += X_design[i, j] * beta_g[j]

                resid = y[i] - pred

                ll_sum += w * (
                    -0.5 * log_2pi
                    - np.log(sigma_g)
                    - resid * resid * inv_2sigma2
                )

                nonzero_count += 1

        if nonzero_count > 0:
            out_lls[g] = ll_sum / nonzero_count
        else:
            out_lls[g] = -np.inf

    return out_lls

def weighted_loglik_numpy(X_design, y, W_all, params):
    n, p = X_design.shape
    n_groups = W_all.shape[1]

    log_2pi = np.log(2.0 * np.pi)
    out_lls = np.zeros(n_groups)

    for g in range(n_groups):
        beta = params[g, :p]
        sigma = params[g, p]

        ll_sum = 0.0
        nonzero = 0

        for i in range(n):
            w = W_all[i, g]
            if w > 1e-12:
                pred = X_design[i] @ beta
                resid = y[i] - pred

                ll = (
                    -0.5 * log_2pi
                    - np.log(sigma)
                    - resid * resid / (2.0 * sigma * sigma)
                )
                ll_sum += w * ll
                nonzero += 1

        out_lls[g] = ll_sum / nonzero if nonzero > 0 else -np.inf

    return out_lls


def filter_data_by_y(X_design, y, W_all, threshold=1e-12):
    """
    筛选掉 y 中为 0 (或绝对值小于阈值) 的样本，并同步筛选 X_design 和 W_all。
    """
    mask = np.abs(y) > threshold
    return X_design[mask], y[mask], W_all[mask]


def log_likelihood_gaussian(params, X, Y, n):
    a = params[:n]
    b = params[n]
    c = params[n + 1:n + 1 + n]
    d = params[n + 1 + n]
    X = X.reshape(X.shape[0], -1)
    mean = X.dot(a) + b
    log_variance = X.dot(c) + d
    variance = np.maximum(np.exp(log_variance), 1e-8)
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + (Y - mean) ** 2 / variance)
    return -log_likelihood


def gradient(params, X, Y, n):
    a = params[:n]
    b = params[n]
    c = params[n + 1:2 * n + 1]
    d = params[2 * n + 1]
    mean = X @ a + b
    log_variance = X @ c + d
    variance = np.maximum(np.exp(log_variance), 1e-8)
    residual = Y - mean
    inv_variance = 1.0 / variance
    grad_a = -X.T @ (residual * inv_variance)
    grad_b = -np.sum(residual * inv_variance)
    temp = -0.5 * (residual ** 2 * inv_variance - 1)
    grad_c = X.T @ temp
    grad_d = np.sum(temp)
    return np.concatenate([grad_a, [grad_b], grad_c, [grad_d]])


def get_initial_params(X, Y, n):
    a_init = np.linalg.lstsq(X, Y, rcond=None)[0]
    b_init = Y.mean() - X.mean(axis=0).dot(a_init)
    d_init = np.log(max(Y.var(), 1e-8))
    c_init = np.zeros(X.shape[1])
    return np.concatenate([a_init, [b_init], c_init, [d_init]])


def _m_step_old_heteroscedastic_average(X, y, responsibilities, groups):
    if X is None:
        raise ValueError("old_heteroscedastic mode requires at least one regulator.")
    X = np.asarray(X).reshape(len(y), -1)
    y = np.asarray(y)
    W_all = prepare_weights(responsibilities, groups)
    n = X.shape[1]
    params = []
    lls = []
    for group_index in range(W_all.shape[1]):
        selected = W_all[:, group_index] > 1e-8
        count = int(np.count_nonzero(selected))
        if count == 0:
            params.append(None)
            lls.append(np.nan)
            continue
        xg = X[selected]
        yg = y[selected]
        initial_params = get_initial_params(xg, yg, n)
        result = minimize(log_likelihood_gaussian, initial_params, args=(xg, yg, n), method="L-BFGS-B", jac=gradient)
        if not result.success:
            raise ValueError(f"Optimization group {group_index} failed: {result.message}")
        params.append(result.x)
        lls.append(-result.fun / count)
    return params, np.array(lls, dtype=float)


# === 封装调用 ===
def m_step_fast(X, y, responsibilities, groups, model="ols", lambda_reg=1e-3, min_sigma=1e-8):
    if model == "old_heteroscedastic":
        return _m_step_old_heteroscedastic_average(X, y, responsibilities, groups)
    if model != "ols":
        raise ValueError(f"Unknown m_step_fast model: {model}")
    # 1. 准备数据
    if X is None:
        X_design = np.ones((y.shape[0], 1))
    else:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    # 2. 转换权重
    W_all = prepare_weights(responsibilities, groups)
    # 3. Numba 计算
    # X_design_f, y_f, W_all_f = filter_data_by_y(X_design, y, W_all)

    return m_step_numba_core_sparse(X_design, y, W_all, lambda_reg, min_sigma)


# 此时我们只需要根据已经有的参数然后去计算似然函数
def unintervene_ll(X, y, responsibilities, groups,params):
    if X is None:
        X_design = np.ones((y.shape[0], 1))
    else:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    # 2. 转换权重
    W_all = prepare_weights(responsibilities, groups)   
    return weighted_loglik_numba_core(X_design, y, W_all, params)




def extract_weight_matrix(parameters, adj_matrix, use_intervene = 0):
    """
    输入：parameters、adj、use_intervene（这个并不代表我们的整个方法是否使用有扰动的信息，而是parameters的形式parameters.append([parameter_intervene_, parameter_unitervene_])就是1，否则就是0）.
    输出：变量之间的关系
    """
    num_genes = len(parameters)
    num_tfs = adj_matrix.shape[0]

    W = np.zeros((num_genes, num_tfs))  # 最终输出矩阵
    list_ = adj_matrix.sum(axis=0)
    B = np.zeros((num_genes, 1))

    for i in range(num_genes):
        if list_[i] != 0:
            n = int(list_[i])
            if use_intervene == 1:
              param_vec = parameters[i][1]  # 拿第一个 array，长度是 2n+2
            else:
              param_vec = parameters[i] # 
            coeffs = param_vec[1:-1]  # 取前 n 个调控系数
            b = param_vec[0]
            regulators = np.where(adj_matrix[:, i] == 1)[0]  # 找调控 gene i 的 TF 的索引
            if len(regulators) != n:
                raise ValueError(
                    f"Mismatch in number of regulators at gene {i}: got {n}, but adj has {len(regulators)}.")
            W[i, regulators] = coeffs  # 把对应的位置填上系数
            B[i] = b
        else:
            W[i, :] = 0  # 把对应的位置填上系数
            B[i] = 0

    return W, B


# 再检查一下
# 这个版本只是想要测试一下。我不再关注pert部分拟合的怎么样了，我只是想看后面的那个部分。
# 如果说这个版本是ok的，那我再调一下这个部分的参数
def z_posterior_Estep2(all_data,score_arrays, adj_matrix, tau,regulator_list):
    """
    all_data: adata_test_pert中的数据，但是按照拟时序的顺序呈现。adata[pert_dict[test_pert_site[0]],:].X
    计算隐变量的后验分布（仅限于adata中有一种扰动数据的场景。如果需要拓展，需要再计算一下） #？？
    p(z|D,G) = p(D|z,G)p(z|G)/(sum_z p(D|z,G)p(z|G))
    1. 计算每一个targt在all data、ctrl data、pert data下所对应的对数似然（输入data、adj，输出lls ）get_target_reulator、ols_linear_fit
    2. 计算在每一种扰动的情况下，整个数据集的 平均 对数似然（对各个gene的对数似然求和即可，但是要考虑扰动的到底是谁）（输入lls、pert_site，输出ll_dataset） get_dataset_ll_from_per_ll
    3. 根据求出的p(D|z,G)，求p(z|D,G)，其中z代表的是latent。
    """

    def get_target_reulator(all_data, adj_matrix, gene_index,regulator_list):
        if len(np.where(adj_matrix[:, gene_index] == 1)[0]) >0:
            regulating_tfs = regulator_list[np.where(adj_matrix[:, gene_index] == 1)[0]]
            regulator = all_data[:-1, regulating_tfs]
        else:
            regulator = None
        target = all_data[1:, gene_index]  
        return target, regulator

    def get_dataset_ll_from_per_ll(arr, row_index, regulator_list=None):
        if arr.shape[1] != 2:
            raise ValueError("输入数组必须是 n x 2 的形状")
        
        # 确定实际被干预的基因在 lls 矩阵中的行索引
        if regulator_list is None:
            TF_index = row_index
        else:
            TF_index = regulator_list[row_index]

        # 逻辑：总似然 = (所有基因在未干预下的似然和) - (该TF在未干预下的似然) + (该TF在干预下的似然)
        # 这样写比创建 mask 更快
        total_unpert_ll = arr[:, 1].sum()
        ll_dataset = total_unpert_ll - arr[TF_index, 1] + arr[TF_index, 0]
        
        return ll_dataset

    # sample_num, gene_num = adata.X.shape
    # all_data = adata.X
    sample_num, gene_num = all_data.shape

    # if target_index is None:
    lls = np.zeros([gene_num, 2])
    for i in range(gene_num):  # 遍历所有的gene作为target
        target, regulator = get_target_reulator(all_data, adj_matrix, i,regulator_list)
        if score_arrays["params_all"][i] is None:
            raise ValueError(f"no perturbation in gene {i}")
        param0 = score_arrays["params_all"][i][0]
        if isinstance(param0, np.ndarray):   #np.isnan(score_arrays["params_all"][i][0]):
            parms_pert = score_arrays["params_all"][i][0] # 表示第i个target在收到干预的情况下
            #if parms_pert.sum() != 0:
            _, ll_pert = ols_linear_fit(regulator, target,parms=parms_pert)
            #else: 
            #    ll_pert = -np.inf # 此时应该不太会有可能是这个扰动的结果了            
        else:
            if i in regulator_list:
                ll_pert = -np.inf # 此时应该不太会有可能是这个扰动的结果了      
                #raise ValueError(f"no perturbation in gene {i}")
            else: # 单纯就是不存在
                ll_pert = np.nan             

        
        parms_unpert = score_arrays["params_all"][i][1]
        _, ll_unperturb = ols_linear_fit(regulator, target,parms=parms_unpert)

        # lls[i] = np.array([ll_all/regulator.shape[0], ll_pert/regulator_p.shape[0], ll_unperturb/regulator_c_.shape[0]])
        lls[i] = np.array([ll_pert, ll_unperturb])

    # pert_num = adj_matrix.shape[0]  # ?? 有多少可能的靶点呢？想清楚。
    # log_ll_dataset = np.zeros(pert_num)  # 有多少个靶点就对应了多少个数据集
    # for pert_site in range(pert_num):  # ？？
    #     log_ll_dataset[pert_site] = -lls[regulator_list[pert_site],1]  # 每一种扰动的情况下，整个数据的对数似然

    log_ll_dataset = - lls[regulator_list][:,1]
    # 下面做log-sum-exp归一化，获得概率
    def logsumexp(a):
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    tau = tau  # 试试10、50、100、500等
    log_ll_dataset_scaled = log_ll_dataset / tau
    log_norm = logsumexp(log_ll_dataset_scaled)
    posterior = np.exp(log_ll_dataset_scaled - log_norm)
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))
    norm_entropy = entropy / np.log(len(posterior))

    # 返回你需要的latent的后验概率
    return posterior, lls, norm_entropy

def z_posterior_Estep(all_data,score_arrays, adj_matrix, tau,regulator_list):
    """
    all_data: adata_test_pert中的数据，但是按照拟时序的顺序呈现。adata[pert_dict[test_pert_site[0]],:].X
    计算隐变量的后验分布（仅限于adata中有一种扰动数据的场景。如果需要拓展，需要再计算一下） #？？
    p(z|D,G) = p(D|z,G)p(z|G)/(sum_z p(D|z,G)p(z|G))
    1. 计算每一个targt在all data、ctrl data、pert data下所对应的对数似然（输入data、adj，输出lls ）get_target_reulator、ols_linear_fit
    2. 计算在每一种扰动的情况下，整个数据集的 平均 对数似然（对各个gene的对数似然求和即可，但是要考虑扰动的到底是谁）（输入lls、pert_site，输出ll_dataset） get_dataset_ll_from_per_ll
    3. 根据求出的p(D|z,G)，求p(z|D,G)，其中z代表的是latent。
    """

    def get_target_reulator(all_data, adj_matrix, gene_index,regulator_list):
        if len(np.where(adj_matrix[:, gene_index] == 1)[0]) >0:
            regulating_tfs = regulator_list[np.where(adj_matrix[:, gene_index] == 1)[0]]
            regulator = all_data[:-1, regulating_tfs]
        else:
            regulator = None
        target = all_data[1:, gene_index]  
        return target, regulator

    def get_dataset_ll_from_per_ll(arr, row_index, regulator_list=None):
        if arr.shape[1] != 2:
            raise ValueError("输入数组必须是 n x 2 的形状")
        
        # 确定实际被干预的基因在 lls 矩阵中的行索引
        if regulator_list is None:
            TF_index = row_index
        else:
            TF_index = regulator_list[row_index]

        # 逻辑：总似然 = (所有基因在未干预下的似然和) - (该TF在未干预下的似然) + (该TF在干预下的似然)
        # 这样写比创建 mask 更快
        total_unpert_ll = arr[:, 1].sum()
        ll_dataset = total_unpert_ll - arr[TF_index, 1] + arr[TF_index, 0]
        
        return ll_dataset

    # sample_num, gene_num = adata.X.shape
    # all_data = adata.X
    sample_num, gene_num = all_data.shape

    # if target_index is None:
    lls = np.zeros([gene_num, 2])
    for i in range(gene_num):  # 遍历所有的gene作为target
        target, regulator = get_target_reulator(all_data, adj_matrix, i,regulator_list)
        if score_arrays["params_all"][i] is None:
            raise ValueError(f"no perturbation in gene {i}")
        param0 = score_arrays["params_all"][i][0]
        if isinstance(param0, np.ndarray):   #np.isnan(score_arrays["params_all"][i][0]):
            parms_pert = score_arrays["params_all"][i][0] # 表示第i个target在收到干预的情况下
            #if parms_pert.sum() != 0:
            _, ll_pert = ols_linear_fit(regulator, target,parms=parms_pert)
            #else: 
            #    ll_pert = -np.inf # 此时应该不太会有可能是这个扰动的结果了            
        else:
            if i in regulator_list:
                ll_pert = -np.inf # 此时应该不太会有可能是这个扰动的结果了      
                #raise ValueError(f"no perturbation in gene {i}")
            else: # 单纯就是不存在
                ll_pert = np.nan             

        
        parms_unpert = score_arrays["params_all"][i][1]
        _, ll_unperturb = ols_linear_fit(regulator, target,parms=parms_unpert)

        # lls[i] = np.array([ll_all/regulator.shape[0], ll_pert/regulator_p.shape[0], ll_unperturb/regulator_c_.shape[0]])
        lls[i] = np.array([ll_pert, ll_unperturb])

    pert_num = adj_matrix.shape[0]  # ?? 有多少可能的靶点呢？想清楚。
    log_ll_dataset = np.zeros(pert_num)  # 有多少个靶点就对应了多少个数据集
    for pert_site in range(pert_num):  # ？？
        log_ll_dataset[pert_site] = get_dataset_ll_from_per_ll(lls, pert_site,regulator_list)  # 每一种扰动的情况下，整个数据的对数似然

    # 下面做log-sum-exp归一化，获得概率
    def logsumexp(a):
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    tau = tau  # 试试10、50、100、500等
    log_ll_dataset_scaled = log_ll_dataset / tau
    log_norm = logsumexp(log_ll_dataset_scaled)
    posterior = np.exp(log_ll_dataset_scaled - log_norm)
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))
    norm_entropy = entropy / np.log(len(posterior))

    # 返回你需要的latent的后验概率
    return posterior, lls, norm_entropy



def z_posterior_Estep_with_pi(all_data,score_arrays, adj_matrix, tau,regulator_list,pi,type = "normal"):
    """
    all_data: adata_test_pert中的数据，但是按照拟时序的顺序呈现。adata[pert_dict[test_pert_site[0]],:].X
    计算隐变量的后验分布（仅限于adata中有一种扰动数据的场景。如果需要拓展，需要再计算一下） #？？
    p(z|D,G) = p(D|z,G)p(z|G)/(sum_z p(D|z,G)p(z|G))
    p(z|G)代表的就是此时的pi。将随着进行的过程不断的迭代，从而。
    1. 计算每一个targt在all data、ctrl data、pert data下所对应的对数似然（输入data、adj，输出lls ）get_target_reulator、ols_linear_fit
    2. 计算在每一种扰动的情况下，整个数据集的 平均 对数似然（对各个gene的对数似然求和即可，但是要考虑扰动的到底是谁）（输入lls、pert_site，输出ll_dataset） get_dataset_ll_from_per_ll
    3. 根据求出的p(D|z,G)，求p(z|D,G)，其中z代表的是latent。
    """

    def get_target_reulator(all_data, adj_matrix, gene_index,regulator_list):
        if len(np.where(adj_matrix[:, gene_index] == 1)[0]) >0:
            regulating_tfs = regulator_list[np.where(adj_matrix[:, gene_index] == 1)[0]]
            regulator = all_data[:-1, regulating_tfs]
        else:
            regulator = None
        target = all_data[1:, gene_index]  
        return target, regulator

    def get_dataset_ll_from_per_ll(arr, row_index, regulator_list=None):
        if arr.shape[1] != 2:
            raise ValueError("输入数组必须是 n x 2 的形状")
        
        # 确定实际被干预的基因在 lls 矩阵中的行索引
        if regulator_list is None:
            TF_index = row_index
        else:
            TF_index = regulator_list[row_index]

        # 逻辑：总似然 = (所有基因在未干预下的似然和) - (该TF在未干预下的似然) + (该TF在干预下的似然)
        # 这样写比创建 mask 更快
        total_unpert_ll = arr[:, 1].sum()
        ll_dataset = total_unpert_ll - arr[TF_index, 1] + arr[TF_index, 0]
        
        return ll_dataset

    # sample_num, gene_num = adata.X.shape
    # all_data = adata.X
    sample_num, gene_num = all_data.shape

    # if target_index is None:
    lls = np.zeros([gene_num, 2])
    for i in range(gene_num):  # 遍历所有的gene作为target
        target, regulator = get_target_reulator(all_data, adj_matrix, i,regulator_list)
        # 这里我希望能够删除掉一部分target为0的部分，从而使得正态分布能够更好的去贴近数据本身
        # mask = np.abs(target) > 1e-12
        # target = target[mask]
        # if regulator is not None:
        #     regulator = regulator[mask]
        # 删除结束！
        if score_arrays["params_all"][i] is None:
            raise ValueError(f"no perturbation in gene {i}")
        param0 = score_arrays["params_all"][i][0]
        if isinstance(param0, np.ndarray):   #np.isnan(score_arrays["params_all"][i][0]):
            parms_pert = score_arrays["params_all"][i][0] # 表示第i个target在收到干预的情况下
            #if parms_pert.sum() != 0:
            if type == "normal":
                _, ll_pert = ols_linear_fit(regulator, target,parms=parms_pert)
            # ll_pert = ll_pert/(target.shape[0])
            elif type == "nb":
                 _, ll_pert = nb_linear_fit(regulator, target,parms=parms_pert)


            #else: 
            #    ll_pert = -np.inf # 此时应该不太会有可能是这个扰动的结果了            
        else:
            if i in regulator_list:
                ll_pert = -np.inf # 此时应该不太会有可能是这个扰动的结果了      
                #raise ValueError(f"no perturbation in gene {i}")
            else: # 单纯就是不存在
                ll_pert = np.nan             

        
        parms_unpert = score_arrays["params_all"][i][1]
        if type == "normal":
            _, ll_unperturb = ols_linear_fit(regulator, target,parms=parms_unpert)
        elif type == "nb":
            _, ll_unperturb = nb_linear_fit(regulator, target,parms=parms_unpert)
        # ll_unperturb = ll_unperturb/(target.shape[0])
        # lls[i] = np.array([ll_all/regulator.shape[0], ll_pert/regulator_p.shape[0], ll_unperturb/regulator_c_.shape[0]])
        lls[i] = np.array([ll_pert, ll_unperturb])

    pert_num = adj_matrix.shape[0]  # ?? 有多少可能的靶点呢？想清楚。
    log_ll_dataset = np.zeros(pert_num)  # 有多少个靶点就对应了多少个数据集
    assert pi.shape[1] == pert_num # 表示的是大家被干预的一个概率之类的东西。
    pi1 = pi.flatten()
    for pert_site in range(pert_num):  # ？？
        #log_ll_dataset[pert_site] = pi1[pert_site] * get_dataset_ll_from_per_ll(lls, pert_site,regulator_list)  # 每一种扰动的情况下，整个数据的对数似然
        log_prior = np.log(pi1[pert_site] + 1e-12)
        log_ll_dataset[pert_site] = log_prior + (lls[regulator_list][pert_site,0]- lls[regulator_list][pert_site,1])

    # 下面做log-sum-exp归一化，获得概率
    def logsumexp(a):
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))

    tau = tau  # 试试10、50、100、500等
    log_ll_dataset_scaled = log_ll_dataset/tau   #tau 
    log_norm = logsumexp(log_ll_dataset_scaled)
    posterior = np.exp(log_ll_dataset_scaled - log_norm)
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))
    norm_entropy = entropy / np.log(len(posterior))

    # 返回你需要的latent的后验概率
    return posterior, lls, norm_entropy


# 由于概率的更新会导致原本的似然函数发生变化，因此对所有的regulator作为target的部分进行更新
def update_ll0(score_arrays,weight_new,data,adj_matrix,regulator_list,gene_name,all_pert_sites,all_regulator_index,all_target_index):
    ll0 = score_arrays["ll0"]
    params_all = score_arrays["params_all"]

    for i in tqdm(regulator_list):
        unpert_params = params_all[i][1]
        unpert_params = unpert_params[None, :]
        variable = data[:,i]
        target_name = gene_name[i]
        target2regulator_index = all_pert_sites.index(target_name) if target_name in all_pert_sites else None
        if target2regulator_index is not None:
            groups =  split_number(len(all_pert_sites) + 1, target2regulator_index + 1)
        else:
            raise ValueError("pert?")
            #groups = [list(range(len(all_pert_sites) + 1))]
        
        if len(np.where(adj_matrix[:, i] == 1)[0]) != 0:
            old_regulator = list(np.where(adj_matrix[:, i] == 1)[0])
            regulater = data[:, old_regulator].reshape(data.shape[0], -1)      
            regulater = reorder_array(regulater, all_regulator_index)   
        else:
            regulater = None
        
        variable = reorder_array(variable, all_target_index) # 来使得
        # 针对干预的部分，重新计算系数和似然
        params, log_likelihood = m_step_fast(regulater, variable, weight_new,[groups[0]])
        # 针对非干预的部分，用之前的系数来算此时的似然
        unintervene_liklelihood = unintervene_ll(regulater, variable, weight_new,[groups[1]],unpert_params)
        # if len(params) == 1:
        #     params_all[i] = (np.nan, params[0])
        #     ll0[i,1] = log_likelihood[0]
        # elif len(params) == 2:
        params_all[i] = (params[0],params_all[i][1]) # 只有干预的才会发生变化，否则不变
        ll0[i,0] = log_likelihood.item()
        ll0[i,1] = unintervene_liklelihood.item()  # 第一个是扰动的参数，第二个是没有被扰动的参数
    score_arrays["ll0"] = ll0
    score_arrays["params_all"] = params_all
    return score_arrays


def update_ll0_nb(score_arrays,weight_new,data,adj_matrix,regulator_list,gene_name,all_pert_sites,all_regulator_index,all_target_index):
    ll0 = score_arrays["ll0"]
    params_all = score_arrays["params_all"]

    for i in tqdm(regulator_list):
        unpert_params = params_all[i][1]
        unpert_params = unpert_params[None, :]
        variable = data[:,i]
        target_name = gene_name[i]
        target2regulator_index = all_pert_sites.index(target_name) if target_name in all_pert_sites else None
        if target2regulator_index is not None:
            groups =  split_number(len(all_pert_sites) + 1, target2regulator_index + 1)
        else:
            raise ValueError("pert?")
            #groups = [list(range(len(all_pert_sites) + 1))]
        
        if len(np.where(adj_matrix[:, i] == 1)[0]) != 0:
            old_regulator = list(np.where(adj_matrix[:, i] == 1)[0])
            regulater = data[:, old_regulator].reshape(data.shape[0], -1)      
            regulater = reorder_array(regulater, all_regulator_index)   
        else:
            regulater = None
        
        variable = reorder_array(variable, all_target_index) # 来使得
        # 针对干预的部分，重新计算系数和似然
        params, log_likelihood = m_step_fast_nb(regulater, variable, weight_new,[groups[0]])
        # 针对非干预的部分，用之前的系数来算此时的似然
        unintervene_liklelihood = unintervene_ll(regulater, variable, weight_new,[groups[1]],unpert_params)
        # if len(params) == 1:
        #     params_all[i] = (np.nan, params[0])
        #     ll0[i,1] = log_likelihood[0]
        # elif len(params) == 2:
        params_all[i] = (params[0],params_all[i][1]) # 只有干预的才会发生变化，否则不变
        ll0[i,0] = log_likelihood.item()
        ll0[i,1] = unintervene_liklelihood.item()  # 第一个是扰动的参数，第二个是没有被扰动的参数
    score_arrays["ll0"] = ll0
    score_arrays["params_all"] = params_all
    return score_arrays


def process_single_regulator(i, data, adj_matrix, gene_name, all_pert_sites, 
                             all_regulator_index, all_target_index, weight_new,regulator_list,downsample_size,type):
    """
    处理单个 regulator 的计算任务，供并行调用。
    返回: (index, params_result, ll0_result)
    """
    variable = data[:, i]
    target_name = gene_name[i]
    # 查找 target2regulator_index
    target2regulator_index = all_pert_sites.index(target_name) if target_name in all_pert_sites else None
    
    if target2regulator_index is not None:
        groups = split_number(len(all_pert_sites) + 1, target2regulator_index + 1) # [[target2regulator_index + 1],[0]]
    else:
        raise ValueError("???")
        return i, None, None


    # 获取 regulator 数据
    regulator_indices = np.where(adj_matrix[:, i] == 1)[0]
    if len(regulator_indices) != 0:
        old_regulator = regulator_list[regulator_indices]
        regulater = data[:, old_regulator].reshape(data.shape[0], -1)      
        regulater = reorder_array(regulater, all_regulator_index)   
    else:
        regulater = None
    
    # 重排 variable
    variable = reorder_array(variable, all_target_index)
    
    # 核心计算
    regulater, variable, weight_new = downsample_by_mask(regulater, variable, weight_new, downsample_size, random_state=0)
    if type == "normal":
        params, log_likelihood = m_step_fast(regulater, variable, weight_new, groups)
    elif type == "nb":
        params, log_likelihood = m_step_fast_nb(regulater, variable, weight_new, groups)

    # 整理结果
    params_res = None
    ll0_res = {} # 使用字典存储需要更新的位置和值 {col_index: value}

    if len(params) == 1:
        raise ValueError("???")
        params_res = (np.nan, params[0])
        ll0_res[1] = log_likelihood[0]
    elif len(params) == 2:
        if params[0].sum() == 0 and log_likelihood[0].sum() == 0: #此时为这种类型的干预的可能性太小了。
            params_res = (np.nan, params[1])
            ll0_res[0] = np.nan
            ll0_res[1] = log_likelihood[1]
        else:
            params_res = (params[0], params[1])
            ll0_res[0] = log_likelihood[0]
            ll0_res[1] = log_likelihood[1]
        
    return i, params_res, ll0_res


def update_ll0_parallel(score_arrays, weight_new, data, adj_matrix, regulator_list, 
                        gene_name, all_pert_sites, all_regulator_index, all_target_index, downsample_size,
                        n_jobs=10,type = "normal"):
    """
    并行版本的 update_ll0
    n_jobs: 并行核心数，-1 表示使用所有可用核心
    """
    ll0 = score_arrays["ll0"]
    params_all = score_arrays["params_all"]

    # 1. 并行执行计算
    # joblib 会自动处理 numpy 数组的共享内存，减少复制开销
    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(process_single_regulator)(
    #         i, data, adj_matrix, gene_name, all_pert_sites, 
    #         all_regulator_index, all_target_index, weight_new,
    #     ) for i in regulator_list
    # )
    results = [
        res for res in tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(process_single_regulator)(
                    i, data, adj_matrix, gene_name, all_pert_sites, 
                    all_regulator_index, all_target_index, weight_new,regulator_list,downsample_size,type
                ) for i in regulator_list
            ),
            total=len(regulator_list),
            desc="Processing Regulators"
        )
    ]
    # 2. 串行汇总结果 (写入操作通常很快，不需要并行)
    for i, params_res, ll0_res in results:
        # 如果返回的是 None (即 continue 的情况)，则跳过
        if params_res is None:
            continue
            
        # 更新 params_all
        params_all[i] = params_res
        
        # 更新 ll0
        for col_idx, val in ll0_res.items():
            ll0[i, col_idx] = val

    # 更新字典
    score_arrays["ll0"] = ll0
    score_arrays["params_all"] = params_all
    
    return score_arrays


import scipy.sparse as sparse

def get_normalize_data(adata):
    normalized_matrix = np.zeros(adata.shape, dtype=np.float32) # 使用float32节省一半内存

    batches = ["train", "test"]
    for batch in batches:
        print(f"Processing {batch}...")

        batch_mask = adata.obs["label"] == batch

        ctrl_mask = (adata.obs["label"] == batch) & (adata.obs["pert"] == "CTRL")

        batch_data = adata[batch_mask].X
        ctrl_data = adata[ctrl_mask].X

        if sparse.issparse(batch_data):
            batch_data = batch_data.toarray()
        if sparse.issparse(ctrl_data):
            ctrl_data = ctrl_data.toarray()
            
        mu = np.mean(ctrl_data, axis=0)
        std = np.std(ctrl_data, axis=0)

        std[std == 0] = 1

        z_scored_data = (batch_data - mu) / std

        normalized_matrix[np.where(batch_mask)[0], :] = z_scored_data
    return normalized_matrix


def get_pesdo_dict(adata,case):
    df = adata.obs.copy()
    df = df.reset_index(drop=True)
    # 按 cell_type 分组，然后对每组的 stemness_score 排序，得到 index
    if case == "fib2ipsc":
        sorted_indices = df.groupby("pert").apply(lambda x: x.sort_values("stemness_score").index.tolist())
    elif case == "fib2ia":
        sorted_indices = df.groupby("pert").apply(lambda x: x.sort_values("FdiAs_score").index.tolist())
    elif case == "fib2dc":
        sorted_indices = df.groupby("pert").apply(lambda x: x.sort_values("DC_score").index.tolist())
    elif case == "differential":
        sorted_indices = df.groupby("pert").apply(lambda x: x.sort_values("AEC_score").index.tolist())    
    sorted_indices_dict = sorted_indices.to_dict()
    return sorted_indices_dict


import numpy as np
from scipy.stats import wasserstein_distance


def get_prior_differential(adata,regulator_list):
    data_ctrl = adata[adata.obs["pert"] == "CTRL"].X
    data_pert = adata[adata.obs["pert"] != "CTRL"].X
    emd_list = []
    for i in regulator_list:
        col1 = data_ctrl[:, i]
        col2 = data_pert[:, i]
        emd = wasserstein_distance(col1, col2)
        emd_list.append(emd)

    emd_array = np.array(emd_list)  # 每列的推土机距离

    def softmax(arr):
        arr = np.asarray(arr)
        # 为了数值稳定性，减去最大值
        exp_arr = np.exp(arr - np.max(arr))
        return exp_arr / np.sum(exp_arr)

    softmax_scores = softmax(emd_array)
    return softmax_scores.reshape([1, -1])




def remove_from_ctrl(d: dict, to_remove: np.ndarray) -> dict:
    remove_set = set(to_remove.tolist())
    new_d = d.copy()
    new_d["CTRL"] = [x for x in d["CTRL"] if x not in remove_set]
    return new_d





##########################
def remove_digits(input_string):
    return re.sub(r"\d+", "", input_string)


def find_index(path, adata):
    matrix_df = pd.DataFrame(as_dense(adata.X))
    df = pd.read_csv(path, index_col=0)
    matrix_df["hash"] = matrix_df.apply(lambda row: hash(tuple(row)), axis=1)
    row_indices = []
    for row in df.values.T:
        row_hash = hash(tuple(row))
        matched_rows = matrix_df[matrix_df["hash"] == row_hash].index
        row_indices.append(matched_rows[0] if not matched_rows.empty else None)
    if any(idx is None for idx in row_indices):
        raise ValueError("Rows are None.")
    return row_indices, True


def get_pesdotime(path, row_indices=None):
    df = pd.read_csv(path, index_col=0)
    df.index = np.arange(len(df)) if row_indices is None else row_indices
    result = {}
    for col in df.columns:
        result[col] = df[col].dropna().sort_values().index.tolist()
    regulator_index = []
    target_index = []
    for key in result.keys():
        regulator_index.extend(result[key][:-1])
        target_index.extend(result[key][1:])
    return regulator_index, target_index


def get_sub_paths(main_path, name, end):
    base_name = remove_digits(name)
    sub_paths = []
    for root, dirs, _files in os.walk(main_path):
        for dir_name in dirs:
            if base_name in dir_name:
                folder_path = os.path.join(root, dir_name)
                for sub_dir in os.listdir(folder_path):
                    sub_folder_path = os.path.join(folder_path, sub_dir)
                    if os.path.isdir(sub_folder_path) and sub_dir.endswith(end):
                        sub_paths.append(sub_folder_path)
    return sub_paths


def extract_suffix(path, default="test1"):
    match = re.search(r"_(\d+)", os.path.basename(path))
    return match.group(1) if match else default


def get_intervene_index(all_regulator_index, all_target_index, target_name):
    unintervene_list_regulator = []
    unintervene_list_target = []
    if target_name in all_regulator_index:
        for key, value in all_regulator_index.items():
            if key != target_name:
                unintervene_list_regulator.extend(value)
        for key, value in all_target_index.items():
            if key != target_name:
                unintervene_list_target.extend(value)
        return (
            all_regulator_index[target_name],
            all_target_index[target_name],
        ), (
            unintervene_list_regulator,
            unintervene_list_target,
        )

    for key, value in all_regulator_index.items():
        if key != target_name:
            unintervene_list_regulator.extend(value)
    for key, value in all_target_index.items():
        if key != target_name:
            unintervene_list_target.extend(value)
    return (None, None), (unintervene_list_regulator, unintervene_list_target)


def get_all_peso_index(
    name,
    beeline_input_root="Data/beeline_inputs",
    beeline_h5ad_root="Data/beeline",
):
    """
    :param name:
    :return: 两个字典。key代表扰动的类型。values代表该扰动位点下所对应的expression数据在adata中所对应的行是什么？
    """
    pert_site_VSC = ['Nkx61', 'Nkx62', 'Nkx22', 'Pax6', 'Dbx1', 'Dbx2', 'Olig2', 'Irx3']
    pert_site_mCAD = ['Fgf8', 'Emx2', 'Pax6', 'Coup', 'Sp8']
    pert_site_GSD = ['UGR', 'CBX2', 'GATA4', 'WT1mKTS', 'WT1pKTS', 'NR5A1', 'NR0B1', 'SRY', 'SOX9', 'FGF9', 'PGD2',
                     'DMRT1', 'DHH', 'DKK1', 'AMH', 'WNT4', 'RSPO1', 'FOXL2', 'CTNNB1']
    pert_site_HSC = ['Gata2', 'Gata1', 'Fog1', 'Eklf', 'Fli1', 'Scl', 'Cebpa', 'Pu1', 'cJun', 'EgrNab', 'Gfi1']

    # name = "VSC"
    end = "-1-50-0.5" #
    main_path = str(Path(beeline_input_root) / name)
    adata = sc.read_h5ad(Path(beeline_h5ad_root) / f"{name}.h5ad")
    sub_paths = get_sub_paths(main_path, name, end)
    # 路径：验证是否支路的长度等于基因的数量+1
    df = pd.read_csv(os.path.join(sub_paths[0], "ExpressionData.csv"), index_col=0)
    if len(sub_paths) != len(df) + 1:
        raise ValueError('path not completed!!')

        # 接下来整合所有的代码。输入主路径，得到支路径，最终输出time path和data所处的path。单个单个的来，然后整合到一起，
    all_regulator_index = {}
    all_target_index = {}
    base_name = remove_digits(name)
    if base_name == "VSC":
        pert_sites = pert_site_VSC
    elif base_name == "mCAD":
        pert_sites = pert_site_mCAD
    elif base_name == "GSD":
        pert_sites = pert_site_GSD
    elif base_name == "HSC":
        pert_sites = pert_site_HSC

    for path in sub_paths:
        num = extract_suffix(path, default="test1") 
        if num == "test1":
            pert_name = "test1"
        else:
            if base_name == "VSC" or base_name == "HSC":
                pert_name = pert_sites[int(num) - 1]
            else:
                pert_name = pert_sites[int(num)]

        time_path = os.path.join(path, "PseudoTime.csv")
        data_path = os.path.join(path, "ExpressionData.csv")
        row_indices, _ = find_index(data_path, adata) # 是否expression data是对应到了adata的对应行的？是。只是可能出现多行对应同一个数值的情况。
        regulator_index, target_index = get_pesdotime(time_path, row_indices) # 针对当前的扰动场景，找到对应的index。
        if pert_name in all_regulator_index:
            raise ValueError('duplicated pert_name in all_regulator_index!!')
        elif pert_name in all_target_index:
            raise ValueError('duplicated pert_name in all_target_index!!')
        all_regulator_index[pert_name] = []  
        all_target_index[pert_name] = []  

        all_regulator_index[pert_name].extend(regulator_index)
        all_target_index[pert_name].extend(target_index)
    return all_regulator_index, all_target_index


import numpy as np
from numba import njit
from math import lgamma, log, exp, sqrt, fabs

# ==========================================
# 1. 辅助数学函数 (Digamma & Trigamma)
# ==========================================
# Numba 不支持 scipy.special，需手动实现以计算 theta 的梯度

@njit(fastmath=True, nogil=True)
def _digamma(x):
    """ Digamma 函数 (Psi) 的近似实现 """
    r = 0.0
    while x <= 5:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    t = f * (-1/12.0 + f * (1/120.0 + f * (-1/252.0 + f * (1/240.0 + f * (-1/132.0 + f * (691/32760.0 + f * (-1/12.0 + f * 3617/8160.0)))))))
    return r + log(x) - 0.5 / x + t

@njit(fastmath=True, nogil=True)
def _trigamma(x):
    """ Trigamma 函数 (Psi') 的近似实现 """
    r = 0.0
    while x <= 5:
        r += 1 / (x * x)
        x += 1
    f = 1 / (x * x)
    t = f * (1/6.0 + f * (-1/30.0 + f * (1/42.0 + f * (-1/30.0 + f * (5/66.0 + f * (-691/2730.0 + f * (7/6.0)))))))
    return r + 1 / x + 0.5 * f + t / x

# ==========================================
# 2. 核心优化函数
# ==========================================

@njit(fastmath=True, nogil=True)
def solve_beta_newton(X, y, w, beta_init, theta, max_iter=10, tol=1e-4, lambda_reg=1e-3):
    """
    固定 theta，求解 beta (保持你原有的逻辑，稍作鲁棒性优化)
    """
    n, p = X.shape
    beta = beta_init.copy()
    reg_diag = np.full(p, lambda_reg)
    reg_diag[0] = 0.0 
    
    for it in range(max_iter):
        # 限制 mu 的范围防止溢出
        mu = np.exp(np.clip(X @ beta, -20, 20))
        g = np.zeros(p)
        H = np.zeros((p, p))
        
        for i in range(n):
            wi = w[i]
            if wi < 1e-8: continue
            yi = y[i]
            mui = mu[i]
            
            denom = theta + mui
            factor = wi * theta / denom 
            resid = yi - mui
            
            grad_coef = factor * resid
            hess_coef = factor * mui # 这里近似 Hessian 也可以用 factor * mui * theta / (theta + mui)
            
            row = X[i]
            for j in range(p):
                xj = row[j]
                g[j] += grad_coef * xj
                for k in range(j, p):
                    H[j, k] -= hess_coef * xj * row[k] 
        
        for j in range(p):
            for k in range(j + 1, p):
                H[k, j] = H[j, k]
            H[j, j] -= reg_diag[j]

        # 增加 try-except 防止奇异矩阵
        try:
            delta = np.linalg.solve(-H, g)
        except:
            break 
        
        beta += delta
        if np.max(np.abs(delta)) < tol:
            break
    return beta

@njit(fastmath=True, nogil=True)
def solve_theta_newton(y, w, mu, theta_init, max_iter=10, tol=1e-4):
    """
    [新增] 固定 mu (即固定 beta)，使用牛顿法求解最优 theta
    """
    theta = theta_init
    n = len(y)
    
    for it in range(max_iter):
        grad = 0.0
        hess = 0.0
        
        for i in range(n):
            wi = w[i]
            if wi < 1e-8: continue
            yi = y[i]
            mui = mu[i]
            
            # 预计算常用项
            theta_mu = theta + mui
            
            # 梯度计算 dLL/dtheta
            # formula: sum( w * (psi(y+theta) - psi(theta) + ln(theta/(theta+mu)) + (mu-y)/(theta+mu)) )
            term_psi = _digamma(yi + theta) - _digamma(theta)
            term_log = log(theta / theta_mu)
            term_frac = (mui - yi) / theta_mu
            
            grad += wi * (term_psi + term_log + term_frac)
            
            # Hessian 计算 d2LL/dtheta2
            # formula: sum( w * (psi'(y+theta) - psi'(theta) + 1/theta - 1/(theta+mu) - (mu-y)/(theta+mu)^2 ) )
            # 注意: d/dtheta ((mu-y)/(theta+mu)) = -(mu-y)/(theta+mu)^2 = (y-mu)/(theta+mu)^2
            
            term_tri = _trigamma(yi + theta) - _trigamma(theta)
            term_inv = 1.0/theta - 1.0/theta_mu
            term_frac2 = (yi - mui) / (theta_mu * theta_mu)
            
            hess += wi * (term_tri + term_inv + term_frac2)
            
        # 阻尼牛顿步
        if fabs(hess) < 1e-10: 
            break
            
        delta = -grad / hess
        
        # 简单的线搜索或步长限制，防止 theta 变为负数
        if theta + delta <= 0:
            delta *= 0.5
            if theta + delta <= 0:
                delta = -0.5 * theta # 激进的回退
        
        theta += delta
        
        # 强制边界
        if theta < 1e-4: theta = 1e-4
        if theta > 1e4: theta = 1e4
        
        if fabs(delta) < tol:
            break
            
    return theta

@njit(fastmath=True, nogil=True, parallel=False)
def m_step_numba_nb(X_design, y, W_all, theta_init=10.0, lambda_reg=1e-3):
    """
    主函数：交替优化 Beta 和 Theta
    """
    n, p = X_design.shape
    n_groups = W_all.shape[1]
    
    # 输出 params: 前 p 个是 beta, 第 p+1 个是 theta
    out_params = np.zeros((n_groups, p + 1))
    out_lls = np.zeros(n_groups)

    # 交替优化的参数
    max_outer_iter = 5  # Beta 和 Theta 交替的次数
    tol_outer = 1e-3

    for g in range(n_groups):
        w_group = W_all[:, g]
        nz_idx = np.where(w_group > 1e-8)[0]
        k = len(nz_idx)
        
        if k == 0: 
            out_lls[g] = -np.inf
            continue

        X_sub = X_design[nz_idx]
        y_sub = y[nz_idx]
        w_sub = w_group[nz_idx]
        
        # --- 1. 初始化 (OLS) ---
        y_log = np.log(y_sub + 1.0)
        A = np.zeros((p, p))
        b = np.zeros(p)
        for i in range(k):
            row = X_sub[i]
            wi = w_sub[i]
            yl = y_log[i]
            for j in range(p):
                wx = wi * row[j]
                b[j] += wx * yl
                for l in range(p):
                    A[j, l] += wx * row[l]
        for j in range(p): A[j, j] += lambda_reg
        
        beta_curr = np.linalg.lstsq(A, b, rcond=-1)[0]
        theta_curr = theta_init
        
        # --- 2. 交替优化 (Coordinate Descent) ---
        for outer_it in range(max_outer_iter):
            beta_old = beta_curr.copy()
            theta_old = theta_curr
            
            # A. 优化 Beta (固定 Theta)
            beta_curr = solve_beta_newton(X_sub, y_sub, w_sub, beta_curr, theta_curr, lambda_reg=lambda_reg)
            
            # 计算当前的 mu 用于 Theta 优化
            mu_vec = np.exp(np.clip(X_sub @ beta_curr, -20, 20))
            
            # B. 优化 Theta (固定 Beta/Mu)
            theta_curr = solve_theta_newton(y_sub, w_sub, mu_vec, theta_curr)
            
            # 检查收敛
            diff_beta = np.max(np.abs(beta_curr - beta_old))
            diff_theta = np.abs(theta_curr - theta_old)
            
            if diff_beta < tol_outer and diff_theta < tol_outer:
                break
        
        # 存储结果
        out_params[g, :p] = beta_curr
        out_params[g, p] = theta_curr # 存储优化后的 theta
        
        # --- 3. 计算最终 Log-Likelihood ---
        # 重新计算 mu (虽然上面算过了，但为了保险起见用最新的 beta)
        mu_vec = np.exp(np.clip(X_sub @ beta_curr, -20, 20))
        
        # 预计算 theta 相关的常数
        lgamma_theta = lgamma(theta_curr)
        log_theta = log(theta_curr)
        
        ll_val = 0.0
        weight_sum = 0.0
        for i in range(k):
            yi = y_sub[i]
            wi = w_sub[i]
            mui = mu_vec[i]
            
            # NB Log-Likelihood
            term1 = lgamma(yi + theta_curr)
            term2 = lgamma(yi + 1.0)
            # term3 = lgamma_theta
            
            term4 = theta_curr * log_theta
            term5 = yi * log(mui) if mui > 1e-10 else -1e10
            term6 = (theta_curr + yi) * log(theta_curr + mui)
            
            log_pmf = term1 - term2 - lgamma_theta + term4 + term5 - term6
            
            ll_val += wi * log_pmf
            weight_sum += wi
        out_lls[g] = ll_val/weight_sum
        
    return out_params, out_lls


def m_step_fast_nb(X, y, responsibilities, groups, lambda_reg=1e-3, min_sigma=1e-8):
    # 1. 准备数据
    if X is None:
        X_design = np.ones((y.shape[0], 1))
    else:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    # 2. 转换权重
    W_all = prepare_weights(responsibilities, groups)
    # 3. Numba 计算
    # X_design_f, y_f, W_all_f = filter_data_by_y(X_design, y, W_all)

    return m_step_numba_nb(X_design, y, W_all, lambda_reg, min_sigma)


from scipy.special import gammaln

# ==========================================
# 1. 你提供的函数 (原封不动)
# ==========================================
def nb_linear_fit(x, y, parms=None):
    y = np.asarray(y)
    if x is None:
        X = np.ones((len(y), 1))
    else:
        # 注意：这里 nb_linear_fit 会自动加一列 1 (截距)
        # 所以传入的 x 应该是纯 Regulator 数据，不带截距列
        X = np.column_stack([np.ones(len(x)), x])
    
    n, p = X.shape
    
    if parms is None:
        lambda_reg = 1e-3
        I = np.eye(p)
        y_log = np.log(y + 1.0)
        beta = np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y_log
    else:
        # === 关键解析点 ===
        # 它假设 parms 的前 p 个是 beta，第 p+1 个是 theta
        beta = parms[:p]
        if len(parms) > p:
            theta = parms[p]
            
    eta = X @ beta
    mu = np.exp(np.clip(eta, -20, 20))
    
    # 计算 Log-Likelihood
    ll = (
        gammaln(y + theta)
        - gammaln(theta)
        - gammaln(y + 1)
        + theta * np.log(theta / (theta + mu))
        + y * np.log(mu / (theta + mu))
    )
    log_likelihood = np.sum(ll)
    
    # 返回 params 以便检查
    params_out = np.concatenate([beta, np.array([theta])])
    return params_out, log_likelihood


def get_tg(tf, distance=1, chip_atlas_root="Data/chip_atlas"):
    main_path = Path(chip_atlas_root) / f"target_genes_hg38_{distance}kb"
    # tf = "ADNP"
    path = os.path.join(main_path, f"{tf}.{distance}.tsv")
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        tg = list(df["Target_genes"])
        return tg, tf
    else:
        return None, tf


def get_chip_atlas(
    adata,
    distance,
    tf_list_path=None,
    chip_atlas_root="Data/chip_atlas",
):
    gene_name = list(adata.var["gene_name"])
    _, regulator_list1 = get_regulator_list(adata, tf_list_path=tf_list_path)
    n = len(gene_name)
    m = len(regulator_list1)
    # 初始化一个m x n的邻接矩阵
    adj_matrix = np.zeros((m, n), dtype=int)

    # 创建基因到索引的映射
    gene_to_index = {gene: idx for idx, gene in enumerate(gene_name)}  # adata.var的出场顺序
    tf_to_index = {tf: idx for idx, tf in enumerate(list(regulator_list1))}  # TF的出场顺序
    missing_tf = []
    for tf in regulator_list1:
        tg, tf = get_tg(tf, distance, chip_atlas_root=chip_atlas_root)
        if tg is None:
            missing_tf.append(tf)  # 这表示这个tf的target gene其实是没有被接收到的。
        else:
            i = tf_to_index[tf]
            j = [gene_to_index.get(item) for item in tg if item in gene_to_index]
            adj_matrix[i, [j]] = 1
    return adj_matrix, missing_tf


def top_k_binary_mask_strict(matrix, k=10):
    """
    保证严格保留前k大的值的位置为1，其余为0
    """
    flat = matrix.flatten()
    if k >= flat.size:
        return np.ones_like(matrix, dtype=int)

    # 获取前k大值的索引
    topk_indices = np.argpartition(-flat, k)[:k]

    # 构建空mask并填充对应位置
    mask_flat = np.zeros_like(flat, dtype=int)
    mask_flat[topk_indices] = 1

    # reshape成原矩阵形状
    return mask_flat.reshape(matrix.shape)



def evaluate_network_reconstruction(predicted_adj, true_adj):
    # def evaluate_network_reconstruction(predicted_adj, true_adj, threshold=0.5):
    """
    评估网络重构性能，计算 Precision 和 Recall.

    :param predicted_adj: ndarray, 预测的邻接矩阵
    :param true_adj: ndarray, 真实的邻接矩阵
    :param threshold: float, 二值化的阈值
    :return: precision, recall
    """
    # 确保输入是 numpy 数组
    predicted_adj = np.abs(np.array(predicted_adj))
    true_adj = np.abs(np.array(true_adj))

    # 对邻接矩阵进行二值化
    # predicted_binary = (predicted_adj >= threshold).astype(int)
    # true_binary = (true_adj > 0).astype(int)  # 真实矩阵也二值化，假设 >0 为存在边
    predicted_binary = predicted_adj
    true_binary = true_adj

    # 忽略对角线元素
    np.fill_diagonal(predicted_binary, 0)
    np.fill_diagonal(true_binary, 0)

    # 计算 TP, FP, FN
    TP = np.sum((predicted_binary == 1) & (true_binary == 1))  # True Positives
    FP = np.sum((predicted_binary == 1) & (true_binary == 0))  # False Positives
    FN = np.sum((predicted_binary == 0) & (true_binary == 1))  # False Negatives
    TN = np.sum((predicted_binary == 0) & (true_binary == 0))

    # Precision 和 Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score


def get_roc(adj_matrix, ground_truth,name):
    # 真实标签：1 是正类，0 是负类
    y_true = ground_truth.flatten()

    # 模型预测得分（可以理解为“是正类的概率”）
    y_scores = adj_matrix.flatten()
    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, marker='o')
    plt.title(f"ROC Curve of {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)

    # Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, marker='o')
    plt.title(f"Precision-Recall Curve of {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    logging.info(f"AUROC: {auroc:.4f}")
    logging.info(f"AUPRC: {auprc:.4f}")
    return auroc, auprc

def setup_logger(log_file="results.log"):
    # 如果已经存在 handler，就先移除（避免重复输出）
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),      # 输出到文件
            logging.StreamHandler()             # 输出到控制台
        ]
    )
