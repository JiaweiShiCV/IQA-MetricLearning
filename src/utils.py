import itertools
import matplotlib.pyplot as plt  # 绘图库
import torch.nn.functional as F
import numpy as np
import os
import torch
import pickle

def plot_confusion_matrix(cm, labels_name, title, acc):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))  # 获取标签的间隔数
    plt.xticks(num_class, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_class, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    plt.savefig(f"./plots/cm_{acc:.4f}.png", format='png')
    plt.show()

def collect_statistics(labelname_list):
    mean, std = [], []
    for name in labelname_list:
        with open(f"inception_{name}.pkl", 'rb') as f:
            info = pickle.load(f)
        mean.append(info['mean'][None, :]) 
        std.append(info['std'][None, :])
    mean = torch.from_numpy(np.concatenate(mean, axis=0))
    std = torch.from_numpy(np.concatenate(std, axis=0))
    return mean, std

def nearest_centroid(features, labelname_list):
    mean, std = collect_statistics(labelname_list) # mean, std [n, 2048]
    preds = []
    for feature_batch in torch.split(features, 8, 0):
        feature_batch_repeat = torch.repeat_interleave(feature_batch[:, None, :], repeats=len(labelname_list), dim=1)
        distmat = torch.abs(feature_batch_repeat - mean) / (std+1e-9) 
        distmat = torch.norm(distmat, dim=2, p=2) 
        _, pred_batch = torch.min(distmat, dim=1)
        preds.append(pred_batch)
    preds = torch.cat(preds)
    return preds

def nearest_neighbor(features, others, labels):
    preds = []
    for feature_batch in torch.split(features, 8, 0):
        distmat = build_dist(feature_batch, others, metric='euclidean')
        _, pred_index_batch = torch.topk(distmat, k=2, dim=1, largest=False)
        pred_batch = labels[pred_index_batch[:, 1]] # Pick the 2nd index because the 1st is itself.
        preds.append(pred_batch)
    preds = torch.cat(preds)
    return preds


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

@torch.no_grad()
def build_dist(feat_1: torch.Tensor, feat_2: torch.Tensor, metric: str = "euclidean", **kwargs) -> np.ndarray:
    r"""Compute distance between two feature embeddings.

    Args:
        feat_1 (torch.Tensor): 2-D feature with batch dimension.
        feat_2 (torch.Tensor): 2-D feature with batch dimension.
        metric:

    Returns:
        numpy.ndarray: distance matrix.
    """
    assert metric in ["cosine", "euclidean", "jaccard"], "Expected metrics are cosine, euclidean and jaccard, " \
                                                         "but got {}".format(metric)

    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)

    elif metric == "jaccard":
        feat = torch.cat((feat_1, feat_2), dim=0)
        dist = compute_jaccard_distance(feat, k1=kwargs["k1"], k2=kwargs["k2"], search_option=0)
        return dist[: feat_1.size(0), feat_1.size(0):]

@torch.no_grad()
def compute_euclidean_distance(features, others):
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())
    return dist_m

@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m

@torch.no_grad()
def compute_jaccard_distance():
    pass