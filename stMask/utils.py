import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import adjusted_mutual_info_score,normalized_mutual_info_score,completeness_score,fowlkes_mallows_score, homogeneity_score

from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score

# the location of R (used for the mclust clustering)
os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'


def fix_seed(seed=2024):
    import random
    import torch
    from torch.backends import cudnn

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def Stats_Spatial_Net(adata):
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)

def Kmeans_cluster(adata, num_cluster, used_obsm='model_pred', key_added_pred="kmeans", random_seed=2024):
    np.random.seed(random_seed)
    cluster_model = KMeans(n_clusters=num_cluster, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(adata.obsm[used_obsm])
    adata.obs[key_added_pred] = cluster_labels
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='model_pred', key_added_pred="mclust", random_seed=2024):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added_pred] = mclust_res
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('int')
    adata.obs[key_added_pred] = adata.obs[key_added_pred].astype('category')
    return adata


def build_args():
    import argparse
    parser = argparse.ArgumentParser(description="stMask")
    parser.add_argument("--model_name", type=str, default="stMask")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--tissue_name", type=str, default="151507")

    parser.add_argument("--top_genes", type=int, default=2000)
    parser.add_argument("--genes_model", type=str, default="pca")
    parser.add_argument("--rad_cutoff", type=int, default=200)
    parser.add_argument("--k_cutoff", type=int, default=12)
    parser.add_argument("--graph_model", type=str, default="KNN")

    parser.add_argument('--nps', type=int, default=30)
    parser.add_argument('--gradient_clipping', type=float, default=5.)
    parser.add_argument("--need_refine", action='store_true', default=False)

    # 各模型的训练设置
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--max_epoch", type=int, default=400, help="number of training epochs")

    # ST params
    parser.add_argument("--edge_drop_rate", type=float, default=0.6)
    parser.add_argument("--feat_mask_rate", type=float, default=0.3)

    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)

    parser.add_argument('--bn', action='store_true', default=True)
    parser.add_argument("--att_dropout_rate", type=float, default=.2)
    parser.add_argument("--fc_dropout_rate", type=float, default=.5)
    parser.add_argument("--use_token", action='store_true', default=True)
    parser.add_argument("--rep_loss", type=str, default="cse")
    parser.add_argument("--rel_loss", type=str, default="ce")
    parser.add_argument("--alpha", type=float, default=2.0)

    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args(args=[])
    return args


def measureClusteringTrueLabel(labels_true, labels_pred):
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    cs = completeness_score(labels_true, labels_pred)
    fms = fowlkes_mallows_score(labels_true, labels_pred)
    vms = v_measure_score(labels_true, labels_pred)
    hs = homogeneity_score(labels_true, labels_pred)
    return ari, ami, nmi, cs, fms, vms, hs


def refine(adata, pred, shape="hexagon"):
    sample_id = adata.obs.index.tolist()
    dis = distance.cdist(adata.obsm['spatial'], adata.obsm['spatial'], 'euclidean')
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred


def save_args_to_file(args, filename):
    with open(filename, 'w') as file:
        file.write('Parsed Arguments:\n')
        for arg, value in vars(args).items():
            arg_info = f"{arg}: {value}\n"
            file.write(arg_info)