import shutil
import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np
import torch

import stMask as stm
from pathlib import Path
import scanpy as sc
import pandas as pd
from sklearn import metrics

def train_one(args, adata, tissue_name=' '):
    net = stm.stMASK(adata,
                     tissue_name=tissue_name,
                     num_clusters=args.n_clusters,
                     genes_model='pca',
                     top_genes=args.top_genes,
                     rad_cutoff=200,
                     k_cutoff=args.k_cutoff,
                     graph_model='KNN',
                     device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                     learning_rate=args.learning_rate,
                     weight_decay=args.weight_decay,
                     max_epoch=args.max_epoch,
                     gradient_clipping=args.gradient_clipping,
                     feat_mask_rate=args.feat_mask_rate,
                     edge_drop_rate=args.edge_drop_rate,
                     hidden_dim=args.hidden_dim,
                     latent_dim=args.latent_dim,
                     bn=args.bn,
                     att_dropout_rate=args.att_dropout_rate,
                     fc_dropout_rate=args.fc_dropout_rate,
                     use_token=args.use_token,
                     rep_loss=args.rep_loss,
                     rel_loss=args.rel_loss,
                     alpha=args.alpha,
                     lam=args.lam,
                     random_seed=args.seed,
                     nps=args.nps)
    net.train()
    method = "kmeans"
    net.process(method=method)
    # net.clustering(method=method)
    adata = net.get_adata()
    sub_adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
    ARI = metrics.adjusted_rand_score(sub_adata.obs['Ground Truth'], sub_adata.obs[method])
    NMI = metrics.normalized_mutual_info_score(sub_adata.obs['Ground Truth'], sub_adata.obs[method])
    print(f"ARI:{ARI}\tNMI:{NMI}")
    return ARI, NMI, adata

def train_dlpfc():
    ari_list = []
    nmi_list = []
    args = stm.utils.build_args()
    args.hidden_dim, args.latent_dim = 512, 256
    args.max_epoch = 500
    args.lam = 1.3
    args.feat_mask_rate = 0.2
    args.edge_drop_rate = 0.2
    args.top_genes = 2000
    args.k_cutoff = 12
    for tissue_name in ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672",
                        "151673", "151674", "151675", "151676"]:
        n_clusters = 5 if tissue_name in ['151669', '151670', '151671', '151672'] else 7
        args.n_clusters = n_clusters

        data_root = Path("D:\\project\\datasets\\DLPFC\\")
        count_file = tissue_name + "_filtered_feature_bc_matrix.h5"
        adata = sc.read_visium(data_root / tissue_name, count_file=count_file)
        truth_path = "D:\\project\\datasets\\DLPFC\\" + tissue_name + '/' + tissue_name + '_truth.txt'
        Ann_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
        Ann_df.columns = ['Ground Truth']
        adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

        ARI, NMI, adata = train_one(args, adata, tissue_name)
        ari_list.append(ARI)
        nmi_list.append(NMI)
    mid_ari = np.median(ari_list)
    mid_nmi = np.median(nmi_list)
    print(f"mid_ari:{mid_ari}\tmid_nmi:{mid_nmi}")


if __name__ == '__main__':
    train_dlpfc()