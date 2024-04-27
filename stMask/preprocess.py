import warnings
warnings.filterwarnings('ignore')
import os
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors
from torch_geometric.data import Data
from pathlib import Path

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)

def load_feat(adata, top_genes=3000, model="pca"):
    assert (model in ['pca', 'hvg', 'other'])
    if model == "pca":
        adata.var_names_make_unique()
        if isinstance(adata.X, np.ndarray):
            adata.layers['count'] = adata.X
        else:
            adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=top_genes)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)
        from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['feat'] = adata_X
        print(f"adata.obsm['feat'].shape:{adata.obsm['feat'].shape}")

    elif model == "hvg":
        # Expression data preprocessing
        adata.var_names_make_unique()
        prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        # Normalization
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
        adata_Vars =  adata[:, adata.var['highly_variable']]
        # sc.pp.scale(adata)
        adata.obsm['feat'] = adata_Vars.X[:, ]

    elif model == "other":
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_genes)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False, max_value=10)
        adata.X = sp.csr_matrix(adata.X)
        adata_Vars =  adata[:, adata.var['highly_variable']]
        adata.obsm['feat'] = adata_Vars.X[:, ]

    return adata



def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    # #########
    # X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    # cells = np.array(X.index)
    # cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # if 'Spatial_Net' not in adata.uns.keys():
    #     raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    #
    # Spatial_Net = adata.uns['Spatial_Net']
    # G_df = Spatial_Net.copy()
    # G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    # G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # G = G + sp.eye(G.shape[0])  # self-loop
    # adata.uns['adj'] = G
    return adata


def Transfer_pytorch_Data(adata,  weightless=True):
    if weightless:
        return weightless_undirected_graph(adata)
    else:
        return powered_undirected_graph(adata)


def weightless_undirected_graph(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    edgeList = np.nonzero(G)
    if type(adata.obsm['feat']) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['feat']))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['feat'].todense()))  # .todense()
    return data

def powered_undirected_graph(adata):
    pass

if __name__ == '__main__':
    # sample name
    sample_name = '151676'
    n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7
    # path
    data_root = Path("D:\\project\\datasets\\DLPFC\\")
    count_file = sample_name + "_filtered_feature_bc_matrix.h5"
    adata = sc.read_visium(data_root / sample_name, count_file=count_file)
    adata = load_feat(adata, model="pca")
    print(adata.obsm['feat'].shape)