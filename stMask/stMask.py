import os.path

import numpy as np
import torch
from tqdm import tqdm

from .preprocess import load_feat, Cal_Spatial_Net, Transfer_pytorch_Data
from .utils import fix_seed, Stats_Spatial_Net, mclust_R, Kmeans_cluster
from .model import stMask_model
class stMASK:
    def __init__(self,
                 adata,
                 tissue_name="BRCA",
                 num_clusters=20,
                 top_genes=4000,
                 genes_model="hvg",  # 'pca', 'hvg'
                 rad_cutoff=300,
                 k_cutoff=12,
                 graph_model='Radius',  # 'Radius', 'KNN'
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 weight_decay=2e-4,
                 max_epoch=1500,
                 gradient_clipping=5,
                 feat_mask_rate=0.3,
                 edge_drop_rate=0.6,
                 hidden_dim=512,
                 latent_dim=256,
                 bn=True,
                 att_dropout_rate=0.2,
                 fc_dropout_rate=0.5,
                 use_token=True,
                 alpha=2,
                 rep_loss="cse",
                 rel_loss="ce",
                 lam=1.4,
                 random_seed=2024,
                 nps=30,
                 ):

        self.__adata = adata.copy()
        self.__tissue_name = tissue_name
        self.__top_genes = top_genes
        self.__genes_model = genes_model
        self.__rad_cutoff = rad_cutoff
        self.__k_cutoff = k_cutoff
        self.__graph_model = graph_model
        self.__device = device
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay
        self.__max_epoch = max_epoch
        self.__gradient_clipping = gradient_clipping
        self.__feat_mask_rate = feat_mask_rate
        self.__edge_drop_rate = edge_drop_rate
        self.__hidden_dim = hidden_dim
        self.__latent_dim = latent_dim
        self.__bn = bn
        self.__att_dropout_rate = att_dropout_rate
        self.__fc_dropout_rate = fc_dropout_rate
        self.__use_token = use_token
        self.__alpha = alpha
        self.__rep_loss = rep_loss
        self.__rel_loss = rel_loss
        self.__lam = lam
        self.__nps = nps


        fix_seed(random_seed)

        if 'highly_variable' not in self.__adata.var.keys() and 'feat' not in adata.obsm.keys():
            self.__adata = load_feat(self.__adata, top_genes=self.__top_genes, model=self.__genes_model)

        if 'Spatial_Net' not in self.__adata.uns.keys():
            Cal_Spatial_Net(self.__adata, rad_cutoff=self.__rad_cutoff, k_cutoff=self.__k_cutoff, model=self.__graph_model)

        self.num_clusters = num_clusters # 5 if self.tissue_name in ['151669', '151670', '151671', '151672'] else 7
        print(self.__adata.obsm['feat'].shape)

    def train(self):
        data = Transfer_pytorch_Data(self.__adata).to(self.__device)
        output_dim = input_dim = data.x.shape[-1]
        features_dims = [input_dim, self.__hidden_dim, self.__latent_dim, output_dim]
        self.model = stMask_model(features_dims, bn=self.__bn,
                                  att_dropout_rate=self.__att_dropout_rate,fc_dropout_rate=self.__fc_dropout_rate,
                                  use_token=self.__use_token, alpha=self.__alpha,
                                  edge_drop_rate=self.__edge_drop_rate, feat_mask_rate=self.__feat_mask_rate,
                                  rep_loss=self.__rep_loss,rel_loss=self.__rel_loss).to(self.__device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.__learning_rate, weight_decay=self.__weight_decay)

        y_pred_last = None
        epoch_iter = tqdm(range(self.__max_epoch))
        for epoch in epoch_iter:
            self.model.train()
            self.optimizer.zero_grad()

            feat_loss, topo_loss = self.model(data)

            loss = feat_loss + topo_loss * self.__lam
            loss.backward()
            gradient_clipping = self.__gradient_clipping
            if gradient_clipping > 1:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
            self.optimizer.step()
            epoch_iter.set_description(f"Dataset_Name:{self.__tissue_name}, Ep {epoch}: train loss:{loss.item():.4f}")

    def process(self, method="kmeans"):
        data = Transfer_pytorch_Data(self.__adata).to(self.__device)
        with torch.no_grad():
            self.model.eval()
            h, z = self.model.recon(data=data)
            rep = h.to('cpu').detach().numpy()
            rec = z.to('cpu').detach().numpy()
            if rep.shape[-1] > 64:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.__nps)
                rep = pca.fit_transform(rep)
            self.__adata.obsm["eval_pred"] = rep
            self.__adata.obsm["eval_recon"] = rec

            if method == "mclust":
                mclust_R(self.__adata, num_cluster=self.num_clusters, used_obsm="eval_pred", key_added_pred=method)
            elif method == "kmeans":
                Kmeans_cluster(self.__adata, num_cluster=self.num_clusters, used_obsm="eval_pred", key_added_pred=method)



    def show_Stats_Spatial_Net(self):
        Stats_Spatial_Net(self.__adata)

    def save_model_dict(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def save_model(self, save_model_file):
        torch.save(self.model, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model_dict(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def load_model(self, save_model_file):
        self.model = torch.load(save_model_file)
        print('Loading model from %s' % save_model_file)


    def get_adata(self):
        return self.__adata

