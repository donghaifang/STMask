import copy
from functools import partial
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import (
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

try:
    import torch_cluster  # noqa

    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.utils import add_self_loops, negative_sampling, degree

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, bn=True, dropout_rate=.1, act="prelu", bias=True):
        super().__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        # self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim, heads=1, dropout=dropout_rate, concat=False, bias=bias)
        # self.bn1 = bn(hidden_dim * 1)
        # self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=latent_dim, heads=1, dropout=dropout_rate, concat=False, bias=bias)
        # self.bn2 = bn(latent_dim * 1)
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.bn1 = bn(hidden_dim * 1)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=latent_dim)
        self.bn2 = bn(latent_dim * 1)
        self.activation = create_activation(act)

    def forward(self, x, edge_index):
        h = self.activation(self.bn2(self.conv2(self.activation(self.bn1(self.conv1(x, edge_index))), edge_index)))
        return h

class FeatureDecoder(nn.Module):
    def __init__(self, latent_dim,  output_dim, dropout_rate=.1, act="prelu", bias=True):
        super().__init__()
        # self.conv1 = GCNConv(in_channels=latent_dim, out_channels=output_dim, heads=1, dropout=dropout_rate, concat=False, bias=bias)
        self.conv1 = GCNConv(in_channels=latent_dim, out_channels=output_dim)
        self.activation = create_activation(act)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        return h

class TopologyDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim=1, dropout_rate=0.5, act="relu"):
        super().__init__()
        self.fc1 = Linear(in_channels=input_dim, out_channels=latent_dim)
        self.fc1.reset_parameters()
        self.fc2 = Linear(in_channels=latent_dim, out_channels=output_dim)
        self.fc2.reset_parameters()
        self.d_drop = nn.Dropout(dropout_rate)
        self.activation = create_activation(act)

    def forward(self, x, edge_index):
        h = x[edge_index[0]] * x[edge_index[1]]
        h = self.fc2(self.activation(self.fc1(self.d_drop(h))))
        return h

class stMask_model(nn.Module):
    def __init__(self, features_dims, bn=False, att_dropout_rate=.2, fc_dropout_rate=.5, use_token=True, alpha=2, edge_drop_rate=0.3, feat_mask_rate=0.3, rep_loss="cse",rel_loss="ce"):
        super().__init__()
        [input_dim, hidden_dim, latent_dim, output_dim] = features_dims
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, bn=bn, dropout_rate=att_dropout_rate, act="prelu", bias=True)

        self.use_token = use_token
        if self.use_token:
            self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self.encoder_to_decoder = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.feat_deocder = FeatureDecoder(latent_dim,  output_dim, dropout_rate=att_dropout_rate, act="prelu", bias=True)
        self.topo_decoder = TopologyDecoder(latent_dim, 2*latent_dim, 1, fc_dropout_rate)


        self.feat_loss = self.setup_loss_fn(rep_loss, alpha)
        self.edge_loss = self.setup_loss_fn(rel_loss)

        self.edge_drop_rate = edge_drop_rate
        self.feat_mask_rate = feat_mask_rate

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        use_mask_x, mask_nodes = self.mask_feature(x, self.feat_mask_rate)
        remaining_edges, masked_edges = self.dropout_edge(edge_index, self.edge_drop_rate)

        rep_x = self.encoder(use_mask_x, edge_index)
        rep_e = self.encoder(x, remaining_edges)

        # remasking feats
        rec_x = self.encoder_to_decoder(rep_x)
        rec_x[mask_nodes] = 0
        rec_x = self.feat_deocder(rec_x, edge_index)
        feat_loss = self.feat_loss(x[mask_nodes], rec_x[mask_nodes])

        # sampling neg edges
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.random_negative_sampler(
            aug_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=masked_edges.view(2, -1).size(1),
        ).view_as(masked_edges)

        pos_edge = self.topo_decoder(rep_e, masked_edges)
        neg_edge = self.topo_decoder(rep_e, neg_edges)
        topo_loss = self.ce_loss(pos_edge, neg_edge)

        return feat_loss, topo_loss

    def setup_loss_fn(self, loss_fn, alpha_l=2):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "cse":
            criterion = partial(self.sce_loss, alpha=alpha_l)
        elif loss_fn == "ce":
            criterion = partial(self.ce_loss)
        else:
            raise NotImplementedError
        return criterion

    def sce_loss(self, x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        loss = loss.mean()
        return loss

    def ce_loss(self, pos_out, neg_out):
        pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
        return pos_loss + neg_loss

    def mask_feature(self, x, feat_mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(feat_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        out_x = x.clone()
        if self.use_token:
            out_x[mask_nodes] += self.enc_mask_token
        else:
            out_x[mask_nodes] = 0.0
        return out_x, mask_nodes #, keep_nodes

    def mask_features(self, x, feat_mask_rate=0.3):
        mask_nodes = torch.empty((x.size(0),), dtype=torch.float32, device=x.device).uniform_(0, 1) < feat_mask_rate
        mask_x = x.clone()
        mask_x[mask_nodes] = 0
        if self.use_token:
            mask_x[mask_nodes] += self.enc_mask_token
        return mask_x, mask_nodes

    def dropout_edge(self, my_edge_index, edge_drop_rate=0.3):
        edge_index = my_edge_index.clone()
        p = torch.zeros(edge_index.shape[1]).to(edge_index.device) + 1 - edge_drop_rate
        stay = torch.bernoulli(p).to(torch.bool)
        mask = ~stay
        remaining_edges, masked_edges = edge_index[:, stay], edge_index[:, mask]
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges

    def random_negative_sampler(self, edge_index, num_nodes, num_neg_samples):
        neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
        return neg_edges

    @torch.no_grad()
    def embed(self, data):
        x = data.x
        edge_index = data.edge_index
        h = self.encoder(x, edge_index)
        return h

    @torch.no_grad()
    def recon(self, data):
        x = data.x
        edge_index = data.edge_index
        h = self.encoder(x, edge_index)
        rec = self.encoder_to_decoder(h)
        rec = self.feat_deocder(rec, edge_index)
        return h, rec


    @torch.no_grad()
    def embed_masking(self, data):
        x = data.x
        edge_index = data.edge_index
        use_mask_x, mask_nodes = self.mask_feature(x, self.feat_mask_rate)
        remaining_edges, masked_edges = self.dropout_edge(edge_index, self.edge_drop_rate)
        rep_x = self.encoder(use_mask_x, edge_index)
        rep_e = self.encoder(x, remaining_edges)
        return rep_x, rep_e
