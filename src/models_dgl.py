
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

import dgl
import time
from dgl.nn.pytorch.conv import GINConv, GraphConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling



def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=2**-0.5)
        # nn.init.constant_(m.bias, 0.0)

class GIN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=1, num_hidden=32, num_layers=5, agg='mean', act=F.relu, pooling='max'):
        super(GIN, self).__init__()

        # if data.x is None:
        #   data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)
        # dataset.data.edge_attr = None

        # num_features = dataset.num_features
        dim = num_hidden

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1, aggregator_type=agg, activation=act)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers-1):
            self.convs.append(GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)), \
                                aggregator_type=agg, activation=act))
            self.bns.append(torch.nn.BatchNorm1d(dim))

        if pooling == 'max':
            self.pool_layer = MaxPooling()
        elif pooling == 'mean':
            self.pool_layer = AvgPooling()
        elif pooling == 'sum':
            self.pool_layer = SumPooling()

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

    def forward(self, x, graph):
        if 'w' in graph.edata.keys():
            edge_weight = graph.edata['w']
            # print('---Edge Weighted!')
        else:
            edge_weight = None
        x = self.conv1(graph, x, edge_weight=edge_weight)
        x = self.bn1(x)
        for i, conv in enumerate(self.convs):
            x = self.convs[i](graph, x, edge_weight=edge_weight)
            x = self.bns[i](x)
        # x = global_add_pool(x, batch)
        node_list = graph.batch_num_nodes()
        x_list = []
        idx = 0
        for num_node in node_list:
            idx += num_node
            x_list.append(x[idx-1, :])
        x = torch.stack(x_list, dim=0)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=2, num_hidden=32, num_layers=5, act=F.relu, pooling='max'):
        super(GCN, self).__init__()

        self.dim = num_hidden
        self.conv1 = GraphConv(in_feats=num_features, out_feats=self.dim, activation=act)
        self.bn1 = torch.nn.BatchNorm1d(self.dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers-1):
            self.convs.append(GraphConv(in_feats=self.dim, out_feats=self.dim, activation=act))
            self.bns.append(torch.nn.BatchNorm1d(self.dim))

        if pooling == 'max':
            self.pool_layer = MaxPooling()
        elif pooling == 'mean':
            self.pool_layer = AvgPooling()
        elif pooling == 'sum':
            self.pool_layer = SumPooling()

        self.fc1 = Linear(self.dim, self.dim)
        self.fc2 = Linear(self.dim, num_classes)
    
    def forward(self, x, graph):
        if 'w' in graph.edata.keys():
            edge_weight = graph.edata['w']
            # print('---Edge Weighted!')
        else:
            edge_weight = None
        x = self.conv1(graph, x, edge_weight=edge_weight)
        x = self.bn1(x)
        for i, conv in enumerate(self.convs):
            x = self.convs[i](graph, x, edge_weight=edge_weight)
            x = self.bns[i](x)
            # x = F.dropout(x, p=0.5, training=self.training)
        # x = global_add_pool(x, batch)
        # x = self.pool_layer(graph, x)
        node_list = graph.batch_num_nodes()
        x_list = []
        idx = 0
        for num_node in node_list:
            idx += num_node
            x_list.append(x[idx-1, :])
        x = torch.stack(x_list, dim=0)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
    


class MySpatialEncoder(nn.Module):
    def __init__(self, max_dist, num_heads=1):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        # deactivate node pair between which the distance is -1,  VNODE dist -2
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_heads, 
            # padding_idx=1
        )
        # self.mlp = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(num_heads, num_heads)
        # )

    def forward(self, dist, mask=None, padding=float('-inf')):
        dist_embedding = self.embedding_table(dist)
        # dist_embedding = self.mlp(dist_embedding)
        if mask is not None:
            dist_embedding[mask.to(torch.bool)] = padding
        
        return dist_embedding
    

class BiasedMultiheadAttention(nn.Module):
    r"""Dense Multi-Head Attention Module with Graph Attention Bias.

    Compute attention between nodes with attention bias obtained from graph
    structures, as introduced in `Do Transformers Really Perform Bad for
    Graph Representation? <https://arxiv.org/pdf/2106.05234>`__

    .. math::

        \text{Attn}=\text{softmax}(\dfrac{QK^T}{\sqrt{d}} \circ b)

    :math:`Q` and :math:`K` are feature representation of nodes. :math:`d`
    is the corresponding :attr:`feat_size`. :math:`b` is attention bias, which
    can be additive or multiplicative according to the operator :math:`\circ`.

    Adapted from dgl.nn.pytorch.graph_transformer.BiasedMultiheadAttention()
    Parameters
    ----------
    feat_size : int
        Feature size.
    num_heads : int
        Number of attention heads, by which attr:`feat_size` is divisible.
    bias : bool, optional
        If True, it uses bias for linear projection. Default: True.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    attn_drop : float, optional
        Dropout probability on attention weights. Defalt: 0.1.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import BiasedMultiheadAttention

    >>> ndata = th.rand(16, 100, 512)
    >>> bias = th.rand(16, 100, 100, 8)
    >>> net = BiasedMultiheadAttention(feat_size=512, num_heads=8)
    >>> out = net(ndata, bias)
    """

    def __init__(
        self,
        feat_size,
        num_heads,
        bias=True,
        attn_bias_type="add",
        attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
            self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of projection matrices, the same settings as that in Graphormer."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None, attn_mul=None):
        """Forward computation.

        Parameters
        ----------
        ndata : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid positions, where
            invalid positions are indicated by non-zero values. Shape: (batch_size, N, N).
        attn_mul : torch.Tensor, optional
            The attention multiplier is used in Graphormer-GD for calculating the Hadamard
            product attention (attn_mul * attn_weight)
        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)
        """
        q_h = self.q_proj(ndata).transpose(0, 1)
        k_h = self.k_proj(ndata).transpose(0, 1)
        v_h = self.v_proj(ndata).transpose(0, 1)
        bsz, N, _ = ndata.shape
        q_h = (
            q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            / self.scaling
        )
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(
            1, 2, 0
        )
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )

        attn_weights = (
            torch.bmm(q_h, k_h)
            .transpose(0, 2)
            .reshape(N, N, bsz, self.num_heads)
            .transpose(0, 2)
        )

        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to(torch.bool)] = float("-inf")
        attn_weights = F.softmax(
            attn_weights.transpose(0, 2)
            .reshape(N, N, bsz * self.num_heads)
            .transpose(0, 2),
            dim=2,
        )

        if attn_mul is not None:
            # bsz N N num_head -> bsz*num_head N N
            attn_mul = attn_mul.transpose(0, 2).reshape(N, N, bsz * self.num_heads).transpose(0, 2)
            attn_weights = attn_mul * attn_weights

        attn_weights = self.dropout(attn_weights)

        attn = torch.bmm(attn_weights, v_h).transpose(0, 1)

        attn = self.out_proj(
            attn.reshape(N, bsz, self.feat_size).transpose(0, 1)
        )

        return attn
    

class GraphormerLayer(nn.Module):
    r"""Graphormer Layer with Dense Multi-Head Attention, as introduced
    in `Do Transformers Really Perform Bad for Graph Representation?
    <https://arxiv.org/pdf/2106.05234>`__

    Adapted from dgl.nn.pytorch.graph_transformer.GraphormerLayer()
    Parameters
    ----------
    feat_size : int
        Feature size.
    hidden_size : int
        Hidden size of feedforward layers.
    num_heads : int
        Number of attention heads, by which :attr:`feat_size` is divisible.
    attn_bias_type : str, optional
        The type of attention bias used for modifying attention. Selected from
        'add' or 'mul'. Default: 'add'.

        * 'add' is for additive attention bias.
        * 'mul' is for multiplicative attention bias.
    norm_first : bool, optional
        If True, it performs layer normalization before attention and
        feedforward operations. Otherwise, it applies layer normalization
        afterwards. Default: False.
    dropout : float, optional
        Dropout probability. Default: 0.1.
    activation : callable activation layer, optional
        Activation function. Default: nn.ReLU().

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import GraphormerLayer

    >>> batch_size = 16
    >>> num_nodes = 100
    >>> feat_size = 512
    >>> num_heads = 8
    >>> nfeat = th.rand(batch_size, num_nodes, feat_size)
    >>> bias = th.rand(batch_size, num_nodes, num_nodes, num_heads)
    >>> net = GraphormerLayer(
            feat_size=feat_size,
            hidden_size=2048,
            num_heads=num_heads
        )
    >>> out = net(nfeat, bias)
    """

    def __init__(
        self,
        feat_size,
        hidden_size,
        num_heads,
        attn_bias_type="add",
        norm_first=False,
        dropout=0.1,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMultiheadAttention(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None, attn_mul=None):
        """Forward computation.

        Parameters
        ----------
        nfeat : torch.Tensor
            A 3D input tensor. Shape: (batch_size, N, :attr:`feat_size`), where
            N is the maximum number of nodes.
        attn_bias : torch.Tensor, optional
            The attention bias used for attention modification. Shape:
            (batch_size, N, N, :attr:`num_heads`).
        attn_mask : torch.Tensor, optional
            The attention mask used for avoiding computation on invalid
            positions. Shape: (batch_size, N, N).
        attn_mask : torch.Tensor, optional
            The attention multiplier used for adaptive spatial encoder scaling in 
            GraphormerGD. Shape: (batch_size, N, N).

        Returns
        -------
        y : torch.Tensor
            The output tensor. Shape: (batch_size, N, :attr:`feat_size`)
        """
        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat = self.attn(nfeat, attn_bias, attn_mask, attn_mul)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat
    


class Graphormer(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=2, num_hidden=64, num_heads=4, max_dist=20, max_degree=20, num_layers=5, act=nn.ReLU(), embed_degree=True, bn=False, device=torch.cuda):
        super(Graphormer, self).__init__()

        self.dim = num_hidden
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.embed_degree = embed_degree
        self.max_degree = max_degree
        self.max_dist = max_dist

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.use_bn = bn
        self.spatial_encoders = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GraphormerLayer(feat_size=self.dim, hidden_size=self.dim, num_heads=self.num_heads, activation=act))
            if self.use_bn:
                self.bns.append(torch.nn.BatchNorm1d(self.dim))
            self.spatial_encoders.append(MySpatialEncoder(max_dist=max_dist, num_heads=self.num_heads))

        self.input_embedding = nn.Sequential(
            Linear(self.num_features, self.dim),
            # nn.ReLU(),
            # Linear(self.dim, self.dim, bias=False),
            # nn.LayerNorm(self.dim)
        )
        
        # self.input_embedding.apply(init_weights)
        
        self.degree_embedding = nn.Embedding(max_degree+2, self.dim, padding_idx=0)
        self.fc1 = Linear(self.dim, self.dim)
        self.fc2 = Linear(self.dim, num_classes)

        # nn.init.xavier_uniform_(self.fc1.weight, gain=2**-0.5)
        # nn.init.constant_(self.fc1.bias, 0.0)
        # nn.init.xavier_uniform_(self.fc2.weight, gain=2**-0.5)
        # nn.init.constant_(self.fc2.bias, 0.0)


    def forward(self, x, graph):
        g_list = dgl.unbatch(graph)
        node_num = graph.batch_num_nodes()
        max_node_num = torch.max(node_num)
        x = []
        in_degs = []
        out_degs = []
        sp_dist_list = []
        mask_list = []
        for idx, g in enumerate(g_list):
            feat = torch.zeros(max_node_num, self.num_features).to(self.device)
            in_deg = torch.zeros(max_node_num, dtype=torch.long).to(self.device)
            out_deg = torch.zeros(max_node_num, dtype=torch.long).to(self.device)
            sp_dist = torch.zeros(max_node_num, max_node_num).long().to(self.device)
            feat[0:node_num[idx]] = g.ndata['node_attr'].to(self.device)
            in_deg[0:node_num[idx]] = torch.clamp(g.in_degrees(), min=-1, max=self.max_degree).to(self.device)
            out_deg[0:node_num[idx]] = torch.clamp(g.out_degrees(), min=-1, max=self.max_degree).to(self.device)
            sp_dist[:node_num[idx], :node_num[idx]] = torch.clamp(g.ndata['sp_dist'][:, :node_num[idx]], min=-1, max=self.max_dist) + 1

            mask = torch.zeros(max_node_num, max_node_num)
            mask[node_num[idx]:, node_num[idx]:] = 1
            mask.fill_diagonal_(0)

            x.append(feat)
            in_degs.append(in_deg)
            out_degs.append(out_deg)
            mask_list.append(mask)
            sp_dist_list.append(sp_dist)
        
        x = torch.stack(x, dim=0).to(self.device)
        sp_dist_list = torch.stack(sp_dist_list, dim=0).to(self.device)
        mask_list = torch.stack(mask_list, dim=0)

        x = self.input_embedding(x)
        if self.embed_degree:
            in_degs = torch.stack(in_degs, dim=0).to(self.device)
            in_deg_emb = self.degree_embedding(in_degs)
            out_degs = torch.stack(out_degs, dim=0).to(self.device)
            out_deg_emb = self.degree_embedding(out_degs)
            x = x + in_deg_emb + out_deg_emb
        
        for i in range(self.num_layers):
            # time1 = time.time()
            attn_bias = self.spatial_encoders[i](sp_dist_list, mask_list, padding=float('-inf'))
            # print(attn_bias[:5, :, :, :])
            x = self.convs[i](x, attn_bias)
            if self.use_bn:
                x = self.bns[i](x.transpose(1, 2)).transpose(1, 2)
        readout_x = []
        for i in range(node_num.shape[0]):
            readout_x.append(x[i, node_num[i]-1, :])
        readout_x = torch.stack(readout_x, dim=0)
        readout_x = F.dropout(readout_x, p=0.5, training=self.training)
        readout_x = F.relu(self.fc1(readout_x))
        readout_x = F.dropout(readout_x, p=0.5, training=self.training)
        logit = self.fc2(readout_x)
        return F.log_softmax(logit, dim=-1)


def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(1, 1)
        self.bias = nn.Embedding(1, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x):
        mul = self.mul.weight[0, 0]
        bias = self.bias.weight[0, 0]
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


class FusedDistEncoder(nn.Module):
    def __init__(self, max_dist, num_hid, num_heads, device):
        super().__init__()
        self.max_dist = max_dist
        self.num_heads = num_heads
        self.device = device
        # deactivate node pair between which the distance is -1, VNODE is -2
        self.embedding_table = nn.Embedding(
            max_dist + 2, num_hid // 2, 
            # padding_idx=1
        )
        self.rd_embedding_layer = GaussianLayer(num_hid // 2)
        self.mlp = nn.Sequential(
            nn.Linear(num_hid, num_heads),
            nn.GELU(),
            nn.Linear(num_heads, num_heads)
        )

    def forward(self, sp_dist, rd_dist, mask=None, padding=float('-inf')):
        # max_num_nodes = max([dst.shape[0] for dst in sp_dist])
        # spatial_encoding = []

        # for i in range(len(sp_dist)):
        #     num_nodes = sp_dist[i].shape[0]
        #     # shape: [n, n, h], n = num_nodes, h = num_heads
        #     sp_dist_embedding = self.embedding_table(sp_dist[i])
        #     rd_dist_embedding = self.rd_embedding_layer(rd_dist[i])
        #     dist_embedding = self.mlp(torch.cat((sp_dist_embedding, rd_dist_embedding), dim=-1))
        #     # print(dist_embedding)
        #     # print(dist_embedding.shape)
        #     # [n, n, h] -> [N, N, h], N = max_num_nodes, padded with -inf
        #     padded_encoding = torch.full(
        #         (max_num_nodes, max_num_nodes, self.num_heads), padding
        #     ).to(self.device)
        #     for j in range(self.num_heads):
        #         padded_encoding[:, :, j].fill_diagonal_(0.)
        #     padded_encoding[0:num_nodes, 0:num_nodes] = dist_embedding
        #     spatial_encoding.append(padded_encoding)

        sp_dist_embedding = self.embedding_table(sp_dist)
        rd_dist_embedding = self.rd_embedding_layer(rd_dist)
        dist_embedding = self.mlp(torch.cat((sp_dist_embedding, rd_dist_embedding), dim=-1))
        if mask is not None:
            dist_embedding[mask.to(torch.bool)] = padding

        return dist_embedding


class GraphormerGD(torch.nn.Module):
    def __init__(self, num_features=1, num_classes=2, num_hidden=64, num_heads=4, max_dist=20, max_degree=20, num_layers=5, act=nn.ReLU(), embed_degree=False, bn=False, device=torch.cuda):
        super().__init__()

        self.dim = num_hidden
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.embed_degree = embed_degree
        self.max_degree = max_degree
        self.max_dist = max_dist

        self.convs = nn.ModuleList()
        self.use_bn = bn
        self.bns = nn.ModuleList()
        self.fused_encoders = nn.ModuleList()
        self.attn_mul_encoders = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(GraphormerLayer(feat_size=self.dim, hidden_size=self.dim, num_heads=self.num_heads, activation=act))
            if self.use_bn:
                self.bns.append(torch.nn.BatchNorm1d(self.dim))
            self.fused_encoders.append(FusedDistEncoder(max_dist=max_dist, num_hid=self.dim, num_heads=self.num_heads, device=self.device))
            self.attn_mul_encoders.append(FusedDistEncoder(max_dist=max_dist, num_hid=self.dim, num_heads=self.num_heads, device=self.device))

        self.input_embedding = nn.Sequential(
            Linear(self.num_features, self.dim),
        ) 
        self.degree_embedding = nn.Embedding(max_degree+2, self.dim, padding_idx=0)
        self.fc1 = Linear(self.dim, self.dim)
        self.fc2 = Linear(self.dim, num_classes)
    
    def forward(self, x, graph):
        g_list = dgl.unbatch(graph)
        node_num = graph.batch_num_nodes()
        max_node_num = torch.max(node_num)
        x = []
        # in_degs = []
        # out_degs = []
        sp_dist_list = []
        rd_dist_list = []
        mask_list = []
        mul_mask_list = []
        for idx, g in enumerate(g_list):
            feat = torch.zeros(max_node_num, self.num_features).to(self.device)
            # in_deg = torch.zeros(max_node_num, dtype=torch.long).to(self.device)
            # out_deg = torch.zeros(max_node_num, dtype=torch.long).to(self.device)
            feat[0:node_num[idx]] = g.ndata['node_attr'].to(self.device)
            # in_deg[0:node_num[idx]] = torch.clamp(g.in_degrees(), min=-1, max=self.max_degree).to(self.device)
            # out_deg[0:node_num[idx]] = torch.clamp(g.out_degrees(), min=-1, max=self.max_degree).to(self.device)
            x.append(feat)
            # in_degs.append(in_deg)
            # out_degs.append(out_deg)

            num_nodes = g.num_nodes()
            sp_dist = (
                torch.clamp(
                    # dgl.shortest_dist(g, root=None, return_paths=False),
                    g.ndata['sp_dist'][:, :num_nodes],
                    min=-1,
                    max=self.max_dist,
                )
                + 1
            )
            sp_dist = torch.zeros(max_node_num, max_node_num).long().to(self.device)
            rd_dist = torch.zeros(max_node_num, max_node_num).to(self.device)
            mask = torch.zeros(max_node_num, max_node_num)
            mul_mask = torch.zeros(max_node_num, max_node_num)
            mask[num_nodes:, num_nodes:] = 1
            mul_mask[num_nodes:, num_nodes:] = 1
            mask.fill_diagonal_(0)

            sp_dist[:num_nodes, :num_nodes] = torch.clamp(g.ndata['sp_dist'][:, :num_nodes], min=-1, max=self.max_dist) + 1
            rd_dist[:num_nodes, :num_nodes] = g.ndata['rd_dist'][:, :num_nodes]
            
            sp_dist_list.append(sp_dist)
            # rd_dist_list.append(g.ndata['rd_dist'][:, :num_nodes])
            rd_dist_list.append(rd_dist)
            mask_list.append(mask)
            mul_mask_list.append(mul_mask)
        
            # shape: [n, n, h], n = num_nodes, h = num_heads
            
        x = torch.stack(x, dim=0).to(self.device)
        sp_dist_list = torch.stack(sp_dist_list, dim=0).to(self.device)
        rd_dist_list = torch.stack(rd_dist_list, dim=0).to(self.device)
        mask_list = torch.stack(mask_list, dim=0)
        mul_mask_list = torch.stack(mul_mask_list, dim=0)
        torch.cuda.empty_cache()
        # print(x.shape)
        x = self.input_embedding(x)
        if self.embed_degree:
            in_degs = torch.stack(in_degs, dim=0).to(self.device)
            in_deg_emb = self.degree_embedding(in_degs)
            out_degs = torch.stack(out_degs, dim=0).to(self.device)
            out_deg_emb = self.degree_embedding(out_degs)
            x = x + in_deg_emb + out_deg_emb

        # print(x[:5, :,:])
        # print(graph.device)
        for i in range(self.num_layers):
            attn_bias = self.fused_encoders[i](sp_dist_list, rd_dist_list, mask_list, padding=float('-inf'))
            # print(attn_bias)
            attn_mul = self.attn_mul_encoders[i](sp_dist_list, rd_dist_list, mul_mask_list, padding=0.)
            # print(attn_mul)
            # print(attn_bias[:5, :, :, :])
            x = self.convs[i](x, attn_bias=attn_bias, attn_mul=attn_mul)
            if self.use_bn:
                x = self.bns[i](x.transpose(1, 2)).transpose(1, 2)
            torch.cuda.empty_cache()

        readout_x = []
        for i in range(node_num.shape[0]):
            readout_x.append(x[i, node_num[i]-1, :])
        readout_x = torch.stack(readout_x, dim=0)
        readout_x = F.dropout(readout_x, p=0.5, training=self.training)
        readout_x = F.relu(self.fc1(readout_x))
        readout_x = F.dropout(readout_x, p=0.5, training=self.training)
        logit = self.fc2(readout_x)
        return F.log_softmax(logit, dim=-1)