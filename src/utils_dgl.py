from typing import List, Tuple
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import copy

from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
import torch
import random

import dgl



def graph_numpy2tensor(graphs: List[np.ndarray]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()



def align_x_graphs(graphs: List[np.ndarray], node_x: List[np.ndarray], padding: bool = False, N: int = None) -> Tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """
    Align multiple graphs by sorting their nodes by descending node degrees

    :param graphs: a list of binary adjacency matrices
    :param padding: whether padding graphs to the same size or not
    :return:
        aligned_graphs: a list of aligned adjacency matrices
        normalized_node_degrees: a list of sorted normalized node degrees (as node distributions)
    """
    num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
    max_num = max(num_nodes)
    min_num = min(num_nodes)
    max_num = max(max_num, N)
    aligned_node_x = np.zeros((max_num, node_x[0].shape[1]))
    cnt = np.zeros(max_num)

    aligned_graphs = []
    normalized_node_degrees = []
    for i in range(len(graphs)):
        num_i = graphs[i].shape[0]

        node_degree = 0.5 * np.sum(graphs[i], axis=0) + 0.5 * np.sum(graphs[i], axis=1)
        node_degree /= np.sum(node_degree)
        idx = np.argsort(node_degree)  # ascending
        idx = idx[::-1]  # descending
        # print(idx)

        sorted_node_degree = node_degree[idx]
        sorted_node_degree = sorted_node_degree.reshape(-1, 1)

        sorted_graph = copy.deepcopy(graphs[i])
        sorted_graph = sorted_graph[idx, :]
        sorted_graph = sorted_graph[:, idx]

        new_node_x = copy.deepcopy( node_x[i] )
        sorted_node_x = new_node_x[ idx, :]

        # if max_num < N:
        #     max_num = max(max_num, N)
        if padding:
            # normalized_node_degree = np.ones((max_num, 1)) / max_num
            normalized_node_degree = np.zeros((max_num, 1))
            normalized_node_degree[:num_i, :] = sorted_node_degree

            aligned_graph = np.zeros((max_num, max_num))
            aligned_graph[:num_i, :num_i] = sorted_graph

            normalized_node_degrees.append(normalized_node_degree)
            aligned_graphs.append(aligned_graph)

            # added
            aligned_node_x[:num_i, :] += sorted_node_x
            cnt[:num_i] += 1


        else:
            normalized_node_degrees.append(sorted_node_degree)
            aligned_graphs.append(sorted_graph)

        if N:
            aligned_graphs = [aligned_graph[:N, :N] for aligned_graph in aligned_graphs]
            normalized_node_degrees = normalized_node_degrees[:N]

            #added
    if N:
        aligned_node_x = aligned_node_x[:N]
        cnt = cnt[:N].reshape(-1, 1)
    aligned_node_x = aligned_node_x / cnt

    return aligned_graphs, aligned_node_x, normalized_node_degrees, max_num, min_num




def two_x_graphons_mixup(two_x_graphons, dataset, la=0.5, num_sample=20):

    label = la * two_x_graphons[0][0] + (1 - la) * two_x_graphons[1][0]
    new_graphon = la * two_x_graphons[0][1] + (1 - la) * two_x_graphons[1][1]
    new_x = la * two_x_graphons[0][2] + (1 - la) * two_x_graphons[1][2]

    sample_graph_label = torch.from_numpy(label).type(torch.FloatTensor)
    sample_graph_x = torch.from_numpy(new_x).type(torch.FloatTensor)
    # print(new_graphon)

    sample_graphs = []
    graph_labels = None
    for i in range(num_sample):

        sample_graph = (np.random.rand(*new_graphon.shape) <= new_graphon).astype(np.int32)
        sample_graph = np.triu(sample_graph)
        sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

        sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
        sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

        A = torch.from_numpy(sample_graph)
        edge_index, _ = dense_to_sparse(A)

        num_nodes = int(torch.max(edge_index)) + 1

        src, dst = torch.nonzero(A, as_tuple=True)
        dgl_graph = dgl.graph((src, dst))
        idx = []
        for i in range(sample_graph_x.shape[0]):
            if i in src or i in dst:
                idx.append(i)
        dgl_graph.ndata['node_attr'] = sample_graph_x[idx]
        dgl_graph.ndata['_ID'] = torch.arange(0, dgl_graph.num_nodes(), 1).to(torch.int64)
        if 'node_labels' in dataset[0][0].ndata.keys():
            dgl_graph.ndata['node_labels'] = torch.bernoulli(torch.empty(dgl_graph.num_nodes(), 1).uniform_(0, 1)).to(torch.int64)
        dgl_graph.edata['_ID'] = torch.arange(0, dgl_graph.num_edges(), 1).to(torch.int64)
        if 'edge_labels' in dataset[0][0].edata.keys():
            dgl_graph.edata['edge_labels'] = torch.bernoulli(torch.empty(dgl_graph.num_edges(), 1).uniform_(0, 1)).to(torch.int64)
        dgl_label = sample_graph_label

        sample_graphs.append(dgl_graph)
        if graph_labels is None:
            graph_labels = dgl_label.unsqueeze(0)
        else:
            graph_labels = torch.cat((graph_labels, dgl_label.unsqueeze(0)), dim=0)
        
        # print(edge_index)
    return sample_graphs, graph_labels




def split_class_x_graphs(dataset):

    y_list = []
    for idx in range(len(dataset)):
        y_list.append(tuple(dataset[idx][1].tolist()))
    num_classes = len(set(y_list))

    all_graphs_list = []
    all_node_x_list = []
    for idx in range(len(dataset)):
        adj = dataset[idx][0].adj().to_dense().numpy()
        all_graphs_list.append(adj)
        all_node_x_list.append(dataset[idx][0].ndata['node_attr'].numpy())
    # print(len(all_node_x_list), all_node_x_list[0])
    class_graphs = []
    for class_label in set(y_list):
        c_graph_list = [all_graphs_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        c_node_x_list = [all_node_x_list[i] for i in range(len(y_list)) if y_list[i] == class_label]
        class_graphs.append( ( np.array(class_label), c_graph_list, c_node_x_list ) )

    return class_graphs



def universal_svd(aligned_graphs: List[np.ndarray], threshold: float = 2.02) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.

    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.

    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs).to( "cuda" )
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0)
    else:
        sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    u, s, v = torch.svd(sum_graph)
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.cpu().numpy()
    torch.cuda.empty_cache()
    return graphon



def stat_graph(dataset):
    num_total_nodes = []
    num_total_edges = []
    for idx in range(len(dataset)):
        num_total_nodes.append(dataset[idx][0].num_nodes())
        # if dataset[idx][0].num_nodes() >= 1000:
        #     print(dataset[idx][0].num_nodes())
        num_total_edges.append(dataset[idx][0].num_edges())
        # if dataset[idx][0].in_degrees().min().item() == 0 and dataset[idx][0].out_degrees().min().item() == 0:
        #     print('0-degree vertice exists in No.', idx)
    avg_num_nodes = sum( num_total_nodes ) / len(dataset)
    avg_num_edges = sum( num_total_edges ) / len(dataset) / 2.0
    avg_density = avg_num_edges / (avg_num_nodes * avg_num_nodes)

    median_num_nodes = np.median( num_total_nodes ) 
    median_num_edges = np.median(num_total_edges)
    max_num_nodes = np.max(num_total_nodes)
    median_density = median_num_edges / (median_num_nodes * median_num_nodes)

    return avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, max_num_nodes
