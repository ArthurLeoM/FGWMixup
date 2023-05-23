from time import time
import logging
import os
import os.path as osp
import numpy as np
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pickle


import dgl
from dgl.data import utils, TUDataset
from dgl.dataloading import GraphDataLoader, Sampler
from sklearn.model_selection import StratifiedKFold

import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm

import random
from copy import deepcopy


from utils_dgl import stat_graph, split_class_x_graphs
from gromov_mixup import FGWMixup
from models_dgl import GIN, GCN, Graphormer, GraphormerGD
import torch.nn as nn

import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')



def prepare_dataset_x(dataset):
    if dataset.name in ['ENZYMES', 'PROTEINS_full']:
        # all_attr = []
        for idx in range(len(dataset)):
            # all_attr.append(dataset[idx][0].ndata['node_attr'])
            dataset[idx][0].ndata['node_attr'] = torch.cat((F.one_hot(dataset[idx][0].ndata['node_labels'].squeeze(), num_classes=3), \
                dataset[idx][0].ndata['node_attr']), dim=-1).type(torch.FloatTensor)
        # all_attr = torch.cat(all_attr, dim=0)
        # mean_attr = torch.mean(all_attr, dim=0)
        # std_attr = torch.std(all_attr, dim=0)
        # for idx in range(len(dataset)):
        #     dataset[idx][0].ndata['node_attr'][:, 3:] = dataset[idx][0].ndata['node_attr'][:, 3:] - mean_attr / std_attr
            
    elif dataset.name == 'PROTEINS':
        for idx in range(len(dataset)):
            dataset[idx][0].ndata['node_attr'] = F.one_hot(dataset[idx][0].ndata['node_labels'].squeeze(), num_classes=3).squeeze().type(torch.FloatTensor)
    elif dataset.name == 'NCI1':
        for idx in range(len(dataset)):
            dataset[idx][0].ndata['node_attr'] = F.one_hot(dataset[idx][0].ndata['node_labels'].squeeze(), num_classes=37).squeeze().type(torch.FloatTensor)
    elif dataset.name == 'NCI109':
        for idx in range(len(dataset)):
            dataset[idx][0].ndata['node_attr'] = F.one_hot(dataset[idx][0].ndata['node_labels'].squeeze(), num_classes=38).squeeze().type(torch.FloatTensor)
    elif dataset.name == 'DD':
        for idx in range(len(dataset)):
            dataset[idx][0].ndata['node_attr'] = F.one_hot(dataset[idx][0].ndata['node_labels'].squeeze(), num_classes=89).squeeze().type(torch.FloatTensor)
    elif dataset.name == 'MUTAG':
        for idx in range(len(dataset)):
            dataset[idx][0].ndata['node_attr'] = F.one_hot(dataset[idx][0].ndata['node_labels'].squeeze(), num_classes=7).squeeze().type(torch.FloatTensor)

    else:
        if 'node_attr' not in dataset[0][0].ndata.keys():
            degs = []
            for idx in range(len(dataset)):
                degs.extend(dataset[idx][0].in_degrees().tolist())
            max_degree = max(degs)
            degs = torch.tensor(degs, dtype=torch.float32)

            if max_degree < 2000:
                # dataset.transform = T.OneHotDegree(max_degree)

                for idx in range(len(dataset)):
                    degrees = dataset[idx][0].in_degrees()
                    dataset[idx][0].ndata['node_attr'] = F.one_hot(degrees, num_classes=max_degree+1).to(torch.float)
            else:
                mean, std = degs.mean().item(), degs.std().item()
                for idx in range(len(dataset)):
                    degrees = dataset[idx][0].in_degrees()
                    dataset[idx][0].ndata['node_attr'] = ( (degrees - mean) / std ).view( -1, 1 )
        else:
            for idx in range(len(dataset)):
                dataset[idx][0].ndata['node_attr'] = dataset[idx][0].ndata['node_attr'].type(torch.FloatTensor)
    return dataset


def prepare_dataset_onehot_y(dataset):
    num_classes = dataset.num_classes
    if not isinstance(num_classes, int):
        num_classes = num_classes.item()
    dataset.graph_labels = F.one_hot(dataset.graph_labels, num_classes=num_classes).type(torch.FloatTensor).squeeze()
    return dataset


def mixup_cross_entropy_loss(input, target, size_average=True):

    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def removeIsolatedNodes(dataset):
    for idx in range(len(dataset)):
        isolated_nodes =((dataset[idx][0].in_degrees() == 0) & (dataset[idx][0].out_degrees() == 0)).nonzero().squeeze(1)
        dataset[idx][0].remove_nodes(isolated_nodes)



def train(model, train_loader):
    model.train()
    loss_all = 0
    graph_all = 0
    for data, label in train_loader:
        # print( "data.y", data.y )
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(data.ndata['node_attr'], data)
        batch_size = output.shape[0]
        label = label.view(-1, num_classes)
        # print(torch.exp(output), label)
        loss = mixup_cross_entropy_loss(output, label)
        loss.backward()
        # for name, params in model.named_parameters():
        #     # if 'input_embedding' in name:
        #     try:
        #         print('--name: ', name, 'grad_l2norm: ', torch.norm(params.grad))
        #     except:
        #         print('***ERROR*** name: ', name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        loss_all += loss.item() * batch_size
        graph_all += batch_size
        optimizer.step()
    loss = loss_all / graph_all
    return model, loss


def test(model, loader, ckpt=None):
    if ckpt is not None:
        model.load_state_dict(ckpt['weights'])
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data.ndata['node_attr'], data)
        batch_size = output.shape[0]
        pred = output.max(dim=1)[1]
        y = label.view(-1, num_classes)
        loss += mixup_cross_entropy_loss(output, y).item() * batch_size
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += batch_size
    acc = correct / total
    loss = loss / total
    return acc, loss


def plot_mixup_graph(graph_i, graph_j, mixup_graph, mixup_weights):
    plt.figure(figsize=(36,10))
    plt.subplot(1,3,1)
    G = nx.Graph()
    edge_index = np.nonzero(graph_i)
    plt_edge = np.array(edge_index).transpose()
    G.add_edges_from(plt_edge)
    nx.draw(G, node_size=5)
    plt.title('{:.5f}'.format(mixup_weights[0]))

    plt.subplot(1,3,2)
    G = nx.Graph()
    edge_index = np.nonzero(graph_j)
    plt_edge = np.array(edge_index).transpose()
    G.add_edges_from(plt_edge)
    nx.draw(G, node_size=5)
    plt.title('{:.5f}'.format(mixup_weights[1]))
    
    plt.subplot(1,3,3)
    G = nx.Graph()
    edge_index = np.nonzero(mixup_graph)
    plt_edge = np.array(edge_index).transpose()
    G.add_edges_from(plt_edge)
    nx.draw(G, node_size=5)
    plt.title('Mixup Graph')
    
    path = osp.join('./img/', args.dataset+'_'+args.metric)
    if not osp.exists(path):
        os.makedirs(path)
    plt.savefig(path+'/'+str(idx)+'_'+str(idx_i)+'_'+str(idx_j)+'.png')

        
def mySampler(dataset):
    for idx in range(len(dataset)):
        if dataset[idx][0].num_nodes() > 1200:
            indices = random.sample(range(dataset[idx][0].num_nodes()), 1200)
            dataset.graph_lists[idx] = dataset[idx][0].subgraph(indices)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    parser.add_argument('--backbone', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gmixup', type=str, default="False")
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--pooling', type=str, default='mean', choices=['max', 'mean', 'sum'])
    parser.add_argument('--agg', type=str, default='mean', choices=['max', 'mean', 'sum'])
    parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('--aug_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--measure', type=str, default='degree', choices=['degree', 'uniform'], help='FGW node measurement')
    parser.add_argument('--metric', type=str, default='sp', choices=['sp', 'adj'], help='FGW Metric: Shortest path or Adjacency Matrix')
    parser.add_argument('--alpha', type=float, default=0.95, help='FGW alpha')
    parser.add_argument('--beta_k', type=float, default=0.2, help='Mixup Weight ~ Beta(k,k)')
    parser.add_argument('--log_screen', type=str, default="False")
    parser.add_argument('--symmetry', action='store_true', default=False, help='force to be symmetric matrix')
    parser.add_argument('--loss_fun', type=str, default="square_loss", help='loss function used in structral distance calcution of FGW')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--kfold', default=False, action='store_true')
    parser.add_argument('--bapg', default=False, action='store_true')
    parser.add_argument('--act', type=str, default='relu')
    parser.add_argument('--bn', default=False, action='store_true', help='Only for Graphormer(GD)')
    parser.add_argument('--early_stopping', type=int, default=50)
    parser.add_argument('--rho', type=float, default=0.1)

    args = parser.parse_args()

    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    gmixup = eval(args.gmixup)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    aug_ratio = args.aug_ratio
    backbone = args.backbone

    if args.act == 'relu':
        act = nn.ReLU()
    elif args.act == 'gelu':
        act = nn.GELU()
    elif args.act == 'elu':
        act = nn.ELU()
    else:
        act = nn.ReLU()

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    
    np.random.seed(seed) #numpy
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    torch.backends.cudnn.deterministic=True # cudnn

    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logger.info(f"runing device: {device}")

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(name=dataset_name, raw_dir=path)
    # print(dataset[0][0])
    mySampler(dataset)
    removeIsolatedNodes(dataset)
    dataset = prepare_dataset_onehot_y(dataset)
    # print(dataset[0][0].ndata)
    dataset = prepare_dataset_x(dataset)

    # print(dataset[0][0].edata)

    if not args.kfold:
        logger.info("Please apply 10-fold CV for the dataset with --kfold argument")
        
    else:
        # nontest_dataset, test_dataset = tuple(dgl.data.utils.split_dataset(dataset, frac_list=[0.9, 0.1], shuffle=True, random_state=seed))
        kfold_y = [torch.argmax(dataset[idx][1]).item() for idx in range(len(dataset))]
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        all_best_acc_list = []
        all_best_loss_list = []

        for fold_cnt, (nontest_idx, test_idx) in enumerate(kfold.split(dataset, kfold_y)):
            # if fold_cnt != 6:
            #     continue
            dataset = TUDataset(name=dataset_name, raw_dir=path)
            mySampler(dataset)
            removeIsolatedNodes(dataset)
            dataset = prepare_dataset_onehot_y(dataset)
            # print(dataset[0][0].ndata)
            dataset = prepare_dataset_x(dataset)

            train_idx = np.random.choice(nontest_idx, size=int(0.9*len(nontest_idx)), replace=False)
            val_idx = []
            for idx in nontest_idx:
                if idx not in train_idx:
                    val_idx.append(idx)
            train_dataset = dgl.data.utils.Subset(dataset, train_idx)
            val_dataset = dgl.data.utils.Subset(dataset, val_idx)
            test_dataset = dgl.data.utils.Subset(dataset, test_idx)
            # print(train_dataset.indices)
            # print(len(train_dataset), train_dataset)

            # resolution = int(median_num_nodes)
            new_graph = {'graph_list': [], 'label_list': []}
            aug_indices = []
            num_all_data = len(dataset)

            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, max_num_nodes = stat_graph(dataset)
            # logger.info(f"avg num nodes of graphs: { avg_num_nodes }")
            # logger.info(f"avg num edges of graphs: { avg_num_edges }") 
            # logger.info(f"avg density of graphs: { avg_density }")
            # logger.info(f"median num nodes of graphs: { median_num_nodes }")
            # logger.info(f"median num edges of graphs: { median_num_edges }")
            # logger.info(f"maximum num nodes of graphs: { max_num_nodes }")
            # logger.info(f"median density of graphs: { median_density }")
            # logger.info(f"Number of graph samples: { len(dataset) }")

            if gmixup == True:
                if args.bapg:
                    if args.rank is not None:
                        mixup_path = path+'/lowrank_mixup_fold'+str(fold_cnt)+'_k'+str(args.beta_k)+'_FGWalpha'+str(args.alpha)+'.dat'
                    else:
                        mixup_path = path+'/tttbapg_new_fgw_mixup_fold'+str(fold_cnt)+'_k'+str(args.beta_k)+'_FGWalpha'+str(args.alpha)+'.dat'
                else:
                    mixup_path = path+'/ttnormal_nearest_dense_mixup_fold'+str(fold_cnt)+'_k'+str(args.beta_k)+'_FGWalpha'+str(args.alpha)+'.dat'
                # mixup_path = path+'/mixup_fold'+str(fold_cnt)+'.dat'
                if os.path.exists(mixup_path):
                    logger.info(f'Loading Mixup Graphs from Files: {mixup_path}')
                    new_graph = pickle.load(open(mixup_path, 'rb'))
                    new_len = int((args.aug_ratio / 0.25) * len(new_graph['graph_list']))
                    dataset.graph_lists.extend(new_graph['graph_list'][:new_len])
                    dataset.graph_labels = torch.cat((dataset.graph_labels, new_graph['label_list'][:new_len]), dim=0)
                    aug_indices = list(range(num_all_data, num_all_data+new_len))
                    # print(aug_indices)
                    # print(len(dataset.graph_lists), len(dataset))
                
                else:
                    logger.info(f'Generating Mixup Graphs')
                    class_graphs = split_class_x_graphs(train_dataset)

                    num_sample = int( len(train_dataset) * aug_ratio )
                    num_class = len(class_graphs)
                    iter_list = []
                    time_list = []
                    if num_class < 20:
                        mixup_types = (num_class - 1) * num_class / 2
                        type_num_sample = max(1, int( num_sample / mixup_types ))
                        for i in range(num_class):
                            for j in range(i+1, num_class):
                                logger.info(f"Mixup-1 label: {class_graphs[i][0]}, num_graphs:{len(class_graphs[i][1])}" )
                                logger.info(f"Mixup-2 label: {class_graphs[j][0]}, num_graphs:{len(class_graphs[j][1])}" )
                                logger.info(f"num_mixtype_sample: {type_num_sample}")
                                idx = 0
                                while idx < type_num_sample:
                                    idx_i = np.random.randint(len(class_graphs[i][1]))
                                    idx_j = np.random.randint(len(class_graphs[j][1]))
                                    graph_i = class_graphs[i][1][idx_i]
                                    graph_j = class_graphs[j][1][idx_j]

                                    graphs = [graph_i, graph_j]
                                    # print(graph_i, graph_j)
                                    labels = [class_graphs[i][0], class_graphs[j][0]]
                                    
                                    features = [class_graphs[i][2][idx_i], class_graphs[j][2][idx_j]]
                                    
                                    mixup_graph, mixup_label, mixup_feature, mixup_weights, n_iter, time = FGWMixup(graphs, labels, features, nodes=int(median_num_nodes), measure=args.measure, \
                                                                                                                metric=args.metric, alpha=args.alpha, k=args.beta_k, rank=args.rank, bapg=args.bapg, rho=args.rho)
                                    iter_list.append(n_iter)
                                    time_list.append(time)
                                    if True in np.isnan(mixup_graph):
                                        continue
                                    # logger.info(f"Graph_adj_raw: {mixup_graph}" )

                                    if args.symmetry:
                                        mixup_graph = np.triu(mixup_graph)
                                        mixup_graph = mixup_graph + mixup_graph.T - np.diag(np.diag(mixup_graph))
                                    
                                    mixup_feature = mixup_feature[mixup_graph.sum(axis=0) != 0]
                                    mixup_feature = torch.from_numpy(mixup_feature).type(torch.FloatTensor)

                                    mixup_label = torch.from_numpy(mixup_label).type(torch.FloatTensor)

                                    mixup_graph = mixup_graph[mixup_graph.sum(axis=1) != 0]
                                    mixup_graph = mixup_graph[:, mixup_graph.sum(axis=0) != 0]

                                    A = torch.from_numpy(mixup_graph)
                                    if A.size(0) == 0:
                                        logger.info(f"Non-graph Generated!")
                                        continue
                                    elif torch.sum(A) > 0.9 * A.size(0) * A.size(0):
                                        ## Over-dense, not good
                                        logger.info(f"Too dense graph generated!")
                                        continue
                                    
                                    # plot_mixup_graph(graph_i, graph_j, mixup_graph, [1-mixup_weights, mixup_weights])

                                    idx += 1
                                    src, dst = torch.nonzero(A, as_tuple=True)
                                    dgl_graph = dgl.graph((src, dst))
                                    dgl_graph.ndata['node_attr'] = mixup_feature
                                    dgl_graph.ndata['_ID'] = torch.arange(0, dgl_graph.num_nodes(), 1).to(torch.int64)
                                    if 'node_labels' in dataset[0][0].ndata.keys():
                                        dgl_graph.ndata['node_labels'] = torch.bernoulli(torch.empty(dgl_graph.num_nodes(), 1).uniform_(0, 1)).to(torch.int64)
                                    dgl_graph.edata['_ID'] = torch.arange(0, dgl_graph.num_edges(), 1).to(torch.int64)
                                    if 'edge_labels' in dataset[0][0].edata.keys():
                                        dgl_graph.edata['edge_labels'] = torch.bernoulli(torch.empty(dgl_graph.num_edges(), 1).uniform_(0, 1)).to(torch.int64)
                                    dgl_label = mixup_label

                                    dataset.graph_lists.append(dgl_graph)
                                    new_graph['graph_list'].append(dgl_graph)
                                    dataset.graph_labels = torch.cat((dataset.graph_labels, torch.tensor(dgl_label).to(torch.float).unsqueeze(0)), dim=0)
                                    new_graph['label_list'].append(torch.tensor(dgl_label).to(torch.float))
                                    aug_indices.append(num_all_data)
                                    num_all_data += 1
                                    logger.info(f"No. {idx}, mixup_ids: {idx_i, idx_j}, graph Generated! Edges: {dgl_graph.num_edges()} Nodes: {dgl_graph.num_nodes()}")
                                    if idx % 20 == 0:
                                        logger.info(f"No.: {idx}, label: {dgl_label}")
                                        print(mixup_graph.shape,flush=True)
                                        print(mixup_graph,flush=True)
                                        print((src, dst),flush=True)
                                    
                    else:
                        idx = 0
                        while idx < num_sample:
                            idx_i = np.random.randint(len(train_dataset))
                            idx_j = np.random.randint(len(train_dataset))
                            graph_i = train_dataset[idx_i][0].adj().to_dense().numpy()
                            graph_j = train_dataset[idx_j][0].adj().to_dense().numpy()

                            graphs = [graph_i, graph_j]
                            # print(graph_i, graph_j)
                            labels = [np.array(train_dataset[idx_i][1]), np.array(train_dataset[idx_j][1])]
                            logger.info(f"Mixup-1 label: {labels[0]}" )
                            logger.info(f"Mixup-2 label: {labels[1]}" )
                            
                            features = [train_dataset[idx_i][0].ndata['node_attr'].numpy(), train_dataset[idx_j][0].ndata['node_attr'].numpy()]
                            
                            mixup_graph, mixup_label, mixup_feature, mixup_weights = FGWMixup(graphs, labels, features, measure=args.measure, metric=args.metric, alpha=args.alpha, k=args.beta_k)
                            # logger.info(f"Graph_adj_raw: {mixup_graph}" )

                            if args.symmetry:
                                mixup_graph = np.triu(mixup_graph)
                                mixup_graph = mixup_graph + mixup_graph.T - np.diag(np.diag(mixup_graph))
                            
                            mixup_feature = mixup_feature[mixup_graph.sum(axis=0) != 0]
                            mixup_feature = torch.from_numpy(mixup_feature).type(torch.FloatTensor)

                            mixup_label = torch.from_numpy(mixup_label).type(torch.FloatTensor)

                            mixup_graph = mixup_graph[mixup_graph.sum(axis=1) != 0]
                            mixup_graph = mixup_graph[:, mixup_graph.sum(axis=0) != 0]

                            A = torch.from_numpy(mixup_graph)
                            if A.size(0) == 0:
                                logger.info(f"Non-graph Generated!")
                                continue
                            elif torch.sum(A) > 0.95 * A.size(0) * A.size(0):
                                ## Over-dense, not good
                                logger.info(f"Too dense graph generated!")
                                continue
                            
                            plot_mixup_graph(graph_i, graph_j, mixup_graph, mixup_weights)

                            idx += 1
                            src, dst = torch.nonzero(A, as_tuple=True)
                            dgl_graph = dgl.graph((src, dst))
                            dgl_graph.ndata['node_attr'] = mixup_feature
                            dgl_graph.ndata['_ID'] = torch.arange(0, dgl_graph.num_nodes(), 1).to(torch.int64)
                            if 'node_labels' in dataset[0][0].ndata.keys():
                                dgl_graph.ndata['node_labels'] = torch.bernoulli(torch.empty(dgl_graph.num_nodes(), 1).uniform_(0, 1)).to(torch.int64)
                            dgl_graph.edata['_ID'] = torch.arange(0, dgl_graph.num_edges(), 1).to(torch.int64)
                            if 'edge_labels' in dataset[0][0].edata.keys():
                                dgl_graph.edata['edge_labels'] = torch.bernoulli(torch.empty(dgl_graph.num_edges(), 1).uniform_(0, 1)).to(torch.int64)
                            dgl_label = mixup_label

                            dataset.graph_lists.append(dgl_graph)
                            new_graph['graph_list'].append(dgl_graph)
                            dataset.graph_labels = torch.cat((dataset.graph_labels, torch.tensor(dgl_label).to(torch.float).unsqueeze(0)), dim=0)
                            new_graph['label_list'].append(torch.tensor(dgl_label).to(torch.float))
                            aug_indices.append(num_all_data)
                            num_all_data += 1
                            logger.info(f"No. {idx}, graph Generated! Edges: {dgl_graph.num_edges()} Nodes: {dgl_graph.num_nodes()}")
                            if idx % 20 == 0:
                                logger.info(f"No.: {idx}, label: {dgl_label}")
                                print(mixup_graph.shape,flush=True)
                                print(mixup_graph,flush=True)
                                print((src, dst),flush=True)

                    new_graph['label_list'] = torch.stack(new_graph['label_list'], dim=0)
                    logger.info(f"Avg FGW Converge Iterations: { np.mean(iter_list) }")
                    logger.info(f"FGW Converge Time Total: { np.sum(time_list) }")
                    logger.info(f"FGW Dist Average Calc Time : { np.sum(time_list) / np.sum(iter_list) }")
                    pickle.dump(new_graph, open(mixup_path, 'wb'))

                avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, max_num_nodes = stat_graph(dgl.data.utils.Subset(dataset, aug_indices))
                logger.info(f"avg num nodes of new graphs: { avg_num_nodes }")
                logger.info(f"avg num edges of new graphs: { avg_num_edges }")
                logger.info(f"avg density of new graphs: { avg_density }")
                logger.info(f"median num nodes of new graphs: { median_num_nodes }")
                logger.info(f"median num edges of new graphs: { median_num_edges }")
                logger.info(f"maximum num nodes of new graphs: { max_num_nodes }")
                logger.info(f"median density of new graphs: { median_density }")
            

            avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density, max_num_nodes = stat_graph(dataset)
            logger.info(f"avg num nodes of graphs: { avg_num_nodes }")
            logger.info(f"avg num edges of graphs: { avg_num_edges }")
            logger.info(f"avg density of graphs: { avg_density }")
            logger.info(f"median num nodes of graphs: { median_num_nodes }")
            logger.info(f"median num edges of graphs: { median_num_edges }")
            logger.info(f"maximum num nodes of graphs: { max_num_nodes }")
            logger.info(f"median density of graphs: { median_density }")
            logger.info(f"Number of graph samples: { len(dataset) }")


            num_features = dataset[0][0].ndata['node_attr'].shape[1]
            num_classes = dataset.num_classes
            if not isinstance(num_classes, int):
                num_classes = num_classes.item()
            
            if backbone == 'GCN':
                for idx in range(len(dataset)):
                    dataset[idx][0].add_self_loop()
            
            if backbone in ['GCN', 'GIN']:
                num_features = dataset[0][0].ndata['node_attr'].shape[1]
                for idx in tqdm(range(len(dataset))):
                    # print(dataset[idx][0].num_nodes(), dataset[idx][0].num_edges())
                    dataset[idx][0].add_nodes(1)
                    vnode_idx = dataset[idx][0].num_nodes() - 1
                    for i in range(vnode_idx):
                        dataset[idx][0].add_edges(i, vnode_idx)
                        dataset[idx][0].add_edges(vnode_idx, i)
                    # print(dataset[idx][0].num_nodes(), dataset[idx][0].num_edges())
                    dataset[idx][0].ndata['node_attr'][vnode_idx] = torch.zeros(num_features)
            
            if backbone in ['Graphormer', 'GraphormerGD']:
                num_features = dataset[0][0].ndata['node_attr'].shape[1]
                for idx in tqdm(range(len(dataset))):
                    # print(dataset[idx][0].num_nodes(), dataset[idx][0].num_edges())
                    dataset[idx][0].add_nodes(1)
                    vnode_idx = dataset[idx][0].num_nodes() - 1
                    # for i in range(vnode_idx):
                    #     dataset[idx][0].add_edges(i, vnode_idx)
                    #     dataset[idx][0].add_edges(vnode_idx, i)
                    # print(dataset[idx][0].num_nodes(), dataset[idx][0].num_edges())
                    dataset[idx][0].ndata['node_attr'][vnode_idx] = torch.zeros(num_features)

                    num_nodes = dataset[idx][0].num_nodes()
                    dataset[idx][0].ndata['sp_dist'] = -1 * torch.ones(num_nodes, max_num_nodes+1, dtype=torch.long)
                    dataset[idx][0].ndata['sp_dist'][:, :num_nodes] = dgl.shortest_dist(dataset[idx][0], root=None, return_paths=False)
                    # dataset[idx][0].ndata['sp_dist'][:, vnode_idx] = -2
                    # dataset[idx][0].ndata['sp_dist'][vnode_idx, :] = -2  # Special Spatial Encoding for VNODE

                    if backbone == 'GraphormerGD':
                        rd_dist = torch.zeros(num_nodes, max_num_nodes+1)
                        rd_dist[num_nodes-1, num_nodes-1] = 1.
                        idx_all = list(range(num_nodes-1))
                        connected_comp = []
                        while len(idx_all) > 0:
                            cur_idx = idx_all[0]
                            conn_idx = []
                            for i in idx_all:
                                if dataset[idx][0].ndata['sp_dist'][cur_idx, i] > -1:
                                    conn_idx.append(i)
                            connected_comp.append(conn_idx)
                            for i in conn_idx:
                                idx_all.remove(i)
                        # print(connected_comp)
                    
                        laplacian = torch.diag(dataset[idx][0].in_degrees()) - dataset[idx][0].adj().to_dense()

                        for idxs in connected_comp:
                            num_subg_nodes = len(idxs)
                            subg_laplacian = laplacian[idxs, :][:, idxs]
                            subg_laplacian = subg_laplacian + torch.ones((num_subg_nodes, num_subg_nodes), dtype=torch.float) / num_subg_nodes
                            try:
                                M_mat = torch.inverse(subg_laplacian)
                            except:
                                print(subg_laplacian)
                                print(torch.diag(subg_laplacian).cpu().numpy().tolist())
                                print(torch.det(subg_laplacian))
                        
                            for i, idx_i in enumerate(idxs):
                                for j, idx_j in enumerate(idxs):
                                    rd_dist[idx_i, idx_j] = M_mat[i, i] + M_mat[j, j] - 2*M_mat[i, j] + 1.
                            dataset[idx][0].ndata['rd_dist'] = rd_dist


            logger.info(f"num_features: {num_features}" )
            # max_label = 0
            # for i in range(len(dataset)):
            #     max_label = max(max_label, torch.max(dataset[i][0].ndata['node_labels']).item())
            
            # logger.info(f"num_node_labels: {max_label}" )
            logger.info(f"num_classes: {num_classes}"  )
            
            # dataset = prepare_dataset_x( dataset 

            if gmixup == True:
                train_dataset.indices = np.append(train_dataset.indices, np.array(aug_indices))

            logger.info(f"train_dataset size: {len(train_dataset)}")
            logger.info(f"val_dataset size: {len(val_dataset)}")
            logger.info(f"test_dataset size: {len(test_dataset)}" )


            train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


            if backbone == "GIN":
                model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden, num_layers=args.num_layers, agg=args.agg, act=act, pooling=args.pooling).to(device)
            elif backbone == "GCN":
                model = GCN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden, num_layers=args.num_layers, act=act, pooling=args.pooling).to(device)
            elif backbone == "Graphormer":
                model = Graphormer(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden, num_heads=4, max_dist=500, max_degree=100, act=act,
                                   embed_degree=False, num_layers=args.num_layers, bn=args.bn, device=device).to(device)
            elif backbone == "GraphormerGD":
                model = GraphormerGD(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden, num_heads=4, max_dist=500, max_degree=100, act=act,
                                   embed_degree=False, num_layers=args.num_layers, bn=args.bn, device=device).to(device)
            else:
                logger.info(f"No model."  )

            # for name, param in model.named_parameters():
            #     print(name, param.data)

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
            # if dataset_name == 'ENZYMES':
            #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 400, 800], gamma=0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            # if backbone == "Graphormer":
            #     scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_epochs)
            best_acc = 0.
            best_loss = 999999
            acc_best_ckpt = {}
            loss_best_ckpt = {}
            escape_epoch = 0

            for epoch in range(1, num_epochs):
                model, train_loss = train(model, train_loader)
                train_acc = 0
                val_acc, val_loss = test(model, val_loader, ckpt=None)
                logger.info('Fold {:02d} Epoch: {:03d}, Train Loss: {:.6f}, Val Loss: {:.6f}, Val Acc: {: .6f}'.format(
                    fold_cnt, epoch, train_loss, val_loss, val_acc))
                escape_epoch += 1
                if val_acc > best_acc:
                    best_acc = val_acc
                    escape_epoch = 0
                    logger.info("Fold Best Acc")
                    acc_best_ckpt = {
                        'acc': val_acc,
                        'weights': deepcopy(model.state_dict()),
                        'epoch': epoch,
                        'fold': fold_cnt
                    }
                    torch.save(acc_best_ckpt, './ckpt/'+args.dataset+'_acc_best_'+args.metric+str(args.seed)+'_fold'+str(fold_cnt)+args.gmixup+'.pt')
                    logger.info("Saving Best Acc Model")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    escape_epoch = 0
                    loss_best_ckpt = {
                        'loss': val_loss,
                        'weights': deepcopy(model.state_dict()),
                        'epoch': epoch,
                        'fold': fold_cnt
                    }
                    torch.save(loss_best_ckpt, './ckpt/'+args.dataset+'_loss_best_'+args.metric+str(args.seed)+'_fold'+str(fold_cnt)+args.gmixup+'.pt')
                    logger.info("Saving Best Loss Model")
                
                scheduler.step()
                if escape_epoch >= args.early_stopping and epoch >= 100:
                    logger.info('Early Stop!')
                    break
            

            with open('./ckpt/'+args.dataset+'test_'+args.metric+'.txt', 'a') as f:
                f.write("args:{}\n".format(args))
                test_acc, test_loss = test(model, test_loader, ckpt=acc_best_ckpt)
                logger.info('-- Fold {:02d} Best Acc Model @ Epoch: {:03d} -- Test Loss: {:.6f}, Test Acc: {: .6f}\n'.format(
                        fold_cnt, acc_best_ckpt['epoch'], test_loss, test_acc))
                f.write('-- Fold {:02d} Best Acc Model @ Epoch: {:03d} -- Test Loss: {:.6f}, Test Acc: {: .6f}\n'.format(
                        fold_cnt, acc_best_ckpt['epoch'], test_loss, test_acc))
                all_best_acc_list.append(test_acc)
                test_acc, test_loss = test(model, test_loader, ckpt=loss_best_ckpt)
                logger.info('-- Fold {:02d} Best Loss Model @ Epoch: {:03d} -- Test Loss: {:.6f}, Test Acc: {: .6f}\n'.format(
                        fold_cnt, loss_best_ckpt['epoch'], test_loss, test_acc))
                f.write('-- Fold {:02d} Best Loss Model @ Epoch: {:03d} -- Test Loss: {:.6f}, Test Acc: {: .6f}\n'.format(
                        fold_cnt, loss_best_ckpt['epoch'], test_loss, test_acc))
                all_best_loss_list.append(test_acc)
            
            logger.info('Fold {:02d} Complete'.format(fold_cnt))
            
        
        with open('./ckpt/'+args.dataset+'test_'+args.metric+'.txt', 'a') as f:
            f.write("args:{}\n".format(args))
            best_acc_mean = np.mean(all_best_acc_list)
            best_acc_std = np.std(all_best_acc_list)
            best_loss_mean = np.mean(all_best_loss_list)
            best_loss_std = np.std(all_best_loss_list)
            logger.info('Best Acc Model: {:.4f}({:.4f})\n'.format(best_acc_mean, best_acc_std))
            f.write('Best Acc Model: {:.4f}({:.4f})\n'.format(best_acc_mean, best_acc_std))
            logger.info('Best Loss Model: {:.4f}({:.4f})\n'.format(best_loss_mean, best_loss_std))
            f.write('Best Loss Model: {:.4f}({:.4f})\n'.format(best_loss_mean, best_loss_std))