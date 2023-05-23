import copy
import cv2
import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
# from ot.gromov import fgw_barycenters, gromov_barycenters, fused_gromov_wasserstein
from FGW_barycenter import my_fgw_barycenters


def find_thresh(C, ori_C_list, mixup_label, inf=0.5, sup=3, step=10, metric='sp'):
    """ Trick to find the adequate thresholds from where value of the C matrix are considered close enough to say that nodes are connected
        Tthe threshold is found by a linesearch between values "inf" and "sup" with "step" thresholds tested.
        The optimal threshold is the one which minimizes the reconstruction error between the shortest_path matrix coming from the thresholded adjency matrix
        and the original matrix.
    Parameters
    ----------
    C : ndarray, shape (n_nodes,n_nodes)
            The structure matrix to threshold
    inf : float
          The beginning of the linesearch
    sup : float
          The end of the linesearch
    step : integer
            Number of thresholds tested
    metric : 'adj' or 'sp'
        Use Shortest Path of Adjacency Matrix as the distance metric
    """
    dist = []
    search = np.linspace(inf, sup, step)
    for thresh in search:
        if metric == 'sp':
            Cprime = sp_to_adjency(C, 0.2, thresh, metric=metric)
            # print(Cprime)
            SC = shortest_path(Cprime, method='D')
            SC[SC == float('inf')] = 100
            dist.append(np.linalg.norm(SC - C))
        elif metric == 'adj':
            SC = sp_to_adjency(C, 0, thresh, metric=metric)
            new_dens = np.sum(SC) / (SC.shape[0]**2)
            ori_dens = mixup_label[0] * (np.sum(ori_C_list[0]) / (ori_C_list[0].shape[0]**2)) + mixup_label[1] * (np.sum(ori_C_list[1]) / (ori_C_list[1].shape[0]**2))
            dist.append(np.abs(new_dens - ori_dens))
    return search[np.argmin(dist)], dist


def sp_to_adjency(C, threshinf=0.2, threshsup=1.8, metric='sp'):
    """ Thresholds the structure matrix in order to compute an adjency matrix.
    All values between threshinf and threshsup are considered representing connected nodes and set to 1. Else are set to 0
    Parameters
    ----------
    C : ndarray, shape (n_nodes,n_nodes)
        The structure matrix to threshold
    threshinf : float
        The minimum value of distance from which the new value is set to 1
    threshsup : float
        The maximum value of distance from which the new value is set to 1
    metric : 'adj' or 'sp'
        Use Shortest Path of Adjacency Matrix as the distance metric
    Returns
    -------
    C : ndarray, shape (n_nodes,n_nodes)
        The threshold matrix. Each element is in {0,1}
    """
    H = np.zeros_like(C)
    np.fill_diagonal(H, np.diagonal(C))
    C = C - H
    ret = np.zeros_like(C)
    if metric == 'sp':
        C = np.minimum(np.maximum(C, threshinf), threshsup)
        ret[C != threshinf] = 1
        ret[C == threshsup] = 0
    elif metric == 'adj':
        ret[C >= threshsup] = 1
        ret[C < threshsup] = 0
    return ret



def FGWMixup(graph_list, label_list, feature_list, nodes, measure='degree', metric='sp', k=0.2, a=0, b=0, alpha=0.95, loss_fun='square_loss', rank=None, bapg=False, rho=0.1):
    graph_0 = graph_list[0]
    graph_1 = graph_list[1]

    dist_0 = shortest_path(graph_0)
    dist_1 = shortest_path(graph_1)
    dist_0[np.isinf(dist_0)] = 999
    dist_1[np.isinf(dist_1)] = 999
    dist_list = [dist_0, dist_1]

    mixup_lambda = np.random.beta(k, k)
    lambdas = [mixup_lambda, 1-mixup_lambda]

    node_size = min(int(mixup_lambda * graph_0.shape[0] + (1-mixup_lambda) * graph_1.shape[0]), 400)
    # node_size = nodes
    # mixup_label = lambdas[0] * label_list[0] + lambdas[1] * label_list[1]

    assert measure in ['degree', 'uniform']

    if measure == 'degree':
        deg_0 = np.sum(graph_0, axis=0)
        deg_1 = np.sum(graph_1, axis=0)
        deg_0 = np.power((deg_0 + a), b)
        deg_1 = np.power((deg_1 + a), b)
        p_0 = deg_0 / np.sum(deg_0)
        p_1 = deg_1 / np.sum(deg_1)
    elif measure == 'uniform':
        p_0 = np.ones(graph_0.shape[0]) / graph_0.shape[0]
        p_1 = np.ones(graph_1.shape[0]) / graph_1.shape[0]
    ps = [p_0, p_1]

    if feature_list is not None:
        if metric == 'sp':
            mixup_feature, mixup_dist, log, n_iter, time = my_fgw_barycenters(N=node_size, Ys=feature_list, Cs=dist_list, ps=ps, max_iter=300, tol=1e-5, rank=rank, bapg=bapg, rho=rho,
                                                        lambdas=lambdas, alpha=alpha, loss_fun=loss_fun, log=True, verbose=False)
            
            print(feature_list, mixup_feature)
            mixup_graph = sp_to_adjency(mixup_dist, threshinf=0, threshsup=find_thresh(mixup_dist, inf=1, sup=5, step=50, metric=metric)[0], metric=metric)
            
        elif metric == 'adj':
            mixup_feature, mixup_graph, log, n_iter, time = my_fgw_barycenters(N=node_size, Ys=feature_list, Cs=graph_list, ps=ps, max_iter=200, tol=5e-4, rank=rank, bapg=bapg, rho=rho,
                                                        lambdas=lambdas, alpha=alpha, loss_fun=loss_fun, log=True, verbose=False)
            mixup_graph = mixup_graph / np.amax(mixup_graph)
            # print(feature_list, mixup_feature)
            real_dists = log['dists']
            if np.amin(real_dists) < 0:
                real_dists -= np.amin(real_dists)
            mixup_w = real_dists[0] / (real_dists[0] + real_dists[1])
            mixup_label = (1 - mixup_w) * label_list[0] + mixup_w * label_list[1]
            ori_label = lambdas[0] * label_list[0] + lambdas[1] * label_list[1]

            # if bapg:
            #     mixup_label = ori_label
            #     mixup_w = lambdas

            threshsup, dist = find_thresh(mixup_graph, graph_list, mixup_label, inf=0, sup=1, step=200, metric=metric)
            # print(threshsup, dist)
            mixup_graph = sp_to_adjency(mixup_graph, threshinf=0, threshsup=threshsup, metric=metric)
            # mixup_label = ori_label

    return mixup_graph, mixup_label, mixup_feature, mixup_w, n_iter, time
    
    

