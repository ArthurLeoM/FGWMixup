import numpy as np
from ot.bregman import sinkhorn
from ot.utils import dist, UndefinedParameter, list_to_array
from ot.optim import cg
from ot.lp import emd_1d, emd
from ot.utils import check_random_state, unif
from ot.backend import get_backend

from ot.gromov import *
import time

def fused_ACC_numpy(M, A, B, a=None, b=None, X=None, alpha=0, epoch=2000, eps=1e-5, rho=1e-1):
    if a is None:
        a = np.ones([A.shape[0], 1], dtype=np.float32)/A.shape[0]
    else:
        a = a[:, np.newaxis]
        
    if b is None:
        b = np.ones([B.shape[0], 1], dtype=np.float32)/B.shape[0]
    else:
        b = b[:, np.newaxis]
    
    if X is None:
        X = a @ b.T
    obj_list = []
    for ii in range(epoch):
        X = X + 1e-10
        X_prev = X
        grad = 4*alpha*A@X@B - (1-alpha)*M
        X = np.exp(grad / rho)*X
        X = X * (a / (X @  np.ones_like(b)))
        grad = 4*alpha*A@X@B - (1-alpha)*M
        X = np.exp(grad / rho)*X
        X = X * (b.T / (X.T @ np.ones_like(a)).T)
        if ii > 0 and ii % 10 == 0:
            objective = np.trace(((1-alpha) * M - 2 * alpha * A @ X @ B) @ X.T)
            if len(obj_list) > 0 and np.abs((objective-obj_list[-1])/obj_list[-1]) < eps:
                # print('iter:{}, smaller than eps'.format(ii))
                break
            obj_list.append(objective)
    return X, obj_list



def my_fgw_barycenters(N, Ys, Cs, ps, lambdas, alpha, fixed_structure=False, fixed_features=False,
                    p=None, loss_fun='square_loss', max_iter=100, tol=1e-9, rank=None, bapg=False, rho=0.1,
                    verbose=False, log=False, init_C=None, init_X=None, random_state=None):
    r"""Compute the fgw barycenter as presented eq (5) in :ref:`[24] <references-fgw-barycenters>`

    Parameters
    ----------
    N : int
        Desired number of samples of the target barycenter
    Ys: list of array-like, each element has shape (ns,d)
        Features of all samples
    Cs : list of array-like, each element has shape (ns,ns)
        Structure matrices of all samples
    ps : list of array-like, each element has shape (ns,)
        Masses of all samples.
    lambdas : list of float
        List of the `S` spaces' weights
    alpha : float
        Alpha parameter for the fgw distance
    fixed_structure : bool
        Whether to fix the structure of the barycenter during the updates
    fixed_features : bool
        Whether to fix the feature of the barycenter during the updates
    loss_fun : str
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0).
    verbose : bool, optional
        Print information along iterations.
    log : bool, optional
        Record log if True.
    init_C : array-like, shape (N,N), optional
        Initialization for the barycenters' structure matrix. If not set
        a random init is used.
    init_X : array-like, shape (N,d), optional
        Initialization for the barycenters' features. If not set a
        random init is used.
    random_state : int or RandomState instance, optional
        Fix the seed for reproducibility

    Returns
    -------
    X : array-like, shape (`N`, `d`)
        Barycenters' features
    C : array-like, shape (`N`, `N`)
        Barycenters' structure matrix
    log : dict
        Only returned when log=True. It contains the keys:

        - :math:`\mathbf{T}`: list of (`N`, `ns`) transport matrices
        - :math:`(\mathbf{M}_s)_s`: all distance matrices between the feature of the barycenter and the other features :math:`(dist(\mathbf{X}, \mathbf{Y}_s))_s` shape (`N`, `ns`)


    .. _references-fgw-barycenters:
    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain
        and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    Cs = list_to_array(*Cs)
    ps = list_to_array(*ps)
    Ys = list_to_array(*Ys)
    p = list_to_array(p)
    nx = get_backend(*Cs, *Ys, *ps)

    S = len(Cs)
    d = Ys[0].shape[1]  # dimension on the node features
    if p is None:
        p = nx.ones(N, type_as=Cs[0]) / N

    if fixed_structure:
        if init_C is None:
            raise UndefinedParameter('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            generator = check_random_state(random_state)
            xalea = generator.randn(N, 2)
            C = dist(xalea, xalea)
            C /= C.max()
            C = nx.from_numpy(C, type_as=p)
        else:
            C = init_C

    if fixed_features:
        if init_X is None:
            raise UndefinedParameter('If X is fixed it must be initialized')
        else:
            X = init_X
    else:
        if init_X is None:
            X = nx.zeros((N, d), type_as=ps[0])
        else:
            X = init_X

    T = [nx.outer(p, q) for q in ps]

    Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_ = {}
        log_['err_feature'] = []
        log_['err_structure'] = []
        log_['Ts_iter'] = []
        log_['dists_iter'] = []

    time_start = time.time()
    while((err_feature > tol or err_structure > tol) and cpt < max_iter):
        Cprev = C
        Xprev = X

        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            X = update_feature_matrix(lambdas, Ys_temp, T, p).T

        Ms = [dist(X, Ys[s]) for s in range(len(Ys))]

        if not fixed_structure and cpt > 0:
            if loss_fun == 'square_loss':
                T_temp = [t.T for t in T]
                C = update_structure_matrix(p, lambdas, T_temp, Cs)
                # print('C:', C)

        # if rank is not None:
        #     T = []
        #     dists = []
        #     for s in range(S):
        #         cur_dist, cur_T = entropic_low_rank_fgw(Ms[s], C, Cs[s], p, ps[s], alpha, 
        #                             rank=rank, gamma=100, reg=0, max_iter=max_iter, tol=1e-7, random_state=random_state)
        #         T.append(cur_T)
        #         dists.append(cur_dist)
        
        if bapg:
            T = []
            dists = []
            for s in range(S):
                cur_T, cur_dist = fused_ACC_numpy(Ms[s], C, Cs[s], p, ps[s], alpha=alpha, epoch=300, eps=1e-5, rho=rho)
                T.append(cur_T)
                c1 = np.dot(C*C, np.outer(p, np.ones_like(ps[s]))) + np.dot(np.outer(np.ones_like(p), ps[s]), Cs[s]*Cs[s])
                res = np.trace(np.dot(c1.T, cur_T))
                dists.append(cur_dist[-1] + alpha * res)
                
        else:
            T = []
            dists = []
            for s in range(S):
                cur_T, cur_log = fused_gromov_wasserstein(Ms[s], C, Cs[s], p, ps[s], loss_fun, alpha,
                                        numItermax=300, stopThr=1e-5, verbose=False, log=True)
                T.append(cur_T)
                dists.append(cur_log['fgw_dist'])
            # dists = [fused_gromov_wasserstein2(Ms[s], C, Cs[s], p, ps[s], loss_fun, alpha,
            #                             numItermax=max_iter, stopThr=1e-5, verbose=False) for s in range(S)]
        # print(f'Solve FGW time @ iter {cpt}: {time.time() - time_start}')

        # T is N,ns
        err_feature = nx.norm(X - nx.reshape(Xprev, (N, d))) / nx.norm(nx.reshape(Xprev, (N, d)))
        err_structure = nx.norm(C - Cprev) / nx.norm(Cprev)
        # print(err_feature, err_structure)
        if log:
            log_['err_feature'].append(err_feature)
            log_['err_structure'].append(err_structure)
            log_['Ts_iter'].append(T)
            log_['dists_iter'].append(dists)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format(
                    'It.', 'Err') + '\n' + '-' * 19)
            # if cpt % 10 == 0:
            #     print('Matrix C: ', C)
            #     print('Matrix C non-zeros: {:.4f}, entry_sum: {:.4f}'.format(np.sum(C!=0), np.sum(np.abs(C))))
            #     print('Matrix Cprev: ', Cprev)
            #     print('Matrix Delta: ', C - Cprev)
            print('{:5d}|{:8e}|'.format(cpt, err_structure))
            print('{:5d}|{:8e}|'.format(cpt, err_feature))

        cpt += 1

    all_time = time.time() - time_start
    print(f'----Avg Solve FGW time @ iter {cpt}: {all_time / cpt}')
    # print(C)
    if log:
        log_['T'] = T  # from target to Ys
        log_['p'] = p
        log_['Ms'] = Ms
        log_['dists'] = dists

    if log:
        return X, C, log_, cpt, all_time
    else:
        return X, C, cpt, all_time