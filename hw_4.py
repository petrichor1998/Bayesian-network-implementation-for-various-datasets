import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
import argparse
import warnings
start = time.time()
warnings.filterwarnings("error")
class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.order = []
    def addEdge(self,u,v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def DFS(self,v,vertex):
        visited = [False]*vertex
        self.DFSUtil(v,visited)

    def DFSUtil(self,v,visited):
        visited[v]=True
        self.order.append(v)

        for i in self.graph[v]:
            if visited[i] == False:
                # print(visited)
                self.DFSUtil(i,visited)
def IBN(trmat, vmat, tsmat):
    tvmat = np.concatenate((trmat, vmat), axis = 0)
    #tvmat = trmat
    pprob = tvmat.sum(axis=0)
    nprob = tvmat.shape[0] - pprob
    pprob = pprob + 1
    nprob = nprob + 1
    lpprob = np.log(np.true_divide(pprob, tvmat.shape[0] + 2))
    lnprob =  np.log(np.true_divide(nprob, tvmat.shape[0] + 2))
    tes1 = (tsmat * lpprob).astype(float)
    tes2 = np.ma.array(tes1, mask=tes1 != 0)
    tes3 = tes2 + lnprob
    tes4 = tes3.data.sum(axis=1)
    tes5 = tes4.mean()
    return tes5

def chow_liu(trmat, vmat, tsmat):
    tvmat = np.concatenate((trmat, vmat), axis=0)
    #tvmat = trmat
    #tsmat = trmat
    pprob = tvmat.sum(axis=0)
    pprob = pprob + 1
    nprob = tvmat.shape[0] - pprob + 2
    pprob = np.true_divide(pprob, tvmat.shape[0] + 2)
    nprob = np.true_divide(nprob, tvmat.shape[0] + 2)
    W_list = np.zeros((tvmat.shape[1], tvmat.shape[1]))
    for col1 in range(int(tvmat.shape[1])):
        for col2 in range(int(tvmat.shape[1])):
            if col1 == col2:
                pass
            else:
                a00 = (tvmat[:, col1] == 0) & (tvmat[:, col2] == 0)
                a01 = (tvmat[:, col1] == 0) & (tvmat[:, col2] == 1)
                a10 = (tvmat[:, col1] == 1) & (tvmat[:, col2] == 0)
                a11 = (tvmat[:, col1] == 1) & (tvmat[:, col2] == 1)
                b00 = a00.sum()
                b01 = a01.sum()
                b10 = a10.sum()
                b11 = a11.sum()
                #add laplace smoothing if things dont go well
                P00 = np.true_divide(b00 + 1, int(tvmat.shape[0]) + 4)
                P01 = np.true_divide(b01 + 1, int(tvmat.shape[0]) + 4)
                P10 = np.true_divide(b10 + 1, int(tvmat.shape[0]) + 4)
                P11 = np.true_divide(b11 + 1, int(tvmat.shape[0]) + 4)
                W = P00 * np.log(np.true_divide(P00, (nprob[col1] * nprob[col2])))
                W = W + P01 * np.log(np.true_divide(P01, (nprob[col1] * pprob[col2])))
                W = W + P10 * np.log(np.true_divide(P10, (pprob[col1] * nprob[col2])))
                W = W + P11 * np.log(np.true_divide(P11, (pprob[col1] * pprob[col2])))
                W_list[col1][col2] = W
    W_list = W_list * (-1)
    e = csr_matrix(W_list)
    f = minimum_spanning_tree(e).toarray()
    f_csr = csr_matrix(f)
    l1, l2 = f_csr.toarray().nonzero()
    edges = zip(l1, l2)
    graph = Graph()
    for e in edges:
        graph.addEdge(e[0], e[1])
#The 0th feature is chosen as the root node
    graph.DFS(0, W_list.shape[0])
    o = graph.order
    pa = {o[0]: np.nan}
    for i in range(1, len(o)):
        if o[i] in graph.graph[o[i - 1]]:
            pa[o[i]] = o[i - 1]
        else:
            for j in range(i - 1):
                if o[i] in graph.graph[o[i - j - 2]]:
                    pa[o[i]] = o[i - j - 2]
                    break
                else:
                    pass

    cpt_mat = []
    for child in pa.keys()[1:]:
        A00 = (tvmat[:, child] == 0) & (tvmat[:, pa[child]] == 0)
        A01 = (tvmat[:, child] == 1) & (tvmat[:, pa[child]] == 0)
        A10 = (tvmat[:, child] == 0) & (tvmat[:, pa[child]] == 1)
        A11 = (tvmat[:, child] == 1) & (tvmat[:, pa[child]] == 1)
        B00 = A00.sum()
        B01 = A01.sum()
        B10 = A10.sum()
        B11 = A11.sum()
        # temp1 = (tvmat[:, pa[child]] == 0).sum()
        # temp2 = (tvmat[:, pa[child]] == 1).sum()
        p00 = np.true_divide(B00 + 1, (tvmat[:, pa[child]] == 0).sum() + 2)
        p01 = np.true_divide(B01 + 1, (tvmat[:, pa[child]] == 0).sum() + 2)
        p10 = np.true_divide(B10 + 1, (tvmat[:, pa[child]] == 1).sum() + 2)
        p11 = np.true_divide(B11 + 1, (tvmat[:, pa[child]] == 1).sum() + 2)
        cpt_mat.append([p00, p01, p10, p11])
    cpt_mat = np.array(cpt_mat)
    lcpt_mat = np.log(cpt_mat)
    # log of probabilities of the first feature which is the root
    lpprobX0 = np.log(np.true_divide((tvmat[:, 0].sum() + 1), tvmat.shape[0] + 2))
    lnprobX0 = np.log(np.true_divide((tvmat.shape[0] - tvmat[:, 0].sum() + 1), tvmat.shape[0] + 2))
    # making a new matrix with the log of conditional probabilities of the values in the test set
    t = tsmat.copy()
    t = t.astype(float)
    for i in range(1, tvmat.shape[1]):
        par = pa[i]
        t[(tsmat[:, i] == 0) & (tsmat[:, par] == 0), i] = lcpt_mat[i - 1][0]
        t[(tsmat[:, i] == 1) & (tsmat[:, par] == 0), i] = lcpt_mat[i - 1][1]
        t[(tsmat[:, i] == 0) & (tsmat[:, par] == 1), i] = lcpt_mat[i - 1][2]
        t[(tsmat[:, i] == 1) & (tsmat[:, par] == 1), i] = lcpt_mat[i - 1][3]
    # filling up the 0th column in t with the respective probabilities
    t[(tsmat[:, 0]) == 0, 0] = lnprobX0
    t[(tsmat[:, 0]) == 1, 0] = lpprobX0
    # summing the logs to find the loglikelihood
    LLH_col = t.sum(axis=1)
    mLL = LLH_col.mean()
    return mLL


def MT(tmat, vmat, tsmat, od, fn):
    trmat = tmat.copy()
    #k_list = [2,3,4]
    K_selection = {}
    #contains all cpts of each k in a list
    K_cpts = {}
    #contains Lambdas of each K ina a list
    K_lambdas = {}
    #storing the results of 10 test set log likelihoods
    tset_llh = []
    #this is the validation set testing code for finding the optimum k
    # for K in range(len(k_list)):
    #     print "training for K = {} ".format(k_list[K])
    #     # E step
    #     h_mat = np.random.rand(trmat.shape[0], k_list[K])
    #     hmat_l = []
    #     llhs_list = []
    #     cpts_l = []
    #     # making the mutual infomation matrix
    #     # M step
    #     epochs = 50
    #     for i in range(epochs):
    #         if (i + 1) % 10 == 0:
    #             print("epoch no. : ", i + 1)
    #         h_mat = h_mat / h_mat.sum(axis=1)[:, None]
    #         #print(h_mat)
    #         hmat_l.append(h_mat.copy())
    #         llh_list = []
    #         cpt_l = []
    #         for k in range(k_list[K]):
    #             l = np.true_divide(h_mat[:, k].sum(), h_mat.shape[0])
    #             W_list = np.zeros((trmat.shape[1], trmat.shape[1]))
    #             for col1 in range(int(trmat.shape[1])):
    #                 for col2 in range(int(trmat.shape[1])):
    #                     if col1 == col2:
    #                         pass
    #                     else:
    #                         a00 = h_mat[:, k][(trmat[:, col1] == 0) & (trmat[:, col2] == 0)]
    #                         a01 = h_mat[:, k][(trmat[:, col1] == 0) & (trmat[:, col2] == 1)]
    #                         a10 = h_mat[:, k][(trmat[:, col1] == 1) & (trmat[:, col2] == 0)]
    #                         a11 = h_mat[:, k][(trmat[:, col1] == 1) & (trmat[:, col2] == 1)]
    #                         b00 = a00.sum()
    #                         b01 = a01.sum()
    #                         b10 = a10.sum()
    #                         b11 = a11.sum()
    #
    #                         P00 = np.true_divide(b00 + l, h_mat[:, k].sum() + 4 * l)
    #                         P01 = np.true_divide(b01 + l, h_mat[:, k].sum() + 4 * l)
    #                         P10 = np.true_divide(b10 + l, h_mat[:, k].sum() + 4 * l)
    #                         P11 = np.true_divide(b11 + l, h_mat[:, k].sum() + 4 * l)
    #
    #                         pprob0 = h_mat[:, k][trmat[:, col1] == 1]
    #                         pprob0 = np.true_divide(pprob0.sum() + l, h_mat[:, k].sum() + 2 * l)
    #                         pprob1 = h_mat[:, k][trmat[:, col2] == 1]
    #                         pprob1 = np.true_divide(pprob1.sum() + l, h_mat[:, k].sum() + 2 * l)
    #                         nprob0 = h_mat[:, k][trmat[:, col1] == 0]
    #                         nprob0 = np.true_divide(nprob0.sum() + l, h_mat[:, k].sum() + 2 * l)
    #                         nprob1 = h_mat[:, k][trmat[:, col2] == 0]
    #                         nprob1 = np.true_divide(nprob1.sum() + l, h_mat[:, k].sum() + 2 * l)
    #
    #                         W = P00 * np.log(np.true_divide(P00, (nprob0 * nprob1)))
    #                         W = W + P01 * np.log(np.true_divide(P01, (nprob0 * pprob1)))
    #                         W = W + P10 * np.log(np.true_divide(P10, (pprob0 * nprob1)))
    #                         W = W + P11 * np.log(np.true_divide(P11, (pprob0 * pprob1)))
    #                         W_list[col1][col2] = W
    #             W_list = W_list * (-1)
    #             e = csr_matrix(W_list)
    #             f = minimum_spanning_tree(e).toarray().astype(float)
    #             f_csr = csr_matrix(f)
    #             l1, l2 = f_csr.toarray().nonzero()
    #             edges = zip(l1, l2)
    #             graph = Graph()
    #             for e in edges:
    #                 graph.addEdge(e[0], e[1])
    #             # The 0th feature is chosen as the root node
    #             graph.DFS(0, W_list.shape[0])
    #             o = graph.order
    #             pa = {o[0]: np.nan}
    #             for i in range(1, len(o)):
    #                 if o[i] in graph.graph[o[i - 1]]:
    #                     pa[o[i]] = o[i - 1]
    #                 else:
    #                     for j in range(i - 1):
    #                         if o[i] in graph.graph[o[i - j - 2]]:
    #                             pa[o[i]] = o[i - j - 2]
    #                             break
    #                         else:
    #                             pass
    #
    #             cpt_mat = []
    #
    #             for child in pa.keys()[1:]:
    #                 try:
    #                     lc0 = np.true_divide((h_mat[:, k][trmat[:, pa[child]] == 0]).sum(), (trmat[:, pa[child]] == 0).sum())
    #                 except RuntimeWarning:
    #                     lc0 = 0.5
    #                 try:
    #                     lc1 = np.true_divide(h_mat[:, k][trmat[:, pa[child]] == 1].sum(), (trmat[:, pa[child]] == 1).sum())
    #                 except RuntimeWarning:
    #                     lc1 = 0.5
    #                 A00 = h_mat[:, k][(trmat[:, child] == 0) & (trmat[:, pa[child]] == 0)]
    #                 A00 = np.true_divide(A00.sum() + lc0, h_mat[:, k][trmat[:, pa[child]] == 0].sum() + 2 * lc0)
    #                 A01 = h_mat[:, k][(trmat[:, child] == 1) & (trmat[:, pa[child]] == 0)]
    #                 A01 = np.true_divide(A01.sum() + lc0, h_mat[:, k][trmat[:, pa[child]] == 0].sum() + 2 * lc0)
    #                 A10 = h_mat[:, k][(trmat[:, child] == 0) & (trmat[:, pa[child]] == 1)]
    #                 A10 = np.true_divide(A10.sum() + lc1, h_mat[:, k][trmat[:, pa[child]] == 1].sum() + 2 * lc1)
    #                 A11 = h_mat[:, k][(trmat[:, child] == 1) & (trmat[:, pa[child]] == 1)]
    #                 A11 = np.true_divide(A11.sum() + lc1, h_mat[:, k][trmat[:, pa[child]] == 1].sum() + 2 * lc1)
    #                 cpt_mat.append([A00, A01, A10, A11])
    #
    #             cpt_mat = np.array(cpt_mat)
    #             lpprobX0 = h_mat[:, k][(trmat[:, 0] == 1)]
    #             lpprobX0 = np.log(np.true_divide(lpprobX0.sum() + l, h_mat[:, k].sum() + 2 * l))
    #
    #             lnprobX0 = h_mat[:, k][(trmat[:, 0] == 0)]
    #             lnprobX0 = np.log(np.true_divide(lnprobX0.sum() + l, h_mat[:, k].sum() + 2 * l))
    #
    #             lcpt_mat = np.log(cpt_mat)
    #             lcpt_mat = np.vstack((np.array([lpprobX0, lnprobX0, 0, 0]), lcpt_mat))
    #             cpt_l.append(lcpt_mat)
    #
    #             t = trmat.copy()
    #             t = t.astype(float)
    #             for i in range(1, trmat.shape[1]):
    #                 par = pa[i]
    #                 t[(trmat[:, i] == 0) & (trmat[:, par] == 0), i] = lcpt_mat[i][0]
    #                 t[(trmat[:, i] == 1) & (trmat[:, par] == 0), i] = lcpt_mat[i][1]
    #                 t[(trmat[:, i] == 0) & (trmat[:, par] == 1), i] = lcpt_mat[i][2]
    #                 t[(trmat[:, i] == 1) & (trmat[:, par] == 1), i] = lcpt_mat[i][3]
    #             # filling up the 0th column in t with the respective probabilities
    #             t[(trmat[:, 0]) == 0, 0] = lcpt_mat[0][1]
    #             t[(trmat[:, 0]) == 1, 0] = lcpt_mat[0][0]
    #             # summing the logs to find the loglikelihood
    #             LLH_col = t.sum(axis=1)
    #             llh_list.append(LLH_col.mean())
    #             Phgx = h_mat[:, k] * np.exp(LLH_col)
    #             if i == epochs - 1:
    #                 pass
    #             else:
    #                 h_mat[:, k] = Phgx
    #         cpts_l.append(cpt_l)
    #         llhs_list.append(llh_list)
    #     K_cpts[k_list[K]] = cpts_l[-1]
    #
    #     temp_l = [np.true_divide(h_mat[:, hcol].sum(), h_mat.shape[0]) for hcol in range(k_list[K])]
    #
    #     K_lambdas[k_list[K]] = temp_l
    #     # testing on the validation set
    #     k_llh_sum = []
    #     for k in range(k_list[K]):
    #         t = vmat.copy()
    #         t = t.astype(float)
    #         for i in range(1, vmat.shape[1]):
    #             par = pa[i]
    #             t[(vmat[:, i] == 0) & (vmat[:, par] == 0), i] = cpts_l[-1][k][i][0]
    #             t[(vmat[:, i] == 1) & (vmat[:, par] == 0), i] = cpts_l[-1][k][i][1]
    #             t[(vmat[:, i] == 0) & (vmat[:, par] == 1), i] = cpts_l[-1][k][i][2]
    #             t[(vmat[:, i] == 1) & (vmat[:, par] == 1), i] = cpts_l[-1][k][i][3]
    #         # filling up the 0th column in t with the respective probabilities
    #         t[(vmat[:, 0]) == 0, 0] = cpts_l[-1][k][0][1]
    #         t[(vmat[:, 0]) == 1, 0] = cpts_l[-1][k][0][0]
    #         llh = t.sum(axis=1)
    #         llh = llh.mean()
    #         k_llh_sum.append(llh * h_mat[:, k].sum() / h_mat.shape[0])
    #     k_llh_sum = np.array(k_llh_sum)
    #     f_mllh = k_llh_sum.sum()
    #     K_selection[k_list[K]] = f_mllh
    #
    # max_k = max(K_selection.iteritems(), key=operator.itemgetter(1))[0]

    print("testing the set 10 times")
    for i in range(10):
        test_llh_sum = []
        print("evaluation no. {}".format(i + 1))
        #E step
        h_mat = np.random.rand(trmat.shape[0], od[fn])
        hmat_l = []
        llhs_list = []
        cpts_l = []
        # making the mutual infomation matrix
        # M step
        epochs = 50
        for i in range(epochs):
            if (i + 1) % 10 == 0:
                print("epoch no. : {}".format(i + 1))
            h_mat = h_mat / h_mat.sum(axis=1)[:, None]
            #print(h_mat)
            hmat_l.append(h_mat.copy())
            llh_list = []
            cpt_l = []
            for k in range(od[fn]):
                l = np.true_divide(h_mat[:, k].sum(), h_mat.shape[0])
                W_list = np.zeros((trmat.shape[1], trmat.shape[1]))
                for col1 in range(int(trmat.shape[1])):
                    for col2 in range(int(trmat.shape[1])):
                        if col1 == col2:
                            pass
                        else:
                            a00 = h_mat[:, k][(trmat[:, col1] == 0) & (trmat[:, col2] == 0)]
                            a01 = h_mat[:, k][(trmat[:, col1] == 0) & (trmat[:, col2] == 1)]
                            a10 = h_mat[:, k][(trmat[:, col1] == 1) & (trmat[:, col2] == 0)]
                            a11 = h_mat[:, k][(trmat[:, col1] == 1) & (trmat[:, col2] == 1)]
                            b00 = a00.sum()
                            b01 = a01.sum()
                            b10 = a10.sum()
                            b11 = a11.sum()

                            P00 = np.true_divide(b00 + l, h_mat[:, k].sum() + 4 * l)
                            P01 = np.true_divide(b01 + l, h_mat[:, k].sum() + 4 * l)
                            P10 = np.true_divide(b10 + l, h_mat[:, k].sum() + 4 * l)
                            P11 = np.true_divide(b11 + l, h_mat[:, k].sum() + 4 * l)

                            pprob0 = h_mat[:, k][trmat[:, col1] == 1]
                            pprob0 = np.true_divide(pprob0.sum() + l, h_mat[:, k].sum() + 2 * l)
                            pprob1 = h_mat[:, k][trmat[:, col2] == 1]
                            pprob1 = np.true_divide(pprob1.sum() + l, h_mat[:, k].sum() + 2 * l)
                            nprob0 = h_mat[:, k][trmat[:, col1] == 0]
                            nprob0 = np.true_divide(nprob0.sum() + l, h_mat[:, k].sum() + 2 * l)
                            nprob1 = h_mat[:, k][trmat[:, col2] == 0]
                            nprob1 = np.true_divide(nprob1.sum() + l, h_mat[:, k].sum() + 2 * l)

                            W = P00 * np.log(np.true_divide(P00, (nprob0 * nprob1)))
                            W = W + P01 * np.log(np.true_divide(P01, (nprob0 * pprob1)))
                            W = W + P10 * np.log(np.true_divide(P10, (pprob0 * nprob1)))
                            W = W + P11 * np.log(np.true_divide(P11, (pprob0 * pprob1)))
                            W_list[col1][col2] = W
                W_list = W_list * (-1)
                e = csr_matrix(W_list)
                f = minimum_spanning_tree(e).toarray().astype(float)
                f_csr = csr_matrix(f)
                l1, l2 = f_csr.toarray().nonzero()
                edges = zip(l1, l2)
                graph = Graph()
                for e in edges:
                    graph.addEdge(e[0], e[1])
                # The 0th feature is chosen as the root node
                graph.DFS(0, W_list.shape[0])
                o = graph.order
                pa = {o[0]: np.nan}
                for i in range(1, len(o)):
                    if o[i] in graph.graph[o[i - 1]]:
                        pa[o[i]] = o[i - 1]
                    else:
                        for j in range(i - 1):
                            if o[i] in graph.graph[o[i - j - 2]]:
                                pa[o[i]] = o[i - j - 2]
                                break
                            else:
                                pass

                cpt_mat = []

                for child in pa.keys()[1:]:
                    try:
                        lc0 = np.true_divide((h_mat[:, k][trmat[:, pa[child]] == 0]).sum(), (trmat[:, pa[child]] == 0).sum())
                    except RuntimeWarning:
                        lc0 = 0.5
                    try:
                        lc1 = np.true_divide(h_mat[:, k][trmat[:, pa[child]] == 1].sum(), (trmat[:, pa[child]] == 1).sum())
                    except RuntimeWarning:
                        lc1 = 0.5
                    A00 = h_mat[:, k][(trmat[:, child] == 0) & (trmat[:, pa[child]] == 0)]
                    A00 = np.true_divide(A00.sum() + lc0, h_mat[:, k][trmat[:, pa[child]] == 0].sum() + 2 * lc0)
                    A01 = h_mat[:, k][(trmat[:, child] == 1) & (trmat[:, pa[child]] == 0)]
                    A01 = np.true_divide(A01.sum() + lc0, h_mat[:, k][trmat[:, pa[child]] == 0].sum() + 2 * lc0)
                    A10 = h_mat[:, k][(trmat[:, child] == 0) & (trmat[:, pa[child]] == 1)]
                    A10 = np.true_divide(A10.sum() + lc1, h_mat[:, k][trmat[:, pa[child]] == 1].sum() + 2 * lc1)
                    A11 = h_mat[:, k][(trmat[:, child] == 1) & (trmat[:, pa[child]] == 1)]
                    A11 = np.true_divide(A11.sum() + lc1, h_mat[:, k][trmat[:, pa[child]] == 1].sum() + 2 * lc1)
                    cpt_mat.append([A00, A01, A10, A11])

                cpt_mat = np.array(cpt_mat)
                lpprobX0 = h_mat[:, k][(trmat[:, 0] == 1)]
                lpprobX0 = np.log(np.true_divide(lpprobX0.sum() + l, h_mat[:, k].sum() + 2 * l))

                lnprobX0 = h_mat[:, k][(trmat[:, 0] == 0)]
                lnprobX0 = np.log(np.true_divide(lnprobX0.sum() + l, h_mat[:, k].sum() + 2 * l))

                lcpt_mat = np.log(cpt_mat)
                lcpt_mat = np.vstack((np.array([lpprobX0, lnprobX0, 0, 0]), lcpt_mat))
                cpt_l.append(lcpt_mat)

                t = trmat.copy()
                t = t.astype(float)
                for i in range(1, trmat.shape[1]):
                    par = pa[i]
                    t[(trmat[:, i] == 0) & (trmat[:, par] == 0), i] = lcpt_mat[i][0]
                    t[(trmat[:, i] == 1) & (trmat[:, par] == 0), i] = lcpt_mat[i][1]
                    t[(trmat[:, i] == 0) & (trmat[:, par] == 1), i] = lcpt_mat[i][2]
                    t[(trmat[:, i] == 1) & (trmat[:, par] == 1), i] = lcpt_mat[i][3]
                # filling up the 0th column in t with the respective probabilities
                t[(trmat[:, 0]) == 0, 0] = lcpt_mat[0][1]
                t[(trmat[:, 0]) == 1, 0] = lcpt_mat[0][0]
                # summing the logs to find the loglikelihood
                LLH_col = t.sum(axis=1)
                llh_list.append(LLH_col.mean())
                Phgx = h_mat[:, k] * np.exp(LLH_col)
                if i == epochs - 1:
                    pass
                else:
                    h_mat[:, k] = Phgx
            cpts_l.append(cpt_l)
            llhs_list.append(llh_list)
            K_cpts[od[fn]] = cpts_l[-1]

            temp_l = [np.true_divide(h_mat[:, hcol].sum(), h_mat.shape[0]) for hcol in range(od[fn])]

            K_lambdas[od[fn]] = temp_l
        for k in range(od[fn]):
            #summation of lambda k multiplied by the tree
            t = tsmat.copy()
            t = t.astype(float)
            for i in range(1, tsmat.shape[1]):
                par = pa[i]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 0), i] = K_cpts[od[fn]][k][i][0]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 0), i] = K_cpts[od[fn]][k][i][1]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 1), i] = K_cpts[od[fn]][k][i][2]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 1), i] = K_cpts[od[fn]][k][i][3]
            # filling up the 0th column in t with the respective probabilities
            t[(tsmat[:, 0]) == 0, 0] = K_cpts[od[fn]][k][0][1]
            t[(tsmat[:, 0]) == 1, 0] = K_cpts[od[fn]][k][0][0]
            llh = t.sum(axis = 1)
            llh = llh.mean()
            test_llh_sum.append(llh * K_lambdas[od[fn]][k])
        test_llh_sum = np.array(test_llh_sum)
        test_mllh = test_llh_sum.sum()
        tset_llh.append(test_mllh)
    tsetllh = np.array(tset_llh)
    t1 = np.mean(tset_llh)
    t2 = np.std(tset_llh)

    return tsetllh, t1, t2

def RFtree(trmat, vmat, tsmat, od, fn):
    #nos = [3, 5, 7]
    sample_cpt_dict = {}
    tset_llh = []
    print("evaluating the loglikelihood on test set 10 times")
    for n in range(10):
        print("evaluation no. : {}".format(n + 1))
        samples = []
        # generating the samples
        for k in range(od[fn][0]):
            temp_arr = np.array([np.inf] * trmat.shape[1])
            for i in range(trmat.shape[0] / 3):
                ri = np.random.randint(0, trmat.shape[0])
                temp_arr = np.vstack((temp_arr, trmat[ri]))
            temp_arr = temp_arr[1:, :]
            samples.append(temp_arr)
        # making the chow_liu trees off the samples generated
        cpt_list = []
        for s in range(len(samples)):
            pprob = samples[s].sum(axis=0)
            pprob = pprob + 1
            nprob = samples[s].shape[0] - pprob + 2
            pprob = np.true_divide(pprob, samples[s].shape[0] + 2)
            nprob = np.true_divide(nprob, samples[s].shape[0] + 2)
            W_list = np.zeros((samples[s].shape[1], samples[s].shape[1]))
            for col1 in range(int(samples[s].shape[1])):
                for col2 in range(int(samples[s].shape[1])):
                    if col1 == col2:
                        pass
                    else:
                        a00 = (samples[s][:, col1] == 0) & (samples[s][:, col2] == 0)
                        a01 = (samples[s][:, col1] == 0) & (samples[s][:, col2] == 1)
                        a10 = (samples[s][:, col1] == 1) & (samples[s][:, col2] == 0)
                        a11 = (samples[s][:, col1] == 1) & (samples[s][:, col2] == 1)
                        b00 = a00.sum()
                        b01 = a01.sum()
                        b10 = a10.sum()
                        b11 = a11.sum()
                        P00 = np.true_divide(b00 + 1, int(samples[s].shape[0]) + 4)
                        P01 = np.true_divide(b01 + 1, int(samples[s].shape[0]) + 4)
                        P10 = np.true_divide(b10 + 1, int(samples[s].shape[0]) + 4)
                        P11 = np.true_divide(b11 + 1, int(samples[s].shape[0]) + 4)
                        W = P00 * np.log(np.true_divide(P00, (nprob[col1] * nprob[col2])))
                        W = W + P01 * np.log(np.true_divide(P01, (nprob[col1] * pprob[col2])))
                        W = W + P10 * np.log(np.true_divide(P10, (pprob[col1] * nprob[col2])))
                        W = W + P11 * np.log(np.true_divide(P11, (pprob[col1] * pprob[col2])))
                        W_list[col1][col2] = W
            for num in range(od[fn][1]):
                ri1 = np.random.randint(0, trmat.shape[1])
                ri2 = np.random.randint(0, trmat.shape[1])
                W_list[ri1][ri2] = 0
                W_list[ri2][ri1] = 0
            W_list = W_list * (-1)
            e = csr_matrix(W_list)
            f = minimum_spanning_tree(e).toarray()
            f_csr = csr_matrix(f)
            l1, l2 = f_csr.toarray().nonzero()
            edges = zip(l1, l2)
            graph = Graph()
            for e in edges:
                graph.addEdge(e[0], e[1])
            # The 0th feature is chosen as the root node
            graph.DFS(0, W_list.shape[0])
            o = graph.order
            pa = {o[0]: np.nan}
            for i in range(1, len(o)):
                if o[i] in graph.graph[o[i - 1]]:
                    pa[o[i]] = o[i - 1]
                else:
                    for j in range(i - 1):
                        if o[i] in graph.graph[o[i - j - 2]]:
                            pa[o[i]] = o[i - j - 2]
                            break
                        else:
                            pass

            cpt_mat = []
            for child in pa.keys()[1:]:
                A00 = (samples[s][:, child] == 0) & (samples[s][:, pa[child]] == 0)
                A01 = (samples[s][:, child] == 1) & (samples[s][:, pa[child]] == 0)
                A10 = (samples[s][:, child] == 0) & (samples[s][:, pa[child]] == 1)
                A11 = (samples[s][:, child] == 1) & (samples[s][:, pa[child]] == 1)
                B00 = A00.sum()
                B01 = A01.sum()
                B10 = A10.sum()
                B11 = A11.sum()
                p00 = np.true_divide(B00 + 1, (samples[s][:, pa[child]] == 0).sum() + 2)
                p01 = np.true_divide(B01 + 1, (samples[s][:, pa[child]] == 0).sum() + 2)
                p10 = np.true_divide(B10 + 1, (samples[s][:, pa[child]] == 1).sum() + 2)
                p11 = np.true_divide(B11 + 1, (samples[s][:, pa[child]] == 1).sum() + 2)
                cpt_mat.append([p00, p01, p10, p11])
            cpt_mat = np.array(cpt_mat)
            lcpt_mat = np.log(cpt_mat)
            # log of probabilities of the first feature which is the root
            lpprobX0 = np.log(np.true_divide((samples[s][:, 0].sum() + 1), samples[s].shape[0] + 2))
            lnprobX0 = np.log(
                np.true_divide((samples[s].shape[0] - samples[s][:, 0].sum() + 1), samples[s].shape[0] + 2))
            lcpt_mat = np.vstack(([lpprobX0, lnprobX0, 0, 0], lcpt_mat))
            cpt_list.append(lcpt_mat)
        sample_cpt_dict[od[fn][0]] = cpt_list
    # # test for the best k in the validation set
    # valid_llh_dict = {}
    #
    # for k in sample_cpt_dict.keys():
    #     k_llh_sum = []
    #     for num in range(k):
    #         t = vmat.copy()
    #         t = t.astype(float)
    #         for i in range(1, vmat.shape[1]):
    #             par = pa[i]
    #             t[(vmat[:, i] == 0) & (vmat[:, par] == 0), i] = sample_cpt_dict[k][num][i][0]
    #             t[(vmat[:, i] == 1) & (vmat[:, par] == 0), i] = sample_cpt_dict[k][num][i][1]
    #             t[(vmat[:, i] == 0) & (vmat[:, par] == 1), i] = sample_cpt_dict[k][num][i][2]
    #             t[(vmat[:, i] == 1) & (vmat[:, par] == 1), i] = sample_cpt_dict[k][num][i][3]
    #         # filling up the 0th column in t with the respective probabilities
    #         t[(vmat[:, 0]) == 0, 0] = sample_cpt_dict[k][num][0][1]
    #         t[(vmat[:, 0]) == 1, 0] = sample_cpt_dict[k][num][0][0]
    #         llh = t.sum(axis=1)
    #         llh = llh.mean()
    #         # TODO: for extra credit change the measure for probability of tree here
    #         k_llh_sum.append(np.true_divide(llh, k))
    #     k_llh_sum = np.array(k_llh_sum)
    #     valid_llh_dict[k] = k_llh_sum.sum()

    # max_k = max(valid_llh_dict.iteritems(), key=operator.itemgetter(1))[0]
    # testing on the test set
        f_llh_sum = []

        for num in range(od[fn][0]):
            t = tsmat.copy()
            t = t.astype(float)
            for i in range(1, tsmat.shape[1]):
                par = pa[i]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 0), i] = sample_cpt_dict[od[fn][0]][num][i][0]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 0), i] = sample_cpt_dict[od[fn][0]][num][i][1]
                t[(tsmat[:, i] == 0) & (tsmat[:, par] == 1), i] = sample_cpt_dict[od[fn][0]][num][i][2]
                t[(tsmat[:, i] == 1) & (tsmat[:, par] == 1), i] = sample_cpt_dict[od[fn][0]][num][i][3]
            # filling up the 0th column in t with the respective probabilities
            t[(tsmat[:, 0]) == 0, 0] = sample_cpt_dict[od[fn][0]][num][0][1]
            t[(tsmat[:, 0]) == 1, 0] = sample_cpt_dict[od[fn][0]][num][0][0]
            llh = t.sum(axis=1)
            llh = llh.mean()
        f_llh_sum.append(np.true_divide(llh, od[fn][0]))
        f_llh_sum = np.array(f_llh_sum)
        f_llh = f_llh_sum.sum()
        tset_llh.append(f_llh)
    tset_llh = np.array(tset_llh)
    t1 = tset_llh.mean()
    t2 = tset_llh.std()
    return tset_llh, t1, t2


#loading train, valid and test set
def main():

    parser = argparse.ArgumentParser(description = "Runs the specified algorithm on the file path mentioned")
    parser.add_argument('-algorithm_number', '--algo_no', type=int)
    parser.add_argument('-train_data', '--train_set', type=str)
    parser.add_argument('-valid_data', '--valid_set', type=str)
    parser.add_argument('-test_data', '--test_set', type=str)

    arg = parser.parse_args()
    algo_no = arg.algo_no
    train_name = arg.train_set
    valid_name = arg.valid_set
    test_name = arg.test_set

    # train_mat = np.loadtxt("small-10-datasets/r52.ts.data", delimiter=',')
    # valid_mat = np.loadtxt("small-10-datasets/r52.valid.data", delimiter=',')
    # test_mat = np.loadtxt("small-10-datasets/r52.test.data", delimiter=',')

    train_mat = np.loadtxt(train_name, delimiter=',')
    valid_mat = np.loadtxt(valid_name, delimiter=',')
    test_mat = np.loadtxt(test_name, delimiter=',')


    if algo_no == 1:
        #implementing the Independent bayesian networks algo
        print("Running the Independent Bayesian Network Algorithm")
        avgLL_ibn = IBN(train_mat, valid_mat, test_mat)
        print("Average log-likelihood for independent bayesian networks : ", avgLL_ibn)

    if algo_no == 2:
        # #Chow-liu tree
        print("Running the Chow-Liu Tree Algorithm")
        avgLL_cl = chow_liu(train_mat, valid_mat, test_mat)
        print("Average log-likelihood for chow_liu : ", avgLL_cl)

    if algo_no == 3:
        #Mixture of trees
        fname = train_name.split("/")[-1]
        fname = fname.split(".")[0]
        print("data_set name = {}".format(fname))
        optimal_k_dict = {"accidents" : 3, "baudio" : 3, "bnetflix" : 2, "dna" : 2, "jester" : 3, "kdd" : 3, "msnbc" : 4, "nltcs" : 4, "plants" : 3, "r52" : 3}
        MTllh_list, MTaverage, MTstd = MT(train_mat, valid_mat, test_mat, optimal_k_dict, fname)
        print("llh list : {}".format(MTllh_list))
        print("average: {}".format(MTaverage))
        print("std: {}".format(MTstd))

    if algo_no == 4:
        #rf tree
        fname = train_name.split("/")[-1]
        fname = fname.split(".")[0]
        print("fname = {}".format(fname))
        optimal_kr_dict = {"accidents": [3, 5], "baudio": [3, 6], "bnetflix": [2, 5], "dna": [2, 6], "jester": [3, 5],
                           "kdd": [3, 7], "msnbc": [4, 6], "nltcs": [4, 5], "plants": [3, 5], "r52": [3, 6]}
        RF_avgllh, RFavg, RFsd = RFtree(train_mat, valid_mat, test_mat, optimal_kr_dict, fname)
        print("likelihood list of 10 evaluations {}".format(RF_avgllh))
        print("Average loglikelihood of the data using RFtree", RFavg)
        print("Standard deviation = {}".format(RFsd))



    print "Time taken to execute is ", time.time() - start
main()



