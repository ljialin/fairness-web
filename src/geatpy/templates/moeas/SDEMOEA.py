import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


# original SDE-MOEA
def SDE_parent_selection1(popobjs):
    # popobjs (N * M): objectives of population, #poluation = n, #objectives = m

    popobjs = (popobjs - np.min(popobjs, axis=0))/(np.max(popobjs, axis=0) - np.min(popobjs, axis=0))

    N = popobjs.shape[0]
    SDE_Mat = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            temp = np.max(np.vstack((popobjs[i, :], popobjs[j, :])), axis=0)
            if i == j:
                SDE_Mat[i, j] = np.inf
            else:
                SDE_Mat[i, j] = np.linalg.norm(popobjs[i, :]-temp, ord=2, axis=0)

    # sdefitness = np.min(SDE_Mat, axis=1)
    SDE_Mat_temp = np.fliplr(np.sort(SDE_Mat, axis=1))
    Remains_idxs1 = np.lexsort(-SDE_Mat_temp.T)

    sdefitness = np.zeros([1, N])
    fitval = N + 0.0
    for i in range(N):
        sdefitness[0, Remains_idxs1[i]] = fitval
        fitval = fitval - 1

    return sdefitness[0].reshape(-1, 1)
    
    
def SDE_env_selection1(popobjs, n):
    # popobjs (N * M): objectives of population, #poluation = n, #objectives = m
    # n:               number of better parents
    popobjs1 = popobjs
    popobjs = (popobjs - np.min(popobjs, axis=0))/(np.max(popobjs, axis=0) - np.min(popobjs, axis=0))

    N = popobjs.shape[0]
    SDE_Mat = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            temp = np.max(np.vstack((popobjs[i, :], popobjs[j, :])), axis=0)
            if i == j:
                SDE_Mat[i, j] = np.inf
            else:
                SDE_Mat[i, j] = np.linalg.norm(popobjs[i, :]-temp, ord=2, axis=0)
    sdefitness = np.min(SDE_Mat, axis=1)
    if np.sum(sdefitness != 0) <= n:
        SDE_Mat_temp = np.fliplr(np.sort(SDE_Mat, axis=1))
        Remains_idxs1 = np.lexsort(-SDE_Mat_temp.T)
        Remains_idxs = Remains_idxs1[0:n]
    else:

        Remains = sdefitness != 0
        Remains_idxs = np.array(np.where(Remains))

        for i in range(N-n):
            temp1 = SDE_Mat[Remains, :]
            SDE_Mat_temp = temp1[:, Remains]
            sdefitness = np.min(SDE_Mat_temp, axis=1)
            delete_idx = np.argmin(sdefitness)
            Remains[Remains_idxs[0, delete_idx]] = False
            Remains_idxs = np.array(np.where(Remains))
            if Remains_idxs.shape[1] == n:
                break
    return Remains_idxs
    

if __name__ == '__main__':
    Mat = np.random.rand(10, 2)
    # fitness = SDE_parent_selection(Mat)
    SRemains_idxs = SDE_env_selection(Mat, 5)






