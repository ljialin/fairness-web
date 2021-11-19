import numpy as np
import pandas as pd
import geatpy as ea
from scipy.spatial.distance import pdist


# SRA + extreme
def SRA_parent_selection3(popobjs):
    # popobjs (N * M): objectives of population, #poluation = n, #objectives = m
    # There is no parent selection, just randomly select
    return None
    
    
def SRA_env_selection3(popobjs, n, levels):
    # popobjs (N * M): objectives of population, #poluation = n, #objectives = m
    # n:               number of better parents
    popobjs1 = popobjs
    popobjs = (popobjs - np.min(popobjs, axis=0))/(np.max(popobjs, axis=0) - np.min(popobjs, axis=0))

    N = popobjs.shape[0]
    SDE_Mat = np.zeros([N, N])
    IBEA_Mat = np.zeros([N, N])

    for i in range(N):
        for j in range(N):
            temp = np.max(np.vstack((popobjs[i, :], popobjs[j, :])), axis=0)
            if i == j:
                SDE_Mat[i, j] = np.inf
            else:
                SDE_Mat[i, j] = np.linalg.norm(popobjs[i, :]-temp, ord=2, axis=0)
                IBEA_Mat[i, j] = np.max((popobjs[i, :] - popobjs[j, :]))

    sdefitness = np.min(SDE_Mat, axis=1)
    C = np.abs(np.max(IBEA_Mat, axis=0))
    IBEA_Mat = IBEA_Mat/C
    ibeafitness = np.sum(-np.exp(-IBEA_Mat/1), axis=0) + 1

    # Stochastic ranking based selection
    Rank = np.arange(0, N)
    pc = 0.4 + np.random.random() * 0.2
    for count_idx in range(int(np.ceil(N/2))):
        is_swap = False
        for i in range(0, N-1):
            if np.random.random() < pc:
                if ibeafitness[Rank[i]] < ibeafitness[Rank[i + 1]]:
                    temp = Rank[i]
                    Rank[i] = Rank[i+1]
                    Rank[i+1] = temp
                    is_swap = True
            else:
                if sdefitness[Rank[i]] < sdefitness[Rank[i + 1]]:
                    temp = Rank[i]
                    Rank[i] = Rank[i + 1]
                    Rank[i + 1] = temp
                    is_swap = True

        if not is_swap:
            break
    # extreme points are preserved
    Remains_idxs = np.array(np.where(levels == 1))
    left_popobjs = popobjs[Remains_idxs[0], :]
    max_idx = np.argmax(left_popobjs, axis=0)
    min_idx = np.argmin(left_popobjs, axis=0)
    min_idx = Remains_idxs[0, min_idx]
    max_idx = Remains_idxs[0, max_idx]
    Chosen = []
    for idx in min_idx:
        Chosen.append(idx)
    for idx in max_idx:
        if idx not in Chosen:
            Chosen.append(idx)

    for idx in Rank:
        if idx not in Chosen:
            Chosen.append(idx)
            if len(Chosen) == n:
                break
    return Chosen
    

if __name__ == '__main__':
    Mat = np.random.rand(100, 2)
    # fitness = SDE_parent_selection(Mat)
    SRemains_idxs = SRA_env_selection(Mat, 5)






