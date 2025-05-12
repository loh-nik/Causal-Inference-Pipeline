import numpy as np
import oct2py
from scipy.stats import chi2
from oct2py import octave

oc = oct2py.Oct2Py()

octave.addpath("./StateSpaceGC")

def significanceMatrix(G, model_ord, samples, alpha):
    d = G.shape[0]
    pVals = 1 - chi2.cdf(G, df = model_ord, loc=0, scale = 1/samples)
    for i in range(d):
        pVals[i,i] = 1
    m = d**2
    q = (1 /(m-d)) *alpha
    flatPValues = pVals.flatten()
    argSorted = np.argsort(flatPValues)
    inverseSort = np.argsort(argSorted)
    sortedPVals = flatPValues[argSorted]
    significanceArray = np.zeros(m)
    sortedSignificanceMatrix = np.zeros(m)
    for i in range(m):
        # reject null hypothesis of insignificance if our p value is small enough
        significanceArray[i] = sortedPVals[i] < (i+1)*q
    sortedSignificanceMatrix = significanceArray[inverseSort]

    sortedSignificanceMatrix = sortedSignificanceMatrix.reshape(G.shape)
    #return sortedSignificanceMatrix
    return pVals < alpha

# X in shape (variables, observations)
def gcss(X, alpha, tau_max, returnAll = False):
    _, N = X.shape
    _, pbic = octave.ar_IC(X, tau_max, nout=2, verbose=False)
    m,A,C,K,V = octave.s4sid_CCA(X, pbic,nout=5,verbose=False)
    G = octave.iss_PWGC(A,C,K,V, nout=1,verbose=False)
    signif = significanceMatrix(G, m, N, alpha)
    if returnAll:
        return np.array(G)
    return np.array(G)*signif