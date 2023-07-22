import numpy as np
import numpy.ma as ma


def kappa_matrix(X, return_agreement=False):
    """Pairwise agreement between items with sample of n respondents.
       Returns matrix of Cohen's kappa.
       Optionally returns matrix of proportion agreement.

    Parameters
    ----------
    X : array_like
        An n by k array of n binary responses to k items.
       
    Returns
    -------
    kappa : ndarray
        A k x k array of Cohen's kappa. 
        Kappa between X[:,i] and X[:,j] is kappa[i, j] or kappa[j, i]. 
    agreement : ndarray, optional
        A k x k array of proportion agreement. 
        Agreement between X[:,i] and X[:,j] is agreement[i, j] or agreement[j, i]. 
    """
    if (X.dtype.kind=='i'):
        X = X.astype(float)
    R = ma.masked_invalid(X)  # (n x k)
    yesYes = ma.dot(R.transpose(), R)  # counts of yes-yes (k x k)
    F = ma.abs(R-1)  # [0,1] -> [1,0]
    noNo = ma.dot(F.transpose(), F)  # counts of no-no   (k x k)
    S = yesYes + noNo  # counts of agreements (k x k)
    valid = np.ones_like(R)  # valid responses (n x k)
    valid[ma.getmaskarray(R)] = 0
    N = np.dot(valid.transpose(), valid)
    A = ma.multiply(S, N**-1)
    assert A[~A.mask].max() == 1  # note: includes diagonals (100% agreement)
    assert A[~A.mask].min() >= 0
    yesNo = ma.dot(R.transpose(), F)  # counts of yes-no  (k x k)
    noYes = ma.dot(F.transpose(), R)  # counts of no-yes  (k x k)
    Y = ma.multiply(ma.multiply(yesYes+yesNo, N**-1), ma.multiply(yesYes+noYes, N**-1))
    N = ma.multiply(ma.multiply(noYes+noNo, N**-1), ma.multiply(yesNo+noNo, N**-1))
    E = Y + N
    kappa = ma.multiply((A-E), (1-E)**-1)
    assert kappa.max() == 1
    if return_agreement:
        return kappa.filled(np.nan), A.filled(np.nan)
    else:
        return kappa.filled(np.nan)
    
# End
#################################################################