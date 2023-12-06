import numpy as np

def aucest(h1c,h0c): 
    n1 = len(h1c)
    n0 = len(h0c)
    h1cmat = h1c[:,np.newaxis]*np.ones(n0)
    h0cmat = np.ones(n1)[:,np.newaxis]*h0c
    mwk = np.zeros([n1,n0])
    mwk[h1cmat>h0cmat] = 1.0
    mwk[h1cmat==h0cmat] = 0.5
    auc = (mwk.sum())/(n0*n1*1.0)
    return auc

def compute_auc(test_small, test_labels, template):
    h1c = []
    h0c = []
    N = len(test_small)
    for k in range(N):
      if test_labels[k].astype('bool'):
        h1c.append((test_small[k,:,:]*template).sum())
      else:
        h0c.append((test_small[k,:,:]*template).sum())

    h1c = np.asarray(h1c)
    h0c = np.asarray(h0c)
    return aucest(h1c,h0c)