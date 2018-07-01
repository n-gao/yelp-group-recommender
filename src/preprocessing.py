import numpy as np
import scipy.sparse as sp

def center(A, axis=1, epsilon=1e-12):
    mat = A.copy()
    mat = mat.tocsc() if axis==1 else mat.tocsr()
    means = np.ndarray(mat.shape[axis])
    for i in range(mat.shape[axis]):
        data = mat[i].data if axis == 0 else mat[:,i].data
        if len(data) == 0:
            means[i] = 0
        else:
            means[i] = data.mean()
            
    lil = mat.tolil()
    nnz = lil.nonzero()
    for i in range(lil.nnz):
        j = nnz[axis][i]
        lil[nnz[0][i], nnz[1][i]] -= means[j] + epsilon
    return lil, means, nnz

def cold_start_preprocessing(R, F=None, B=None, A=None, UW=None, BW=None, min_entries=10):
    print("Shape before: {}".format(R.shape))
    
    M, N = R.shape
    user_idx = np.arange(N, dtype=np.int64)
    business_idx = np.arange(M, dtype=np.int64)
    
    shape = (-1, -1)
    while R.shape != shape:
        shape = R.shape
        nnz = R>0
        row_ixs = nnz.sum(1).A1 > min_entries
        R = R[row_ixs]
        if B is not None:
            B = B[row_ixs]
        if A is not None:
            A = A[row_ixs]
            A = A[:,row_ixs]
        if BW is not None:
            BW = BW[row_ixs]
        
        nnz = R>0
        col_ixs = nnz.sum(0).A1 > min_entries
        R = R[:,col_ixs]
        if F is not None:
            F = F[col_ixs]
            F = F[:,col_ixs]
        if UW is not None:
            UW = UW[col_ixs]
        # Correct indices
        M, N = R.shape
        removed_rows = np.where(row_ixs == False)[0]
        business_idx = update_idx(business_idx, removed_rows)
        
        removed_cols = np.where(col_ixs == False)[0]
        user_idx = update_idx(user_idx, removed_cols)
        
    print("Shape after: {}".format(R.shape))
    nnz = R>0
    assert (nnz.sum(0).A1 > min_entries).all()
    assert (nnz.sum(1).A1 > min_entries).all()
    result = [R, user_idx, business_idx]
    if F is not None:
        result.append(F)
    if B is not None:
        result.append(B)
    if A is not None:
        result.append(A)
    if UW is not None:
        result.append(UW)
    if BW is not None:
        result.append(BW)
    return tuple(result)

def update_idx(idx, removed):
    return idx[np.where(np.invert(np.isin(np.arange(len(idx)), removed)))]
    # Code below is not working.
    l_r = len(removed)
    n = len(idx) - l_r
    new_idx = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ind = i + (removed <= i).sum()
        while not (ind >= l_r) and (idx[ind] in removed):
            ind += 1
        new_idx[i] = idx[ind]
    return new_idx

def get_buss_conn_mat(M, relations, threshold=4, min_users=1):
    business_conn = sp.lil_matrix((M,M))
    for row, data in zip(relations.transpose().rows, relations.transpose().data):
        for i, entry_1 in enumerate(row):
            for j, entry_2 in enumerate(row[i+1:]):
                if abs(data[i] - data[j + i]) <= threshold:
                    business_conn[entry_1, entry_2] += 1
                    business_conn[entry_2, entry_1] += 1
    business_conn[business_conn >= min_users] = -1
    business_conn[business_conn >= 1] = 0
    business_conn[business_conn == -1] = 1
    return business_conn
