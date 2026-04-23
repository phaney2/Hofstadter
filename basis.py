import numpy as np


def outer_product(mtx1, labels1, mtx2, labels2):
    """
    Kronecker-style outer product of two matrices with label tracking.

    Returns (new_matrix, new_labels) where new_matrix[i,j] blocks are
    mtx1[i1,j1] * mtx2, and labels are concatenated.
    """
    dim1 = mtx1.shape[0]
    dim2 = mtx2.shape[0]
    newmtx = np.zeros((dim1 * dim2, dim1 * dim2), dtype=complex)
    newlabels = []

    for row1 in range(dim1):
        newrows = slice(row1 * dim2, (row1 + 1) * dim2)
        for col1 in range(dim1):
            newcols = slice(col1 * dim2, (col1 + 1) * dim2)
            newmtx[newrows, newcols] = mtx1[row1, col1] * mtx2
        for lab2 in labels2:
            newlabels.append(f"{labels1[row1]}_{lab2}_")
    return newmtx, newlabels


def getindices(labelset, labels):
    """
    Find indices in labelset whose entries contain ALL of the given label substrings.
    """
    result = None
    for lab in labels:
        hits = {i for i, s in enumerate(labelset) if lab in s}
        result = hits if result is None else result & hits
    return sorted(result) if result else []
