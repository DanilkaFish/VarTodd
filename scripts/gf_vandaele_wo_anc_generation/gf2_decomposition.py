import numpy as np
from typing import List, Tuple
from collections import Counter
import sys

def invert_matrix(matrix: np.matrix):
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("Matrix elements must be 0 or 1")
    
    n = matrix.shape[0]
    
    augmented = np.hstack([matrix.astype(int), np.eye(n, dtype=int)])
    
    for col in range(n):
        pivot_row = None
        for row in range(col, n):
            if augmented[row, col] == 1:
                pivot_row = row
                break
        
        if pivot_row is None:
            return None
        
        if pivot_row != col:
            augmented[[col, pivot_row]] = augmented[[pivot_row, col]]
        
        for row in range(n):
            if row != col and augmented[row, col] == 1:
                augmented[row] ^= augmented[col]
    inverse = augmented[:, n:]
    
    return inverse

def build_q_matrix(poly_coeffs):
    n = len(poly_coeffs) - 1
    Q = [[0 for _ in range(n)] for _ in range(n)]
    
    Q[0] = [poly_coeffs[n-j] % 2 for j in range(n)]
    
    for i in range(1, n):
        prev_row = Q[i-1]
        new_row = [0] * n
        
        for j in range(1, n):
            new_row[j] = prev_row[j-1]

        if prev_row[-1] == 1:
            for j in range(n):
                new_row[j] = (new_row[j] + poly_coeffs[n-j]) % 2
        Q[i] = new_row
    
    return np.array(Q).T

def gf_mult_synt_rec(A: np.matrix, B: np.matrix, C: np.matrix, D: np.matrix, ccz_list: List):
    s = A.shape[0] 
    if s == 1:
        if A.sum() > 0 and B.sum() > 0 and B.sum() > 0:
            ccz_list.append((A[0], B[0], C[0]))
        return 
    
    if s % 2 == 1:
        s += 1
        A = np.concat((A, np.zeros((1, A.shape[1]))))
        B = np.concat((B, np.zeros((1, B.shape[1]))))
        C = np.concat((C, D[:1, :]))
        D = np.concat((D[1:], np.zeros((2, D.shape[1]))))
    mid = s // 2
    AL = A[:mid]
    BL = B[:mid]
    CL = C[:mid]
    DL = D[:mid]
    AR = A[mid:]
    BR = B[mid:]
    CR = C[mid:]
    DR = D[mid:]
    gf_mult_synt_rec((AL + AR) % 2, (BL + BR) % 2, CR.copy(), DL.copy(), ccz_list )
    gf_mult_synt_rec(AR.copy(), BR.copy(), (DL + CR) % 2, (DL + DR) % 2, ccz_list )
    gf_mult_synt_rec(AL.copy(), BL.copy(), (CL + CR) % 2, (DL + CR) % 2, ccz_list )

def gf_mult_synt(p: List[int]):
    D = invert_matrix(build_q_matrix(p))
    A = np.eye(len(p) - 1)
    B = np.eye(len(p) - 1)
    C = np.eye(len(p) - 1)
    ccz_list = []
    gf_mult_synt_rec(A, B, C, D, ccz_list)
    return ccz_list

def toffoli_to_parity(ccz_list: List[Tuple]):
    res = [0] * len(ccz_list) * 7
    for i, ccz in enumerate(ccz_list):
        A, B, C = ccz
        if np.all(A == 0):
            raise KeyError()
        
        if np.all(B == 0):

            raise KeyError()
        if np.all(C == 0):

            raise KeyError()

        res[7 * i + 0] = np.concat((A, B, C))
        res[7 * i + 1] = np.concat((A, B, np.zeros_like(A)))
        res[7 * i + 2] = np.concat((A, np.zeros_like(A), C))
        res[7 * i + 3] = np.concat((np.zeros_like(A), B, C))
        res[7 * i + 4] = np.concat((A, np.zeros_like(A), np.zeros_like(A)))
        res[7 * i + 5] = np.concat((np.zeros_like(A), B, np.zeros_like(A)))
        res[7 * i + 6] = np.concat((np.zeros_like(A), np.zeros_like(A), C))
    return res

def keep_odd_frequency_vectors(vectors):
    if not vectors:
        return []
    vector_tuples = [tuple(vector.tolist()) for vector in vectors]
    frequency = Counter(vector_tuples)
    odd_frequency_tuples = [vec_tuple for vec_tuple, count in frequency.items() 
                           if count % 2 == 1]
    
    result = [np.array(vec_tuple, dtype=bool) for vec_tuple in odd_frequency_tuples if np.any(np.array(vec_tuple, dtype=bool))]
    
    return result
    
def mult_gf2(A, B):
    n = A.shape[0]
    res = [0]* n
    for i in range(n):
        res[i] = A[i,:]*B[:,i] % 2
    return np.array(res)


if __name__ == "__main__":
    # Example polynomial p(x) = x^4 + x + 1 for GF(2^4)
    if len(sys.argv) < 2:
        print("Usage: python gf_mult.py <index1> <index2> ...")
        sys.exit(1)
    
    try:
        indices = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    # Create polynomial representation
    if indices:
        max_index = max(indices)
        p = [0] * (max_index + 1)
        for index in indices:
            if index <= max_index:
                p[index] = 1
    else:
        print("No indices provided.")
    
    n = len(p)
    print(f"Creating GF(2^{n-1}) multiplication circuit")
    print(f"Polynomial representation: {p}")
    # print(mult_gf2(A, B))
    ccz_list = gf_mult_synt(p)
    par_mat = toffoli_to_parity(ccz_list)
    par_mat = keep_odd_frequency_vectors(par_mat)
    print(len(par_mat))
    name = f"gf2^{n-1}_{"".join(sys.argv[1:])}.npy"
    np.save(f"{name}", np.array(par_mat, dtype=np.bool))
    print(f"Circuit saved to: {name}")

    # print(len())
    