import numpy as np
from typing import List

def verify_representations(Q: np.ndarray, A: List[np.ndarray], B: List[np.ndarray], C: List[np.ndarray]) -> bool:
    """
    Verify that the two representations of the boolean polynomial are equivalent.
    
    Parameters:
    -----------
    Q : np.ndarray
        (n-1) x n matrix for transforming z to z'
    A, B, C : List[np.ndarray]
        Lists of coefficient vectors for representation 1
    
    Returns:
    --------
    bool : True if representations are equivalent for all boolean vectors x,y,z
    """
    # Get dimensions from Q
    n_minus_1, n = Q.shape
    if n_minus_1 != n:
        raise ValueError(f"Q should be (n-1) x n, got {Q.shape}")
    
    # Get number of terms in representation 1
    m = len(A)
    if len(B) != m or len(C) != m:
        raise ValueError(f"A, B, C should have same length, got {len(A)}, {len(B)}, {len(C)}")
    
    # Check vector dimensions
    for i in range(m):
        if A[i].shape != (n,):
            raise ValueError(f"A[{i}] should have shape ({n},), got {A[i].shape}")
        if B[i].shape != (n,):
            raise ValueError(f"B[{i}] should have shape ({n},), got {B[i].shape}")
        if C[i].shape != (n,):
            raise ValueError(f"C[{i}] should have shape ({n},), got {C[i].shape}")
    
    def representation1(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> int:
        """Compute f(x,y,z) using representation 1."""
        result = 0
        for i in range(m):
            # Dot products mod 2 (using XOR for boolean)
            term = (np.dot(A[i], x) % 2) * (np.dot(B[i], y) % 2) * (np.dot(C[i], z) % 2)
            result = (result + term) 
        return result % 2
    
    def representation2(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> int:
        """Compute f(x,y,z) using representation 2."""
        # Compute z' = Qz mod 2
        z_prime = (Q @ z) % 2
        
        # Compute g(x,y,z)
        g_val = 0
        for i in range(n):
            for j in range(i + 1):  # j from 0 to i
                term = (x[j] * y[i - j] * z[i]) % 2
                g_val = (g_val + term) % 2
        
        # Compute h(x,y,z)
        h_val = 0
        for i in range(n - 1):  # i from 0 to n-2
            for j in range(i + 1, n):  # j from i+1 to n-1
                # Note: y index should be n + i - j, but y has length n
                # We need to check bounds
                y_idx = n + i - j
                if 0 <= y_idx < n:
                    term = (x[j] * y[y_idx] * z_prime[i]) % 2
                    h_val = (h_val + term) % 2
        
        return (g_val + h_val) % 2
    
    # Test with all possible boolean vectors up to a reasonable size
    # For n > 6, exhaustive testing becomes too expensive (2^(3n) cases)
    # We'll use randomized testing for larger n
    
    max_exhaustive_n = 4  # Can test exhaustively for n <= 4 (2^12 = 4096 cases)
    
    if n <= max_exhaustive_n:
        # Exhaustive testing for small n
        for x_val in range(2**n):
            x = np.array([(x_val >> k) & 1 for k in range(n)], dtype=int)
            for y_val in range(2**n):
                y = np.array([(y_val >> k) & 1 for k in range(n)], dtype=int)
                for z_val in range(2**n):
                    z = np.array([(z_val >> k) & 1 for k in range(n)], dtype=int)
                    
                    val1 = representation1(x, y, z)
                    val2 = representation2(x, y, z)
                    
                    if val1 != val2:
                        print(f"Mismatch found:")
                        print(f"x = {x}")
                        print(f"y = {y}")
                        print(f"z = {z}")
                        print(f"Representation 1: {val1}")
                        print(f"Representation 2: {val2}")
                        return False
        return True
    else:
        # Randomized testing for larger n
        num_tests = min(10000, 2**(3*n) // 100)  # Test a reasonable number of cases
        
        print(f"n = {n} is too large for exhaustive testing.")
        print(f"Performing {num_tests} random tests...")
        
        for test in range(num_tests):
            x = np.random.randint(0, 2, size=n)
            y = np.random.randint(0, 2, size=n)
            z = np.random.randint(0, 2, size=n)
            
            val1 = representation1(x, y, z)
            val2 = representation2(x, y, z)
            
            if val1 != val2:
                print(f"Mismatch found in test {test + 1}:")
                print(f"x = {x}")
                print(f"y = {y}")
                print(f"z = {z}")
                print(f"Representation 1: {val1}")
                print(f"Representation 2: {val2}")
                return False
        
        print(f"All {num_tests} random tests passed.")
        return True

# Example usage
if __name__ == "__main__":
    from gf2_decomposition import gf_mult_synt, build_q_matrix, invert_matrix
    p = [1,1,0,0,1]
    Q = invert_matrix(build_q_matrix(p))
    ccz_list = gf_mult_synt(p)
    A = [ccz[0] for ccz in ccz_list] 
    B = [ccz[1] for ccz in ccz_list] 
    C = [ccz[2] for ccz in ccz_list] 
    print(verify_representations(Q, A, B, C))