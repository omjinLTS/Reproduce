import numpy as np

def calculate_sum_rate(H, B, sigma_sq=1.0):
    """
    R = sum_k log2(det(I + SINR_k))
    SINR_k = H_k B_k B_k^H H_k^H (I + sum_{j!=k} H_k B_j B_j^H H_k^H)^-1
    """
    K, Q, P = H.shape
    sum_rate = 0.0
    
    for k in range(K):
        H_k = H[k]
        B_k = B[k]
        
        # 신호 성분 (Signal)
        Signal = H_k @ B_k @ B_k.conj().T @ H_k.conj().T
        
        # 간섭 + 잡음 성분 (Interference + Noise)
        Int_Noise = np.eye(Q, dtype=complex) * sigma_sq
        for j in range(K):
            if j != k:
                Int_Signal = H_k @ B[j] @ B[j].conj().T @ H_k.conj().T
                Int_Noise += Int_Signal
                
        # Rate = log2(det(I + Signal * Int_Noise^-1))
        #      = log2(det(Int_Noise + Signal)) - log2(det(Int_Noise))
        
        Total_Cov = Int_Noise + Signal
        
        rate_k = np.log2(np.linalg.det(Total_Cov).real) - np.log2(np.linalg.det(Int_Noise).real)
        sum_rate += rate_k
        
    return sum_rate