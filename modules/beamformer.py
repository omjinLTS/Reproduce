import numpy as np
from scipy.linalg import block_diag


class WMMSEBeamformer:
    def __init__(self, P, K, Q, E_tx):
        self.P = P
        self.K = K
        self.Q = Q
        self.E_tx = E_tx
        self.B, self.W, self.A = None, None, None

    def power_scale(self, B_raw):
        B_scaled = B_raw * np.sqrt(self.E_tx / np.sum(np.abs(B_raw)**2))
        return B_scaled

    def initialize_txmf(self, H):
        """TxMF 초기화"""
        B_init = H.conj().transpose(0, 2, 1)
        self.B = self.power_scale(B_init)
        return self.B

    def initialize_global_zfbf(self, H):
        """
        Global Zero-Forcing Beamforming (Channel Inversion)
        모든 유저의 채널을 통째로 역행렬 연산하여 간섭 제거
        조건: P >= K*Q
        """
        # 1. 전체 채널 행렬 하나로 합치기
        # H_global shape: (K*Q, P) -> (Total_Rx, Tx)
        H_global = np.vstack(H)
        
        # 2. 채널의 역행렬(Pseudo-Inverse) 계산
        # 공식: B_global = H^H * (H H^H)^-1
        try:
            # (H H^H)는 (KQ, KQ) 크기
            Inv_HH = np.linalg.inv(H_global @ H_global.conj().T)
            B_global = H_global.conj().T @ Inv_HH # (P, KQ)
        except np.linalg.LinAlgError:
            # P < KQ 이거나 채널 상태가 나빠서 역행렬이 없으면 TxMF로 대체
            B_global = H_global.conj().T

        # 3. 유저별로 쪼개기 (Reshape & Transpose)
        # (P, K*Q) -> (P, K, Q) -> (K, P, Q)
        B_final = B_global.reshape(self.P, self.K, self.Q).transpose(1, 0, 2)
        
        # 4. Power Scaling
        self.B = self.power_scale(B_final)
        return self.B

    def initialize_zfbf(self, H):
        """ZFBF 초기화"""
        B_unscaled = np.zeros((self.K, self.P, self.Q), dtype=complex)

        # for k in range(1): # User Selection
        for k in range(self.K):
            H_k = H[k]
            H_Hh = H_k @ H_k.conj().T
            Inv_H_Hh = np.linalg.inv(H_Hh)
            B_unscaled[k] = H_k.conj().T @ Inv_H_Hh
        self.B = self.power_scale(B_unscaled)
        return self.B

    def initialize_random(self):
        """Random 초기화"""
        # 랜덤한 복소수 행렬 생성
        B_rand = np.random.randn(self.K, self.P, self.Q) + \
            1j * np.random.randn(self.K, self.P, self.Q)
        self.B = self.power_scale(B_rand)
        return self.B

    def update(self, H):
        # Update A_k for each user k
        self.A = np.zeros((self.K, self.Q, self.Q), dtype=complex)
        for k in range(self.K):
            H_k, B_k = H[k], self.B[k]

            Rx_cov = np.eye(self.Q, dtype=complex)
            for i in range(self.K):
                H_kB_i = H_k @ self.B[i]
                Rx_cov += H_kB_i @ H_kB_i.conj().T

            self.A[k] = B_k.conj().T @ H_k.conj().T @ np.linalg.inv(Rx_cov)

        # Update W_k for each user k
        self.W = np.zeros((self.K, self.Q, self.Q), dtype=complex)
        for k in range(self.K):
            H_k, B_k, A_k = H[k], self.B[k], self.A[k]
            E_K = np.eye(self.Q, dtype=complex) - A_k @ H_k @ B_k
            self.W[k] = np.linalg.inv(E_K)
        # Update B
        H_global = np.vstack(H)
        A_global = block_diag(*self.A)
        W_global = block_diag(*self.W)

        mu = np.trace(W_global @ A_global @ A_global.conj().T).real / self.E_tx

        AH = A_global @ H_global
        B_new_global = AH.conj().T @ W_global @ AH + mu * np.eye(self.P, dtype=complex)
        B_new_global = np.linalg.inv(B_new_global) @ (AH.conj().T @ W_global)

        B_new = B_new_global.reshape(self.P, self.K, self.Q).transpose(1, 0, 2)
        self.B = self.power_scale(B_new)

        return self.B
