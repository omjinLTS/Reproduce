import parameters as p
from modules import (
    rayleigh_channel, WMMSEBeamformer,
    calculate_sum_rate, save_simulation_data
)
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed  # 핵심 라이브러리

# --- 병렬 처리를 위해 '1회 시행'을 함수로 분리 ---


def run_single_iteration(snr_db, P, K, Q, E_TX, MAX_ITER):
    """
    몬테카를로 1회 시행에 대한 로직
    반환값: (rate_txmf, rate_wmmse1, rate_wmmse2) 튜플
    """
    # 1. 독립적인 Solver 객체 생성 (프로세스 충돌 방지)
    solver = WMMSEBeamformer(P, K, Q, E_TX)

    # 2. 채널 생성
    H = rayleigh_channel(P, K, Q, snr_db)

    # ------------------------------- Curve 1: TxMF ------------------------------ #
    B_txmf = solver.initialize_txmf(H)
    r_txmf = calculate_sum_rate(H, B_txmf)

    # ------------------------------- Curve 2: ZFBF ------------------------------ #
    B_zfbf = solver.initialize_zfbf(H)
    r_zfbf = calculate_sum_rate(H, B_zfbf)

    # ------------------- Curve 3: WMMSE2 (TxMF Init + 10 Iter) ------------------ #
    solver.B = B_txmf.copy()
    for _ in range(10):
        solver.update(H)
    r_wmmse2 = calculate_sum_rate(H, solver.B)

    # -------------- Curve 4: WMMSE1 (10 Random Init + Convergence) -------------- #
    best_rate = 0
    NUM_RAND_INIT = 10
    MAX_CONV_ITER = 50
    EPSILON = 1e-3

    for _ in range(NUM_RAND_INIT):
        solver.initialize_random()
        prev_rate = 0
        for _ in range(MAX_CONV_ITER):
            solver.update(H)
            curr_rate = calculate_sum_rate(H, solver.B)
            if abs(curr_rate - prev_rate) < EPSILON:
                break
            prev_rate = curr_rate
        if curr_rate > best_rate:
            best_rate = curr_rate

    r_wmmse1 = best_rate

    return r_txmf, r_wmmse1, r_wmmse2, r_zfbf


# --- Main 실행 부분 ---
if __name__ == "__main__":  # 윈도우/리눅스 멀티프로세싱 필수 구문

    results = {
        "TxMF": [],
        "ZFBF": [],
        "WSRBF-WMMSE1 (convergence/10 random init)": [],
        "WSRBF-WMMSE2 (10 iterations - TxMF)": []
    }

    current_params = {"P": p.P, "K": p.K, "Q": p.Q,
                      "SNR_RANGE": list(p.SNR_DB_RANGE)}

    print(f"🚀 Simulation Start (Parallel): P={p.P}, K={p.K}, Q={p.Q}")
    print(f"   Using ALL Available CPU Cores...")

    # SNR 루프는 순차적으로
    for snr_db in tqdm(p.SNR_DB_RANGE, desc="SNR Loop"):

        parallel_results = Parallel(n_jobs=-1)(
            delayed(run_single_iteration)(
                snr_db, p.P, p.K, p.Q, p.E_TX, p.MAX_ITERATIONS
            ) for _ in range(p.MONTE_CARLO_RUNS)
        )

        # parallel_results는 [(r1, r2, r3), (r1, r2, r3), ...] 형태
        r_txmf_list, r_wmmse1_list, r_wmmse2_list, r_zfbf_list = zip(*parallel_results)

        # 평균 계산 및 저장
        results["TxMF"].append(np.mean(r_txmf_list))
        results["ZFBF"].append(np.mean(r_zfbf_list))
        results["WSRBF-WMMSE1 (convergence/10 random init)"].append(
            np.mean(r_wmmse1_list))
        results["WSRBF-WMMSE2 (10 iterations - TxMF)"].append(np.mean(r_wmmse2_list))

    # 저장
    save_simulation_data(results, current_params)
