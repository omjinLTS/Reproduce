# parameters.py

# --- System Configuration ---
P = 4       # 송신 안테나 수
K = 4       # 사용자 수
Q = 1       # 사용자 당 수신 안테나 수 (논문 Fig. 1, 4 등 시나리오에 따라 변경)
E_TX = 1.0  # 총 송신 전력

# --- Simulation Settings ---
# 논문의 X축 범위에 맞춰 설정 (-10dB ~ 30dB)
SNR_DB_RANGE = range(-10, 31, 5)

MONTE_CARLO_RUNS = 1000  # 몬테카를로 반복 횟수 (채널 realization 수)
MAX_ITERATIONS = 10      # WMMSE 알고리즘 내부 반복 횟수 (WSRBF-WMMSE2)
