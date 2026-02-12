import numpy as np

# Rayleigh Channel Module


def rayleigh_channel(P, K, Q, snr_db):
    channel_variance = 10 ** (snr_db / 10.0)

    real_part = np.random.randn(K, Q, P)
    imag_part = np.random.randn(K, Q, P)

    scale_factor = np.sqrt(channel_variance / 2.0)

    H = scale_factor * (real_part + 1j * imag_part)

    return H


if __name__ == "__main__":
    # 테스트 코드
    Q = 2  # 사용자 당 수신 안테나 수
    P = 4  # 송신 안테나 수
    K = 4  # 사용자 수
    snr_db = 10  # 예시 SNR 값 (dB)

    H = rayleigh_channel(P, K, Q, snr_db)
    print("Generated Channel H shape:", H.shape)
    print("Channel H:", H)
