import argparse
import glob
import os
import sys
from modules import load_simulation_data, plot_simulation_results


def main():
    parser = argparse.ArgumentParser(description="결과 pkl 파일을 시각화합니다.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="사용할 pkl 파일 경로 (생략하면 최신 파일 자동 선택)"
    )
    args = parser.parse_args()

    # 사용할 파일 결정
    if args.file:
        target_file = args.file
        if not os.path.exists(target_file):
            print(f"❌ Error: 파일을 찾을 수 없습니다: {target_file}")
            sys.exit(1)
    else:
        list_of_files = glob.glob('results_*.pkl')

        if not list_of_files:
            print("❌ Error: 저장된 결과 파일(*.pkl)을 찾을 수 없습니다.")
            print("   먼저 'run_simulation.py'를 실행해서 데이터를 생성해주세요.")
            sys.exit(1)

        target_file = max(list_of_files, key=os.path.getctime)

    print(f"📂 데이터 파일 로드 중: {target_file}")

    results, params = load_simulation_data(target_file)

    plot_title = f"Sum-Rate Performance (P={params['P']}, K={params['K']}, Q={params['Q']})"

    print(f"📊 그래프 그리는 중... (SNR 범위: {params['SNR_RANGE']})")

    plot_simulation_results(
        x_axis=params['SNR_RANGE'],
        results_dict=results,
        title=plot_title,
        x_label="SNR [dB]",
        y_label="Sum-Rate [bits / complex dim.]"
    )


if __name__ == "__main__":
    main()
