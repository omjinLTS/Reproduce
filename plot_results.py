import glob
import os
import sys
from modules import load_simulation_data, plot_simulation_results

def main():
    # 1. 가장 최근에 생성된 결과 파일(.pkl) 자동 검색
    # (일일이 파일명 치기 귀찮으니까 자동화)
    list_of_files = glob.glob('results_*.pkl') 
    
    if not list_of_files:
        print("❌ Error: 저장된 결과 파일(*.pkl)을 찾을 수 없습니다.")
        print("   먼저 'run_simulation.py'를 실행해서 데이터를 생성해주세요.")
        sys.exit()
        
    # 생성 시간 기준 가장 최신 파일 선택
    target_file = max(list_of_files, key=os.path.getctime)
    print(f"📂 데이터 파일 로드 중: {target_file}")

    # 2. 데이터 불러오기 (모듈 활용)
    # results: { 'TxMF': [..], 'WMMSE': [..] }
    # params: { 'P': 4, 'K': 4, ... }
    results, params = load_simulation_data(target_file)

    # 3. 그래프 제목 및 라벨 설정
    # 파라미터 정보를 제목에 자동으로 넣어주면 나중에 구별하기 좋음
    plot_title = f"Sum-Rate Performance (P={params['P']}, K={params['K']}, Q={params['Q']})"
    
    print(f"📊 그래프 그리는 중... (SNR 범위: {params['SNR_RANGE']})")

    # 4. Plotter 모듈 호출
    plot_simulation_results(
        x_axis=params['SNR_RANGE'], 
        results_dict=results, 
        title=plot_title,
        x_label="SNR [dB]",
        y_label="Sum-Rate [bits / complex dim.]"
    )

if __name__ == "__main__":
    main()