import matplotlib.pyplot as plt
import scienceplots # 이걸 import 해야 스타일이 등록됨

def plot_simulation_results(x_axis, results_dict, title, x_label="SNR [dB]", y_label="Sum-Rate [bits/s/Hz]"):
    """
    SciencePlots 라이브러리를 활용한 논문용 고퀄리티 그래프 출력
    """
    
    # 'science' : 기본 과학 논문 스타일
    # 'ieee'    : IEEE Transaction 스타일 (통신 분야 국룰)
    # 'no-latex': LaTeX가 안 깔려 있어도 돌아가게 함 (깔려 있으면 이 항목 제거 추천)
    with plt.style.context(['science', 'ieee', 'no-latex']):
        
        plt.figure(figsize=(8, 6)) # IEEE 표준보다는 조금 크게 (발표용)
        
        # 논문에서 자주 쓰는 마커와 선 스타일 조합
        markers = ['o', 's', '^', 'D', 'v', 'x']
        linestyles = ['-', '--', '-.', ':', '-', '--']
        
        for i, (algo_name, y_data) in enumerate(results_dict.items()):
            plt.plot(
                x_axis, 
                y_data, 
                label=algo_name, 
                marker=markers[i % len(markers)], 
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,     # 선 굵기 적당히
                markersize=6,      # 마커 크기
                clip_on=False      # 마커가 그래프 밖으로 나가도 잘리지 않게
            )

        # 제목과 축 레이블 (LaTeX 수식 문법 사용 가능)
        # 예: Sum-Rate [bits/s/Hz] -> Sum-Rate [bits/s/Hz]
        plt.title(title, fontsize=12, pad=15)
        plt.xlabel(x_label, fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        
        # 축 범위 설정 (데이터에 맞게 자동 조정되지만, 여유를 둠)
        plt.autoscale(enable=True, axis='x', tight=True)
        
        # 범례 위치 (가장 빈 공간에 자동 배치)
        plt.legend(frameon=True, fontsize=9, loc='best')
        
        # 그리드 (IEEE 스타일은 보통 그리드를 연하게 넣음)
        plt.grid(True, which='major', linestyle=':', alpha=0.5)

        # 저장 및 출력
        save_name = title.replace(" ", "_").replace("=", "").replace(",", "") + ".png"
        
        # dpi=300 이상이어야 논문 제출용 고해상도
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프가 고해상도 '{save_name}'로 저장되었습니다.")
        
        plt.show()