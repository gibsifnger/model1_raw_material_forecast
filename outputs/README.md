# Outputs

이 폴더는 `python src/main.py` 실행 결과로 생성되는 포트폴리오용 CSV 산출물을 보관합니다.

- `forecast_signal_result.csv`: final holdout 월별 가격 상승 확률을 BUY/WAIT 구매 검토 신호와 TP/FP/FN/TN으로 정리한 판단 테이블입니다.
- `threshold_policy_summary.csv`: 단일 threshold 정책별 구매 검토 신호 수, precision/recall/FPR, FP/FN 비용을 비교합니다.
- `regime_policy_summary.csv`: 시장 regime별 threshold 정책의 전체 성과와 regime별 신호 품질을 비교합니다.
- `false_positive_negative_cost.csv`: FP와 FN을 불필요한 구매 검토 비용, 가격 상승을 놓친 비용 관점으로 해석합니다.
- `latest_decision_signal.csv`: 가장 최신 월 기준으로 구매회의 또는 시황회의에서 검토할 BUY/WAIT 판단을 제공합니다.
- `feature_importance.csv`: 로지스틱 회귀 계수 방향과 각 feature의 구매·원가 의사결정 의미를 설명합니다.
- `backtest_summary.csv`: 학습/탐색/최종검증 크기, label 기준, 최종 성과, 선택 정책 요약을 담습니다.
