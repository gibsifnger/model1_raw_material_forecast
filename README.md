# Raw Material Price Forecast Pipeline

원자재 가격 상승 가능성을 예측하고, 예측 확률을 구매 의사결정 신호로 변환하는 실험형 파이프라인입니다.

## 목적

단순히 가격을 예측하는 것이 아니라, 원자재 가격 상승 가능성을 바탕으로 다음과 같은 구매 의사결정 질문에 답하는 것을 목표로 합니다.

- 가격 상승 가능성이 높은가?
- 어느 확률 이상일 때 구매 검토 신호를 줄 것인가?
- 단일 threshold 정책과 시장 국면별 regime 정책 중 어떤 방식이 더 안정적인가?
- false positive, false negative, signal 수를 비용 관점에서 어떻게 비교할 것인가?

## 구조

```text
src/main.py      # 최종 실행 코드
data/            # cheese, corn, milk 가격 데이터
scripts/         # 외부 데이터 다운로드 스크립트
docs/            # 모델 구조 설명 문서
```

## Outputs

`python src/main.py` 실행 후 `outputs/` 폴더에 포트폴리오 PDF 제작용 CSV가 생성됩니다. 모든 CSV는 엑셀에서 한글이 깨지지 않도록 UTF-8-SIG로 저장됩니다.

- `forecast_signal_result.csv`: final holdout 월별 가격 상승 확률을 BUY/ACCELERATE 또는 HOLD/WAIT 구매 검토 신호로 변환한 최종 판단 테이블입니다.
- `threshold_policy_summary.csv`: 단일 threshold 정책별 신호 수, TP/FP/FN/TN, precision/recall/FPR, FP/FN 비용을 비교합니다.
- `regime_policy_summary.csv`: 시장 regime별 threshold 정책의 전체 성과와 regime별 신호 품질을 비교합니다.
- `false_positive_negative_cost.csv`: FP를 불필요한 조기 구매 검토 비용, FN을 가격 상승을 놓친 비용 관점으로 해석한 요약표입니다.
- `latest_decision_signal.csv`: 가장 최신 월 기준 구매 검토 대상 여부와 회의 활용 문구를 제공합니다.
- `feature_importance.csv`: LogisticRegression 계수 방향과 각 feature가 구매·원가 판단에서 갖는 의미를 정리합니다.
- `backtest_summary.csv`: 학습/정책탐색/최종검증 크기, label 기준, 최종 성과, 선택 정책 요약을 담습니다.
