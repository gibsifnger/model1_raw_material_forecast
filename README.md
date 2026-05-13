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