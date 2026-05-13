# main_2.py 최종 통합 구조표

| 실행 단계(v1) | 구조 층(v2) | 블록 | 역할 한 줄 요약 | 구분 |
|---|---|---|---|---|
| 0단계 | A. 설정층 | `0) 설정` | 데이터 경로, 라벨 모드, lag, test 비율, 기본 threshold 등 전역 규칙 선언 | helper/전역 |
| 0단계 | A. 설정층 | `Cost Function 설정값` | 리뷰 비용, FP 추가비용, FN 비용 정의 | helper/전역 |
| 0단계 | A. 설정층 | `출력 가독성 설정` | 출력 미리보기 개수 등 표시용 설정 | helper/전역 |
| 0단계 | A. 설정층 | `Threshold Search 설정값` | threshold 탐색 시 hard/soft 조건 기준 정의 | helper/전역 |
| 1단계 | B. 계산 도구층 | `print_section`, `print_df_compact` | 출력 정리용 보조 함수 | helper |
| 1단계 | B. 계산 도구층 | `get_label_description`, `get_forecast_header` | 라벨 의미/출력 제목 설명 함수 | helper |
| 1단계 | B. 계산 도구층 | `walk_forward_validate` | 시계열 walk-forward 안정성 점검 함수 | helper |
| 1단계 | B. 계산 도구층 | `1. 기본 성능지표 계산` | TP/FP/FN/TN, precision, recall, fpr 계산 | helper |
| 1단계 | B. 계산 도구층 | `2. 연간 signal 수 계산` | 월별 signal을 연환산 빈도로 변환 | helper |
| 1단계 | B. 계산 도구층 | `3. regime별 threshold 적용` | regime 1, 2에 다른 threshold 적용 | helper |
| 1단계 | B. 계산 도구층 | `4. regime별 성능 계산` | regime별 precision/recall/fpr/signal 계산 | helper |
| 1단계 | B. 계산 도구층 | `5. 소프트조건 점수 계산` | precision, fpr, regime 균형을 합쳐 점수화 | helper |
| 1단계 | B. 계산 도구층 | `5-1. cost function 계산` | signal, FP, FN을 비용으로 환산 | helper |
| 1단계 | B. 계산 도구층 | `5-2. decision simulation (single threshold)` | 단일 threshold 정책의 운영 결과 시뮬레이션 | helper |
| 1단계 | B. 계산 도구층 | `5-3. decision simulation (regime policy)` | regime별 threshold 정책 결과 시뮬레이션 | helper |
| 1단계 | B. 계산 도구층 | `6. threshold 조합 1개 평가` | 한 쌍의 regime threshold 조합 평가 | helper |
| 1단계 | B. 계산 도구층 | `7. 전체 threshold grid search` | 여러 threshold 조합을 전부 탐색 | helper |
| 1단계 | B. 계산 도구층 | `8. holdout 분리 / regime leakage 방지용 helper` | search/final 분리, regime leakage 방지 처리 | helper |
| 2단계 | C. 데이터 준비층 | `1) 데이터 로드 & 정렬` | cheese/corn/milk 데이터 로드, 월 단위 정렬/병합 | 실행 |
| 3단계 | C. 데이터 준비층 | `2) 수익률 & 라벨(y) 만들기` | 수익률 계산 후 y 생성 | 실행 |
| 4단계 | C. 데이터 준비층 | `3) lag 피처 만들기` | 과거 수익률 lag 변수 생성 | 실행 |
| 4단계 | C. 데이터 준비층 | `3-1) 추가 시계열 피처 만들기` | 모멘텀, 변동성, 이동평균, spread 등 생성 | 실행 |
| 4단계 | C. 데이터 준비층 | `4) 계절성 피처 (월)` | month_sin, month_cos 생성 | 실행 |
| 5단계 | C. 데이터 준비층 | `5) 결측 제거` | 학습 가능한 최종 테이블 `df_model` 확정 | 실행 |
| 6단계 | D. 학습/평가 분리층 | `6) 시간순 Train / Holdout / Search / Final 분리` | 학습/정책탐색/최종평가 구간 분리 | 실행 |
| 7단계 | E. 모델 학습층 | `7) 모델 학습 (Logistic Regression)` | 확률 모델 학습 | 실행 |
| 7단계 | E. 모델 학습층 | `7-1) Logistic 계수 확인` | 어떤 피처가 모델에 크게 작용하는지 확인 | 실행 |
| 8단계 | F. 단일 정책 탐색층 | `8) 예측 & 정책 탐색 (search holdout)` | search holdout에서 확률 예측 및 threshold sweep | 실행 |
| 8단계 | F. 단일 정책 탐색층 | `8-1) 운영용 threshold 추천 (search holdout)` | 운영형/비용형 threshold 후보 선택 | 실행 |
| 9단계 | F. 단일 정책 탐색층 | `8-1-1) Decision Simulation: final holdout single policies` | 선택한 single threshold들을 final holdout에서 검증 | 실행 |
| 10단계 | G. 검증/진단층 | `8-2) Walk-forward evaluation` | 시계열적으로 성능이 얼마나 흔들리는지 확인 | 실행 |
| 10단계 | G. 검증/진단층 | `8-3) Correlation / Autocorrelation Check` | 피처 구조, 자기상관, 단순 상관 진단 | 실행 |
| 11단계 | H. Regime 실험 및 최종 운영결정층 | `9) Regime Detection (KMeans, train fit only)` | train 기준으로 regime 생성 후 search/final에 부착 | 실행 |
| 12단계 | H. Regime 실험 및 최종 운영결정층 | `10) Search holdout에서 regime 성능 / threshold 탐색` | regime별 성능 상태를 먼저 점검 | 실행 |
| 12단계 | H. Regime 실험 및 최종 운영결정층 | `10-1) search holdout에서 regime threshold search` | regime별 threshold 조합 탐색 | 실행 |
| 13단계 | H. Regime 실험 및 최종 운영결정층 | `10-2) Final holdout에서 single vs regime 비교` | single 정책과 regime 정책 최종 비교 | 실행 |
| 14단계 | H. Regime 실험 및 최종 운영결정층 | `11) 마지막 달 기준 forecast 출력` | 최신월 확률과 최종 액션 출력 | 실행 |

---

## 구조 읽는 법

이 표는 두 축으로 읽는다.

### 1. 구조 축(v2)
- A. 설정층
- B. 계산 도구층
- C. 데이터 준비층
- D. 학습/평가 분리층
- E. 모델 학습층
- F. 단일 정책 탐색층
- G. 검증/진단층
- H. Regime 실험 및 최종 운영결정층

### 2. 실행 축(v1)
- 0단계: 전역 룰 선언
- 1단계: 계산 도구 준비
- 2~5단계: 데이터 준비
- 6단계: 학습/평가 구간 분리
- 7단계: 모델 학습
- 8~9단계: 단일 정책 탐색 및 검증
- 10단계: 진단
- 11~13단계: regime 실험 및 비교
- 14단계: 마지막 운영 신호 출력

---

## 핵심 해석

이 파일은 단순히 예측 모델만 만드는 파일이 아니다.

즉, 아래 순서로 읽는 것이 맞다.

1. 데이터와 라벨을 만든다.
2. 피처를 만든다.
3. 시간순으로 train/search/final을 분리한다.
4. train으로 확률 모델을 학습한다.
5. search holdout에서 threshold 정책을 탐색한다.
6. final holdout에서 그 정책을 검증한다.
7. 필요하면 regime 정책까지 실험한다.
8. 마지막 달에 실제 운영 액션을 출력한다.

---

## 중심 축

### 중심축 1. 라벨 정의
`2) 수익률 & 라벨(y) 만들기`

이 블록에서 예측 대상이 결정된다.  
지금 파일 전체는 이 라벨 정의 위에 올라가 있다.

### 중심축 2. 시간순 분리
`6) 시간순 Train / Holdout / Search / Final 분리`

정책 탐색 구간과 최종 평가 구간을 분리하는 핵심이다.  
이게 흔들리면 결과 해석도 흔들린다.

### 중심축 3. 정책 탐색 → 최종 검증
`8) ~ 10-2)`

이 파일의 진짜 목적은 모델 점수표가 아니라  
운영 가능한 구매 의사결정 정책을 찾고 검증하는 것이다.