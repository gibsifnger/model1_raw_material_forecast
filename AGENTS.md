# AGENTS.md

## 프로젝트 역할

이 레포지토리는 원자재 가격 상승 가능성을 예측하고, 예측 확률을 구매 검토 신호로 변환하는 포트폴리오 프로젝트다.

코드 주석은 단순 머신러닝 문법 설명이 아니라, 가격 상승 리스크를 구매·SCM·원자재 소싱 의사결정으로 어떻게 연결하는지 설명해야 한다.

## Decision Row Unit

- 기본 판단 단위: 원재료 × 기준일자/월 × 예측기간 × 구매판단시점
- 현재 `src/main.py`의 실제 row 단위: 치즈 월별 가격 관측일 × 보조 원가 신호 × 향후 가격 상승 label × 구매판단시점

## 주석 작성 언어

- 모든 주석은 한국어로 작성한다.
- pandas, list, loop, import 같은 Python 문법 설명형 주석은 작성하지 않는다.
- “무엇을 계산한다”보다 “왜 이 기준이 구매 의사결정에 필요한가”를 설명한다.

## 강조해야 할 의사결정 요소

- 가격 상승 가능성 예측 확률
- 단일 `threshold` 정책
- 시장 국면별 `regime` 정책
- false positive와 false negative 비용 차이
- 구매 검토 `signal` 및 `BUY/ACCELERATE` 또는 `HOLD/WAIT` action
- 실무 적용 시 교체해야 할 데이터와 기준값

## 파일 상단 주석 기준

모든 Python 파일에는 아래 항목을 포함한다.

- `[FILE PURPOSE]`: 파일이 원자재 가격 예측 및 구매 신호 파이프라인에서 맡는 역할
- `[BUSINESS UNIT]`: 원재료 × 기준일자/월 × 예측기간 × 구매판단시점
- `[INPUT]`: 읽는 데이터, 주요 컬럼, 기준값
- `[OUTPUT]`: 예측 확률, threshold별 signal, regime별 signal, confusion matrix, FP/FN 비용 비교 결과
- `[현업 적용 시 교체 대상]`: 계약단가, 공급사 견적, CME/ICE/FOB/CIF 가격, 환율, 운임, 관세, 리드타임, 안전재고, MOQ, 계약 잔량

## 블럭 주석 기준

주요 처리 구간에는 아래 항목을 포함한 블럭 주석을 우선 적용한다.

- `[BLOCK]`
- `[현업 의미]`
- `[판단 기준]`
- `[산출물]`
- `[수정 포인트]`
- `[WHY]`
- `[ASSUMPTION]`
- `[DESIGN LOGIC]`
- `[DATA LINEAGE]`
- `[REAL DATA REPLACEMENT]`
- `[INTERVIEW CHECK]`

적용 우선순위는 데이터 로딩, 날짜 정렬, feature 생성, target 생성, 시간순 split, 모델 학습, 예측 확률 산출, threshold 정책, regime 정책, FP/FN 비용 계산, 구매 검토 signal 생성, 최종 결과 출력이다.

## 변수/컬럼 인라인 주석 기준

구매 의사결정 기준이 되는 변수와 컬럼에는 짧은 한국어 인라인 주석을 단다.

대상 예시는 아래와 같다.

- `Date`, `Price`, `corn_price`, `milk_price`
- `THRESH_UP`, `PROB_THRESHOLD`
- lag feature, rolling mean/std, month seasonality
- `y`, `proba`, `y_pred`, `regime`
- `TP`, `FP`, `FN`, `TN`
- `REVIEW_COST`, `FP_EXTRA_COST`, `FN_COST`, `total_cost`
- `decision`, `BUY/ACCELERATE`, `HOLD/WAIT`

## 조건문/분기 주석 기준

`if/else`, threshold 판단, regime 판단, action 분기에는 “왜 이 기준으로 판단하는지”를 설명한다.

예를 들어 false negative는 가격 상승을 놓치는 경우이므로 실제 구매에서는 단가 상승, 긴급 발주, 협상력 약화로 이어질 수 있다고 설명한다.

## 수정 금지 원칙

이번 주석 체계 작업에서는 아래 항목을 임의로 변경하지 않는다.

- 기존 코드 로직
- 함수명
- 파일명
- 컬럼명
- 출력 파일명
- 모델 구조
- 평가 지표 계산 방식
- threshold 기준
- regime 기준
- signal/action 분기 기준
- 데이터 파일 경로

개선이 필요해 보이는 부분은 코드에 직접 반영하지 않고 최종 요약의 개선 제안으로만 남긴다.

## README 원칙

README는 이번 주석 체계 변환 작업에서 수정하지 않는다.

README에 보강하면 좋을 내용이 보이면 최종 요약의 README 개선 제안에만 적는다.

## 검증 명령

주석 수정 후 아래 명령어로 기존 파이프라인이 정상 실행되는지 확인한다.

```bash
python src/main.py
```

실행 실패 시 코드 로직을 변경하지 말고 실패 원인과 필요한 조치만 요약한다.
