"""
[FILE PURPOSE]
- 치즈 가격 상승 가능성을 과거 원자재 가격 흐름으로 예측하고, 예측 확률을 구매 검토 신호로 변환하는 단일 실행 파이프라인이다.
- 단순히 가격 방향을 맞히는 모델이 아니라, threshold 정책, regime 정책, FP/FN 비용 비교를 통해 구매·원자재 소싱 의사결정 기준을 점검한다.

[BUSINESS UNIT]
- 치즈 원재료 × 월별 가격 관측일 × 1~3개월 target horizon × 구매판단시점
- 코드상 row는 월 단위 치즈 가격 관측치를 기준으로 옥수수·우유 가격 feature와 향후 가격 상승 label을 결합한 판단 단위다.

[INPUT]
- data/cheese_price.csv: Date, Price 기준 치즈 가격 시계열
- data/corn_price.csv: Date, corn_price 기준 옥수수 가격 시계열
- data/milk_price.csv: Date, milk_price 기준 우유 가격 시계열
- 전역 기준값: LABEL_MODE, THRESH_UP, PROB_THRESHOLD, regime threshold grid, FP/FN 비용 가중치

[OUTPUT]
- 콘솔 출력: 데이터 분할 결과, Logistic 계수, threshold sweep, single/regime 정책 비교, confusion matrix, walk-forward 결과, 마지막 월 구매 판단
- 모델 산출 컬럼: y, proba, y_pred, threshold, regime, TP/FP/FN/TN, total_cost, BUY/ACCELERATE 또는 HOLD/WAIT 판단
- 모델 학습 전 확인용 파일: 모델1_학습전.csv

[현업 적용 시 교체 대상]
- 공개 예시 가격 데이터는 실제 원재료 계약단가, 공급사 견적, CME/ICE/FOB/CIF 가격, 환율, 운임, 관세, 리드타임, 안전재고, 발주 MOQ, 기존 계약 잔량으로 교체해야 한다.
- threshold와 비용 가중치는 회사의 구매 정책, 재고 자금 부담, 긴급 발주 비용, 공급사 협상 리드타임 기준으로 재검토해야 한다.
"""

import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib

# KMeans 내부 병렬 처리기가 Windows CPU 정보를 탐지하다가 경고를 내도 구매 판단 결과에는 영향이 없으므로 실행 로그를 안정화한다.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# 콘솔 실행 환경에 GUI가 없어도 확률 분포 진단 단계가 구매 판단 파이프라인을 막지 않도록 비대화형 백엔드를 사용한다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss

"""
A. 설정층 운영 원칙/판정 기준 정의
모델 label, threshold, 비용 가중치, 출력 요약 범위를 한곳에 모아 구매 판단 정책의 전역 룰을 고정한다.
"""
# ============================================================
# [BLOCK] 구매 판단 정책 기준값 정의
# [현업 의미] 원재료 가격 상승 리스크를 어떤 horizon, threshold, 비용 기준으로 구매 검토 신호화할지 정한다.
# [판단 기준] 원재료 가격 파일, 월별 가격 컬럼, 상승률 label, holdout 비율, 확률 threshold, FP/FN 비용 가중치
# [산출물] 전체 파이프라인이 참조하는 상수와 컬럼명
# [수정 포인트] 실무 적용 시 데이터 경로, 가격 컬럼, horizon, 상승률 기준, 비용 가중치를 회사 정책에 맞춘다.
# [WHY] 구매 의사결정 기준이 코드 곳곳에 흩어지면 threshold와 비용 정책을 설명하기 어려우므로 전역 설정으로 고정한다.
# [ASSUMPTION] 치즈를 중심 원재료로 두고 옥수수·우유 가격을 보조 원가 신호로 사용한다고 가정한다.
# [DESIGN LOGIC] 데이터 정의, label 정의, 기간 분할, 비용 가중치를 먼저 고정한 뒤 모든 정책 비교가 같은 기준을 참조하게 한다.
# [DATA LINEAGE] data/*.csv 입력값이 df_model, threshold sweep, final decision 출력으로 이어진다.
# [REAL DATA REPLACEMENT] ERP 계약단가, 공급사 견적, 거래소 지표, 환율·운임·관세, MOQ·리드타임 기준값과 연결한다.
# [INTERVIEW CHECK] 이 모델은 예측값보다 구매 정책 기준을 명시적으로 관리하는 구조라고 설명할 수 있어야 한다.
# ============================================================
CSV_PATH = "data/cheese_price.csv"  # 구매 검토 대상인 치즈 원재료 가격 시계열
CORN_CSV_PATH = "data/corn_price.csv"  # 치즈 원가 압력에 영향을 줄 수 있는 옥수수 가격 시계열
MILK_CSV_PATH = "data/milk_price.csv"  # 치즈 원가의 핵심 투입재인 우유 가격 시계열

DATE_COL = "Date"  # 원재료 가격 관측 월을 맞추는 기준일 컬럼
PRICE_COL = "Price"  # 치즈 구매단가 상승 리스크 판단의 기준 가격 컬럼
CORN_PRICE_COL = "corn_price"  # 보조 원가 신호로 사용하는 옥수수 가격 컬럼
MILK_PRICE_COL = "milk_price"  # 보조 원가 신호로 사용하는 우유 가격 컬럼

THRESH_UP = 0.03  # 향후 가격이 이 상승률 이상이면 구매단가 상승 이벤트로 보는 기준
HORIZON = 1  # 기본 예측기간 기준값이며 label mode에 따라 1개월 또는 3개월 판단으로 해석된다.

LAGS = [1, 2, 3, 6]             # 치즈 가격 흐름을 구매 판단에 반영하는 과거 월수
CORN_LAGS = [2, 3, 4, 6]        # 옥수수 가격 변동이 치즈 가격에 뒤늦게 반영될 가능성을 보는 lag 기준
MILK_LAGS = [1, 2, 3, 6]        # 우유 가격 변동이 치즈 가격 리스크로 이어지는 시차 기준
TEST_RATIO = 0.2  # 최근 구간을 정책 검증용 holdout으로 남기는 비율
POLICY_SEARCH_RATIO = 0.5  # holdout 중 앞쪽은 threshold 탐색, 뒤쪽은 최종 평가

PROB_THRESHOLD = 0.6  # 이 확률 이상이면 가격 상승 리스크가 있다고 보고 구매 검토 신호를 발생시키는 기본 기준

LABEL_MODE = "max_3m"  # 향후 3개월 중 한 번이라도 +3% 이상 상승하면 상승 이벤트로 보는 운영 label
# "strict_1m" : 다음 달 +3% 이상
# "cum_3m"    : 3개월 뒤 누적수익률 +3% 이상
# "max_3m"    : 앞으로 3개월 안에 한 번이라도 +3% 이상

REVIEW_COST = 1.0  # 구매 검토 신호 1건이 만드는 분석·승인·실행 부담 가중치
FP_EXTRA_COST = 1.5  # 불필요한 조기 구매 검토로 발생하는 재고·자금 부담 가중치
FN_COST = 4.0  # 가격 상승을 놓쳤을 때 실제 구매단가 상승과 협상력 약화로 이어질 수 있는 리스크 가중치

COEF_TOP_N = 10
THRESHOLD_PREVIEW_N = 10
REGIME_SEARCH_PREVIEW_N = 10
OUTPUT_DIR = Path("outputs")  # 포트폴리오 PDF 제작에 필요한 구매 판단 CSV를 모으는 산출물 폴더

# A-5. 출력 helper 1: print_section
def print_section(title: str):
    print(f"\n===== {title} =====")

# A-6. 출력 helper 2: print_df_compact(데이터프레임을 요약된 형태로 예쁘게 출력한다.)
def print_df_compact(df: pd.DataFrame, title: str, max_rows: int = 10):
    print_section(title)
    if df is None or len(df) == 0:
        print("(empty)")
        return

    preview = df.head(max_rows).copy()
    print(preview.to_string(index=False))

    if len(df) > max_rows:
        print(f"... ({len(df) - max_rows} more rows)")

# A-7. 라벨 설명 helper: get_label_description
   # 설정값으로 정의된 LABEL_MODE를 사람이 읽을 수 있는 설명문으로 바꾼다.
def get_label_description(label_mode: str, thresh_up: float) -> str:
    pct = int(thresh_up * 100)

    if label_mode == "strict_1m":
        return f"next month >= +{pct}% up"
    elif label_mode == "cum_3m":
        return f"next 3m cumulative return >= +{pct}%"
    elif label_mode == "max_3m":
        return f"at least one +{pct}% rise within next 3m"
    else:
        raise ValueError("LABEL_MODE must be one of: strict_1m, cum_3m, max_3m")

# A-8. 출력 제목 helper: get_forecast_header(라벨 모드에 따라 출력 제목을 맞춰준다.)
def get_forecast_header(label_mode: str) -> str:
    if label_mode == "strict_1m":
        return "===== Next Month Forecast ====="
    elif label_mode == "cum_3m":
        return "===== Next 3M Cumulative Forecast ====="
    elif label_mode == "max_3m":
        return "===== Next 3M Max-Move Forecast ====="
    else:
        raise ValueError("LABEL_MODE must be one of: strict_1m, cum_3m, max_3m")

# ============================================================
# [BLOCK] Walk-forward 시계열 안정성 검증
# [현업 의미] 한 번의 holdout 성과가 아니라 여러 구매판단시점에서 신호 품질이 유지되는지 점검한다.
# [판단 기준] 시간순 학습 구간, step 단위 평가 구간, PROB_THRESHOLD, AUC, precision, recall, TP/FP/FN/TN
# [산출물] 시점별 walk-forward 성과 테이블
# [수정 포인트] 실무 적용 시 월별 구매 회의 주기, 계약 갱신 주기, 최소 학습기간을 반영한다.
# [WHY] 특정 기간에만 맞는 threshold는 실제 구매 운영에서 재현성이 낮을 수 있으므로 시점별 안정성을 확인한다.
# [ASSUMPTION] 120개월 학습 후 12개월 단위 검증이 가격 사이클을 평가하기에 충분하다고 가정한다.
# [DESIGN LOGIC] 과거에서 미래로만 이동하며 모델을 재학습해 미래 정보 누수를 막고 구매판단시점별 성과를 비교한다.
# [DATA LINEAGE] df_model의 feature와 y가 walk-forward 성과표로 변환되어 콘솔 진단 결과에 출력된다.
# [REAL DATA REPLACEMENT] 실제 적용 시 rolling forecast origin, 계약월, 공급사 가격 갱신주기 기준으로 검증 창을 바꾼다.
# [INTERVIEW CHECK] 단일 test score가 아니라 여러 시점에서 운영 기준이 버티는지 확인했다는 점을 설명한다.
# ============================================================
def walk_forward_validate(X, y, start_train=120, step=12, threshold=0.6):
    rows = []

    for split_end in range(start_train, len(X) - step + 1, step):
        X_train_wf = X.iloc[:split_end]
        y_train_wf = y.iloc[:split_end]
        X_test_wf = X.iloc[split_end:split_end + step]
        y_test_wf = y.iloc[split_end:split_end + step]

        if y_train_wf.nunique() < 2 or y_test_wf.nunique() < 2:
            continue

        model_wf = LogisticRegression(max_iter=2000)
        model_wf.fit(X_train_wf, y_train_wf)

        proba_wf = model_wf.predict_proba(X_test_wf)[:, 1]
        pred_wf = (proba_wf >= threshold).astype(int)

        auc_wf = roc_auc_score(y_test_wf, proba_wf)
        tn, fp, fn, tp = confusion_matrix(y_test_wf, pred_wf, labels=[0, 1]).ravel()

        precision_wf = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_wf = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        rows.append({
            "train_end_idx": int(split_end),
            "test_size": int(len(X_test_wf)),
            "auc": round(float(auc_wf), 4),
            "precision": round(float(precision_wf), 4),
            "recall": round(float(recall_wf), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return pd.DataFrame(rows)


# =========================================================
# Threshold Search 설정값
# =========================================================
# A-10. Threshold Search 기준 상수 묶음
   # (threshold 정책을 평가할 때 쓸 통과 기준 + 순위 기준을 선언한다.)
# hard조건
MIN_RECALL = 0.50
MIN_SIGNALS_PER_YEAR = 6
MAX_SIGNALS_PER_YEAR = 12
MAX_REG2_FPR = 0.35
# soft조건
W_PRECISION = 3.0
W_TOTAL_FPR = 2.0
W_REGIME_BALANCE = 1.0

"""
B. 계산 도구층 (helper layer) 성과 계산식/정책 평가식 모음
실행 본문에서 계속 불러 쓰는 재사용 함수 묶음
즉, “직접 실행의 흐름”이 아니라 “실행을 도와주는 계산 부품”
"모델이 낸 확률/예측값을 운영 판단 숫자로 바꾸는 변환 규칙"
예측 결과 → 성능지표
예측 결과 → 연간 신호 수
확률 + threshold → 신호
신호 + 실제값 → 비용/정책평가
후보 여러 개 → 비교/정렬
holdout/raw cluster → 운영 가능한 구조로 변환
"""

# ============================================================
# [BLOCK] 구매 신호 성과지표 계산
# [현업 의미] 모델 신호가 실제 가격 상승을 얼마나 잡았고 얼마나 불필요한 검토를 만들었는지 구분한다.
# [판단 기준] TP, TN, FP, FN, precision, recall, false positive rate
# [산출물] threshold와 regime 정책 평가에 공통으로 쓰는 성과 딕셔너리
# [수정 포인트] 실무 적용 시 핵심 원자재별 가중치, 구매금액 기준 가중 confusion matrix를 추가한다.
# [WHY] 구매에서는 맞춘 신호와 놓친 상승, 불필요한 검토의 의미가 다르므로 오류 유형을 분리해야 한다.
# [ASSUMPTION] 각 월별 판단 행의 중요도를 동일하게 두고 성과를 집계한다.
# [DESIGN LOGIC] 모든 정책 후보가 같은 지표 계산식을 쓰게 해 threshold 비교의 기준을 통일한다.
# [DATA LINEAGE] y_true와 y_pred가 정책 성과 및 비용 계산의 기본 입력으로 이어진다.
# [REAL DATA REPLACEMENT] 실제 구매액, 공급 안정성, 계약 잔량별 오류 가중치를 연결한다.
# [INTERVIEW CHECK] false negative와 false positive를 같은 오류로 보지 않는 이유를 설명할 수 있어야 한다.
# ============================================================
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr
    }


# =========================================================
# 2. 연간 signal 수 계산
# =========================================================
#신호 개수를 연간 기준 검토 횟수로 환산한다.
# 신호/성과를 다루는 helper
def calc_signals_per_year_from_count(signal_count, months_count):
    if months_count <= 0:
        return np.nan
    return (signal_count / months_count) * 12

# 신호/성과를 다루는 helper
def calc_signals_per_year(df, pred_col="y_pred", months_count=None, date_col="Date"):
    signal_count = int(df[pred_col].sum())
    if months_count is None:
        months_count = len(df)
    return calc_signals_per_year_from_count(signal_count, months_count)


# ============================================================
# [BLOCK] Regime별 구매 검토 threshold 적용
# [현업 의미] 시장 국면별 변동성이 다르면 같은 확률 기준도 과소/과잉 대응이 될 수 있어 차등 기준을 적용한다.
# [판단 기준] regime, proba, th_reg1, th_reg2
# [산출물] regime 정책의 최종 구매 검토 신호 y_pred
# [수정 포인트] 실무 적용 시 고변동/저변동/상승장 정의와 각 threshold를 구매 정책으로 승인받아야 한다.
# [WHY] 장세가 다른데 동일 threshold만 쓰면 신호 빈도와 오류 비용이 특정 국면에 쏠릴 수 있다.
# [ASSUMPTION] regime 1과 2는 차등 threshold 대상이고 그 외 regime은 보수적으로 무신호 처리한다.
# [DESIGN LOGIC] 확률 예측과 시장 국면을 분리해 만든 뒤 마지막 정책 단계에서만 결합한다.
# [DATA LINEAGE] search/final 평가 테이블의 proba와 regime가 y_pred 정책 신호로 변환된다.
# [REAL DATA REPLACEMENT] 실제 시장 국면 라벨, 원자재 desk 전망, 변동성 지수, 공급 리스크 등과 연결한다.
# [INTERVIEW CHECK] regime 정책은 모델 구조 변경이 아니라 운영 threshold 정책 비교라는 점을 설명한다.
# ============================================================
def apply_regime_threshold(df, th_reg1, th_reg2,
                           proba_col="proba", regime_col="regime"):
    out = df.copy()

    def decide(row):
        # 상대적으로 낮은 수익률 국면은 별도 기준으로 가격 상승 신호 민감도를 조정한다.
        if row[regime_col] == 1:
            return int(row[proba_col] >= th_reg1)
        # 상대적으로 강한 가격 국면은 같은 확률도 다른 구매 대응 기준으로 해석할 수 있다.
        elif row[regime_col] == 2:
            return int(row[proba_col] >= th_reg2)
        else:
            # 고변동 또는 해석이 불안정한 regime은 포트폴리오 모델에서 보수적으로 구매 신호를 내지 않는다.
            return 0

    out["y_pred"] = out.apply(decide, axis=1)
    return out


# =========================================================
# 4. regime별 성능 계산
# =========================================================
#전체가 아니라 regime 1, regime 2 각각의 성능을 따로 계산한다.
# 신호/성과를 다루는 helper
def calc_regime_metrics(df, y_col="y_true", pred_col="y_pred", regime_col="regime"):
    result = {}

    for reg in [1, 2]:
        sub = df[df[regime_col] == reg].copy()

        if len(sub) == 0:
            result[f"reg{reg}_precision"] = np.nan
            result[f"reg{reg}_recall"] = np.nan
            result[f"reg{reg}_fpr"] = np.nan
            result[f"reg{reg}_signals"] = 0
            continue

        m = calc_metrics(sub[y_col], sub[pred_col])

        result[f"reg{reg}_precision"] = m["precision"]
        result[f"reg{reg}_recall"] = m["recall"]
        result[f"reg{reg}_fpr"] = m["fpr"]
        result[f"reg{reg}_signals"] = int(sub[pred_col].sum())

    return result


# =========================================================
# 5. 소프트조건 점수 계산
# =========================================================
# hard 조건을 통과한 정책 후보들 사이에서 soft ranking 점수를 만든다.
# 정책후보를 다루는 helper
def calc_score(total_precision, total_fpr, reg1_precision, reg2_precision):
    precision_gap = np.nan

    if pd.notna(reg1_precision) and pd.notna(reg2_precision):
        precision_gap = abs(reg1_precision - reg2_precision)

    score = 0.0

    if pd.notna(total_precision):
        score += W_PRECISION * total_precision

    if pd.notna(total_fpr):
        score -= W_TOTAL_FPR * total_fpr

    if pd.notna(precision_gap):
        score -= W_REGIME_BALANCE * precision_gap

    return score

# ============================================================
# [BLOCK] FP/FN 비용 기반 정책 비용 계산
# [현업 의미] 구매 검토 신호가 만든 업무 부담, 불필요한 조기구매, 가격 상승을 놓친 리스크를 같은 비용 축으로 비교한다.
# [판단 기준] signal_count, FP, FN, REVIEW_COST, FP_EXTRA_COST, FN_COST
# [산출물] threshold 정책별 total_cost
# [수정 포인트] 실무 적용 시 원재료별 구매금액, 보관비, 긴급 발주 비용, 계약 미체결 리스크를 비용 가중치로 반영한다.
# [WHY] 예측 정확도만으로는 구매 의사결정의 손익 영향을 설명하기 어렵기 때문에 오류 유형별 비용을 분리한다.
# [ASSUMPTION] 포트폴리오용 단순 모델이라 모든 signal과 오류에 고정 가중치를 적용한다.
# [DESIGN LOGIC] 신호 비용, FP 비용, FN 비용을 더해 threshold 후보를 비용 최소화 관점으로 비교할 수 있게 한다.
# [DATA LINEAGE] threshold sweep, single simulation, regime simulation의 비용 컬럼으로 이어진다.
# [REAL DATA REPLACEMENT] 실제 조기 구매 재고비, spot 구매 프리미엄, 공급사 협상 리드타임 비용으로 교체한다.
# [INTERVIEW CHECK] false negative 비용을 크게 둔 이유가 구매단가 상승과 협상력 약화 때문임을 설명한다.
# ============================================================
def calc_total_cost(fp, fn, signal_count,
                    review_cost=REVIEW_COST,
                    fp_extra_cost=FP_EXTRA_COST,
                    fn_cost=FN_COST):
    total_cost = (
        signal_count * review_cost
        + fp * fp_extra_cost
        + fn * fn_cost
    )
    return total_cost

# =========================================================
# 5-2. decision simulation (single threshold)
# =========================================================
# 단일 threshold 정책 1개를 받아서, 실제 운영 결과표 한 장으로 정리한다.
# 확률을 다루는 helper
def run_decision_simulation(y_true, y_prob, threshold,
                            months_per_test,
                            review_cost=REVIEW_COST,
                            fp_extra_cost=FP_EXTRA_COST,
                            fn_cost=FN_COST):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())  # 가치 있는 개입
    fp = int(((y_true == 0) & (y_pred == 1)).sum())  # 불필요 개입
    fn = int(((y_true == 1) & (y_pred == 0)).sum())  # 놓침
    tn = int(((y_true == 0) & (y_pred == 0)).sum())  # 잘 기다림

    signal_count = int(y_pred.sum())
    hold_count = int((y_pred == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    signals_per_year = (signal_count / months_per_test) * 12 if months_per_test > 0 else np.nan

    total_cost = calc_total_cost(
        fp=fp,
        fn=fn,
        signal_count=signal_count,
        review_cost=review_cost,
        fp_extra_cost=fp_extra_cost,
        fn_cost=fn_cost
    )

    cost_per_month = total_cost / months_per_test if months_per_test > 0 else np.nan

    return {
        "policy": f"single_{threshold:.2f}",
        "threshold_or_rule": f"single threshold = {threshold:.2f}",
        "buy_accelerate_count": signal_count,
        "hold_wait_count": hold_count,
        "valuable_action_tp": tp,
        "unnecessary_action_fp": fp,
        "missed_opportunity_fn": fn,
        "correct_wait_tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "signals_per_year": signals_per_year,
        "total_cost": total_cost,
        "cost_per_month": cost_per_month,
    }

# =========================================================
# 5-3. decision simulation (regime policy)
# =========================================================
# regime별 threshold 정책 1개를 받아서, 국면차등 정책의 최종 운영 결과표를 만든다.
  # 단일 정책과 regime 정책을 같은 형식으로 비교하려면,
     # regime 정책도 운영 결과표가 필요하다.
# 확률을 다루는 helper
def run_regime_decision_simulation(df_eval, th_reg1, th_reg2,
                                   date_col="Date",
                                   y_col="y_true",
                                   proba_col="proba",
                                   regime_col="regime",
                                   review_cost=REVIEW_COST,
                                   fp_extra_cost=FP_EXTRA_COST,
                                   fn_cost=FN_COST):
    pred_df = apply_regime_threshold(
        df=df_eval,
        th_reg1=th_reg1,
        th_reg2=th_reg2,
        proba_col=proba_col,
        regime_col=regime_col
    )

    y_true = pred_df[y_col].values
    y_pred = pred_df["y_pred"].values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    signal_count = int(pred_df["y_pred"].sum())
    hold_count = int((pred_df["y_pred"] == 0).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

    signals_per_year = calc_signals_per_year(
        pred_df,
        pred_col="y_pred",
        date_col=date_col
    )

    reg_m = calc_regime_metrics(
        pred_df,
        y_col=y_col,
        pred_col="y_pred",
        regime_col=regime_col
    )

    total_cost = calc_total_cost(
        fp=fp,
        fn=fn,
        signal_count=signal_count,
        review_cost=review_cost,
        fp_extra_cost=fp_extra_cost,
        fn_cost=fn_cost
    )

    months_per_test = len(pred_df)
    cost_per_month = total_cost / months_per_test if months_per_test > 0 else np.nan

    return {
        "policy": f"regime_{th_reg1:.2f}_{th_reg2:.2f}",
        "threshold_or_rule": f"reg1={th_reg1:.2f}, reg2={th_reg2:.2f}",
        "buy_accelerate_count": signal_count,
        "hold_wait_count": hold_count,
        "valuable_action_tp": tp,
        "unnecessary_action_fp": fp,
        "missed_opportunity_fn": fn,
        "correct_wait_tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "signals_per_year": signals_per_year,
        "total_cost": total_cost,
        "cost_per_month": cost_per_month,
        "reg1_signals": reg_m["reg1_signals"],
        "reg2_signals": reg_m["reg2_signals"],
        "reg1_precision": reg_m["reg1_precision"],
        "reg2_precision": reg_m["reg2_precision"],
        "reg1_fpr": reg_m["reg1_fpr"],
        "reg2_fpr": reg_m["reg2_fpr"],
    }


# =========================================================
# 6. threshold 조합 1개 평가
# =========================================================
# regime threshold 조합 (th_reg1, th_reg2) 하나를 받아서
   # **“이 후보가 운영안으로 통과 가능한지 + 점수는 몇 점인지”**를 평가한다.
     # 따라서 이 함수는 grid search의 원자 단위 평가기다.
# 정책후보를 다루는 helper
def evaluate_one_combo(df, th_reg1, th_reg2,
                       date_col="Date", y_col="y_true",
                       proba_col="proba", regime_col="regime"):

    pred_df = apply_regime_threshold(
        df=df,
        th_reg1=th_reg1,
        th_reg2=th_reg2,
        proba_col=proba_col,
        regime_col=regime_col
    )

    total_m = calc_metrics(pred_df[y_col], pred_df["y_pred"])

    signals_per_year = calc_signals_per_year(
        pred_df,
        pred_col="y_pred",
        date_col=date_col
    )

    reg_m = calc_regime_metrics(
        pred_df,
        y_col=y_col,
        pred_col="y_pred",
        regime_col=regime_col
    )

    pass_hard = True

    # 1) recall 조건
    if pd.isna(total_m["recall"]) or total_m["recall"] < MIN_RECALL:
        pass_hard = False

    # 2) 연간 signal 수 조건
    if pd.isna(signals_per_year) or not (MIN_SIGNALS_PER_YEAR <= signals_per_year <= MAX_SIGNALS_PER_YEAR):
        pass_hard = False

    # 3) regime2 FPR 조건
    if pd.isna(reg_m["reg2_fpr"]) or reg_m["reg2_fpr"] > MAX_REG2_FPR:
        pass_hard = False

    # 4) 무신호 금지 조건
    if reg_m["reg1_signals"] < 1:
        pass_hard = False

    if reg_m["reg2_signals"] < 1:
        pass_hard = False

    if pd.isna(reg_m["reg1_precision"]):
        pass_hard = False

    if pd.isna(reg_m["reg2_precision"]):
        pass_hard = False

    score = calc_score(
        total_precision=total_m["precision"],
        total_fpr=total_m["fpr"],
        reg1_precision=reg_m["reg1_precision"],
        reg2_precision=reg_m["reg2_precision"]
    )

    result = {
        "th_reg1": th_reg1,
        "th_reg2": th_reg2,

        "total_precision": total_m["precision"],
        "total_recall": total_m["recall"],
        "total_fpr": total_m["fpr"],
        "signals_per_year": signals_per_year,
        "total_signals": int(pred_df["y_pred"].sum()),

        "reg1_precision": reg_m["reg1_precision"],
        "reg1_recall": reg_m["reg1_recall"],
        "reg1_fpr": reg_m["reg1_fpr"],
        "reg1_signals": reg_m["reg1_signals"],

        "reg2_precision": reg_m["reg2_precision"],
        "reg2_recall": reg_m["reg2_recall"],
        "reg2_fpr": reg_m["reg2_fpr"],
        "reg2_signals": reg_m["reg2_signals"],

        "pass_hard": pass_hard,
        "score": score
    }

    return result


# =========================================================
# 7. 전체 threshold grid search
# =========================================================
# 여러 threshold 조합을 전부 돌려서 전체 결과표와 통과 후보표를 만든다.
   # regime별 검토기준을 여러 조합으로 돌려보고, 실무 제약을 통과하는 운영안 후보만 남기자
# 정책후보를 다루는 helper
def run_threshold_search(df,
                         th_grid_reg1=np.arange(0.45, 0.81, 0.05),
                         th_grid_reg2=np.arange(0.55, 0.91, 0.05),
                         date_col="Date",
                         y_col="y_true",
                         proba_col="proba",
                         regime_col="regime"):

    results = []

    for th1, th2 in product(th_grid_reg1, th_grid_reg2):
        res = evaluate_one_combo(
            df=df,
            th_reg1=th1,
            th_reg2=th2,
            date_col=date_col,
            y_col=y_col,
            proba_col=proba_col,
            regime_col=regime_col
        )
        results.append(res)

    result_df = pd.DataFrame(results)

    filtered_df = result_df[result_df["pass_hard"]].copy()

    filtered_df = filtered_df.sort_values(
        by=["score", "total_precision", "total_recall"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    result_df = result_df.sort_values(
        by=["pass_hard", "score", "total_precision"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return result_df, filtered_df


# =========================================================
# 7-2. 포트폴리오 CSV 산출 helper
# =========================================================
# ============================================================
# [BLOCK] 구매 판단 산출물 CSV 저장
# [현업 의미] 콘솔 진단에 머물던 예측 확률, threshold 정책, regime 정책, FP/FN 비용을 구매회의 자료로 재사용 가능한 표로 남긴다.
# [판단 기준] final holdout 실제값, 모델 예측 확률, 선택된 single threshold, 선택된 regime threshold, FP/FN 비용 가중치
# [산출물] outputs/ 하위 7개 CSV와 outputs/README.md
# [수정 포인트] 실무 적용 시 material, action 문구, 회의체명, 비용 항목을 회사 구매 프로세스에 맞춘다.
# [WHY] 포트폴리오 PDF에서는 모델 성능 숫자보다 "어떤 월에 BUY/WAIT 판단이 났고 왜 그런지"를 표로 보여줘야 한다.
# [ASSUMPTION] 현재 모델은 치즈 단일 품목이므로 material은 cheese로 고정하되, 향후 다품목 확장을 위해 컬럼을 유지한다.
# [DESIGN LOGIC] 기존 모델 로직이 만든 결과만 받아 CSV로 직렬화하고, 저장 과정에서는 임의 성과나 새 threshold를 만들지 않는다.
# [DATA LINEAGE] final_df, proba_final, sim_compare_df_display, coef_df가 PDF 제작용 outputs/*.csv로 이어진다.
# [REAL DATA REPLACEMENT] 계약단가, 공급사 견적, CME/ICE/FOB/CIF 가격, 환율, 운임, 관세, 리드타임, 안전재고, MOQ, 계약 잔량을 연결한다.
# [INTERVIEW CHECK] 이 산출물은 예측 모델 결과를 구매 검토 신호와 FP/FN 비용 언어로 바꾼 결과라고 설명한다.
# ============================================================
def _format_signal(flag):
    return "BUY/ACCELERATE" if int(flag) == 1 else "HOLD/WAIT"


def _safe_metric(metric_func, y_true, y_prob):
    try:
        return float(metric_func(y_true, y_prob))
    except ValueError:
        return np.nan


def _feature_business_meaning(feature):
    if feature == "ret_3m":
        return "최근 3개월 가격 모멘텀"
    if feature == "ret_6m":
        return "최근 6개월 가격 모멘텀"
    if feature == "vol_3m":
        return "단기 가격 변동성"
    if feature == "vol_6m":
        return "중기 가격 변동성"
    if feature == "price_ma6_ratio":
        return "최근 6개월 평균 대비 현재 가격 위치"
    if feature == "ma_3_minus_ma_12":
        return "단기 평균이 장기 평균보다 높은지 보는 추세 압력"
    if feature == "range_pos_6":
        return "최근 6개월 가격 범위 안에서 현재 가격의 위치"
    if feature in ["month_sin", "month_cos"]:
        return "월별 계절성"
    if feature.startswith("lag_milk_ret_"):
        return "우유 가격 변동의 후행 영향"
    if feature.startswith("lag_corn_ret_"):
        return "옥수수 가격 변동의 후행 영향"
    if feature.startswith("lag_ret_"):
        return "치즈 가격 자체의 후행 모멘텀"
    if feature.startswith("ret_milk_"):
        return "우유 가격의 최근 상승 압력"
    if feature == "cheese_milk_spread":
        return "치즈 가격과 우유 가격 변동 괴리로 보는 원가 전이 신호"
    return "구매 판단에 사용한 가격 흐름 기반 보조 신호"


def _add_cost_columns(policy_df):
    out = policy_df.copy()
    out["fp_count"] = out["unnecessary_action_fp"].astype(int)
    out["fn_count"] = out["missed_opportunity_fn"].astype(int)
    out["tp_count"] = out["valuable_action_tp"].astype(int)
    out["tn_count"] = out["correct_wait_tn"].astype(int)
    out["review_cost"] = REVIEW_COST
    out["fp_extra_cost"] = FP_EXTRA_COST
    out["fn_cost"] = FN_COST
    out["total_review_cost"] = out["buy_accelerate_count"] * REVIEW_COST
    out["total_fp_extra_cost"] = out["fp_count"] * FP_EXTRA_COST
    out["total_fn_cost"] = out["fn_count"] * FN_COST
    out["business_interpretation"] = (
        "FP: 불필요한 조기 구매 검토, 재고/자금 부담 가능성 | "
        "FN: 가격 상승을 놓친 경우, 구매단가 상승/협상력 약화 가능성"
    )
    return out


def save_outputs(final_df, proba_final, coef_df,
                 sim_df_display, sim_compare_df_display,
                 best_threshold, best_regime_th1, best_regime_th2,
                 train_size, policy_search_size, final_eval_size,
                 selected_policy_row, latest_row, latest_proba,
                 label_desc):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_policy = str(selected_policy_row["policy"])  # 최종 CSV의 y_pred 기준으로 사용할 운영 정책
    selected_single_threshold = (
        float(selected_policy.replace("single_", ""))
        if selected_policy.startswith("single_")
        else float(best_threshold)
    )
    single_pred = (np.asarray(proba_final) >= selected_single_threshold).astype(int)

    signal_df = final_df[[DATE_COL, PRICE_COL, "y", "regime"]].copy()
    signal_df = signal_df.rename(columns={
        PRICE_COL: "actual_price",
        "y": "y_true",
    })
    signal_df["material"] = "cheese"
    signal_df["proba"] = np.asarray(proba_final).astype(float)
    signal_df["single_threshold"] = float(selected_single_threshold)
    signal_df["single_signal"] = [_format_signal(v) for v in single_pred]
    signal_df["regime_threshold"] = np.where(
        signal_df["regime"] == 1,
        float(best_regime_th1),
        np.where(signal_df["regime"] == 2, float(best_regime_th2), np.nan)
    )
    signal_df["regime_pred"] = apply_regime_threshold(
        signal_df[[DATE_COL, "y_true", "proba", "regime"]].copy(),
        best_regime_th1,
        best_regime_th2,
    )["y_pred"].values
    signal_df["regime_signal"] = [_format_signal(v) for v in signal_df["regime_pred"]]
    signal_df["y_pred"] = signal_df["regime_pred"] if selected_policy.startswith("regime_") else single_pred
    signal_df["final_signal"] = [_format_signal(v) for v in signal_df["y_pred"]]
    signal_df["decision_reason"] = np.where(
        signal_df["y_pred"] == 1,
        "예측 확률이 선택된 구매 검토 기준 이상이므로 가격 상승 리스크 안건으로 검토",
        "예측 확률이 선택된 구매 검토 기준 미만이므로 즉시 구매보다 모니터링 우선"
    )
    signal_df["TP"] = ((signal_df["y_true"] == 1) & (signal_df["y_pred"] == 1)).astype(int)
    signal_df["FP"] = ((signal_df["y_true"] == 0) & (signal_df["y_pred"] == 1)).astype(int)
    signal_df["FN"] = ((signal_df["y_true"] == 1) & (signal_df["y_pred"] == 0)).astype(int)
    signal_df["TN"] = ((signal_df["y_true"] == 0) & (signal_df["y_pred"] == 0)).astype(int)
    signal_df = signal_df[[
        DATE_COL, "material", "actual_price", "y_true", "proba",
        "single_threshold", "single_signal", "regime", "regime_threshold",
        "regime_signal", "final_signal", "decision_reason", "y_pred",
        "TP", "FP", "FN", "TN"
    ]]

    threshold_policy_summary = sim_df_display.copy()
    threshold_policy_summary["threshold"] = threshold_policy_summary["policy"].str.replace("single_", "", regex=False).astype(float)
    threshold_policy_summary = threshold_policy_summary[[
        "policy", "threshold", "buy_accelerate_count", "hold_wait_count",
        "valuable_action_tp", "unnecessary_action_fp", "missed_opportunity_fn",
        "correct_wait_tn", "precision", "recall", "fpr", "signals_per_year",
        "total_cost", "cost_per_month"
    ]]

    regime_policy_summary = sim_compare_df_display[sim_compare_df_display["policy"].str.startswith("regime_")].copy()
    regime_policy_summary = regime_policy_summary.rename(columns={"threshold_or_rule": "threshold_or_rule"})
    regime_policy_summary = regime_policy_summary[[
        "policy", "threshold_or_rule", "buy_accelerate_count", "hold_wait_count",
        "valuable_action_tp", "unnecessary_action_fp", "missed_opportunity_fn",
        "correct_wait_tn", "precision", "recall", "fpr", "signals_per_year",
        "total_cost", "cost_per_month", "reg1_signals", "reg2_signals",
        "reg1_precision", "reg2_precision", "reg1_fpr", "reg2_fpr"
    ]]

    cost_source = pd.concat([
        threshold_policy_summary.rename(columns={"threshold": "threshold_or_rule"}),
        regime_policy_summary
    ], ignore_index=True, sort=False)
    false_positive_negative_cost = _add_cost_columns(cost_source)[[
        "policy", "fp_count", "fn_count", "tp_count", "tn_count",
        "review_cost", "fp_extra_cost", "fn_cost",
        "total_review_cost", "total_fp_extra_cost", "total_fn_cost",
        "total_cost", "business_interpretation"
    ]]

    latest_single_pred = int(float(latest_proba) >= selected_single_threshold)
    latest_regime = int(latest_row["regime"].iloc[0])
    latest_regime_threshold = (
        float(best_regime_th1) if latest_regime == 1
        else float(best_regime_th2) if latest_regime == 2
        else np.nan
    )
    latest_regime_pred = (
        int(float(latest_proba) >= latest_regime_threshold)
        if pd.notna(latest_regime_threshold)
        else 0
    )
    latest_final_pred = latest_regime_pred if selected_policy.startswith("regime_") else latest_single_pred
    latest_final_signal = _format_signal(latest_final_pred)
    latest_decision_signal = pd.DataFrame([{
        "Date": latest_row[DATE_COL].iloc[0],
        "material": "cheese",
        "latest_price": latest_row[PRICE_COL].iloc[0],
        "proba": float(latest_proba),
        "single_threshold": float(selected_single_threshold),
        "single_signal": _format_signal(latest_single_pred),
        "regime": latest_regime,
        "regime_threshold": latest_regime_threshold,
        "regime_signal": _format_signal(latest_regime_pred),
        "final_signal": latest_final_signal,
        "decision_summary": (
            f"BUY 검토: 향후 3개월 내 +{int(THRESH_UP * 100)}% 이상 상승 가능성이 threshold를 초과하여 사전 구매 검토 필요"
            if latest_final_pred == 1
            else "WAIT 유지: 상승 가능성이 기준 미만이므로 즉시 구매보다 모니터링 우선"
        ),
        "recommended_meeting_use": (
            "구매회의에서 계약 타이밍 또는 사전 물량 확보 여부 검토"
            if latest_final_pred == 1
            else "시황회의에서 가격 상승 리스크 안건으로 모니터링"
        ),
    }])

    feature_importance = coef_df.copy()
    feature_importance["direction"] = np.where(
        feature_importance["coef"] > 0,
        "상승 가능성 증가 방향",
        "상승 가능성 감소 방향"
    )
    feature_importance["business_meaning"] = feature_importance["feature"].map(_feature_business_meaning)
    feature_importance = feature_importance[["feature", "coef", "abs_coef", "direction", "business_meaning"]]

    selected_total_cost = float(selected_policy_row["total_cost"])
    selected_precision = float(selected_policy_row["precision"]) if pd.notna(selected_policy_row["precision"]) else np.nan
    selected_recall = float(selected_policy_row["recall"]) if pd.notna(selected_policy_row["recall"]) else np.nan
    selected_signals_per_year = float(selected_policy_row["signals_per_year"]) if pd.notna(selected_policy_row["signals_per_year"]) else np.nan

    backtest_summary = pd.DataFrame([
        {"metric": "train_size", "value": train_size, "interpretation": "모델 학습에 사용한 과거 월별 구매판단 행 수"},
        {"metric": "policy_search_size", "value": policy_search_size, "interpretation": "threshold 정책 탐색에 사용한 holdout 행 수"},
        {"metric": "final_eval_size", "value": final_eval_size, "interpretation": "정책 선택 이후 최종 검증에 사용한 최근 행 수"},
        {"metric": "label_mode", "value": LABEL_MODE, "interpretation": label_desc},
        {"metric": "threshold_up", "value": THRESH_UP, "interpretation": "가격 상승 이벤트로 보는 기준 상승률"},
        {"metric": "base_probability_threshold", "value": PROB_THRESHOLD, "interpretation": "사전에 둔 기본 구매 검토 확률 기준"},
        {"metric": "final_auc", "value": _safe_metric(roc_auc_score, y_final, proba_final), "interpretation": "최종 검증 구간에서 확률 순위 품질"},
        {"metric": "final_brier_score", "value": _safe_metric(brier_score_loss, y_final, proba_final), "interpretation": "최종 검증 구간에서 확률 보정 오차"},
        {"metric": "final_log_loss", "value": _safe_metric(log_loss, y_final, proba_final), "interpretation": "최종 검증 구간에서 확률 예측 손실"},
        {"metric": "selected_single_policy", "value": f"single_{best_threshold:.2f}", "interpretation": "search holdout에서 선택한 단일 threshold 정책"},
        {"metric": "selected_regime_policy", "value": f"regime_{best_regime_th1:.2f}_{best_regime_th2:.2f}", "interpretation": "search holdout에서 선택한 regime별 threshold 정책"},
        {"metric": "selected_policy_total_cost", "value": selected_total_cost, "interpretation": "최종 선택 정책의 FP/FN/검토 비용 합산"},
        {"metric": "selected_policy_precision", "value": selected_precision, "interpretation": "선택 정책 신호 중 실제 상승으로 이어진 비율"},
        {"metric": "selected_policy_recall", "value": selected_recall, "interpretation": "실제 상승 이벤트 중 선택 정책이 포착한 비율"},
        {"metric": "selected_policy_signals_per_year", "value": selected_signals_per_year, "interpretation": "선택 정책이 만드는 연간 구매 검토 신호 수"},
    ])

    readme_text = """# Outputs

이 폴더는 `python src/main.py` 실행 결과로 생성되는 포트폴리오용 CSV 산출물을 보관합니다.

- `forecast_signal_result.csv`: final holdout 월별 가격 상승 확률을 BUY/WAIT 구매 검토 신호와 TP/FP/FN/TN으로 정리한 판단 테이블입니다.
- `threshold_policy_summary.csv`: 단일 threshold 정책별 구매 검토 신호 수, precision/recall/FPR, FP/FN 비용을 비교합니다.
- `regime_policy_summary.csv`: 시장 regime별 threshold 정책의 전체 성과와 regime별 신호 품질을 비교합니다.
- `false_positive_negative_cost.csv`: FP와 FN을 불필요한 구매 검토 비용, 가격 상승을 놓친 비용 관점으로 해석합니다.
- `latest_decision_signal.csv`: 가장 최신 월 기준으로 구매회의 또는 시황회의에서 검토할 BUY/WAIT 판단을 제공합니다.
- `feature_importance.csv`: 로지스틱 회귀 계수 방향과 각 feature의 구매·원가 의사결정 의미를 설명합니다.
- `backtest_summary.csv`: 학습/탐색/최종검증 크기, label 기준, 최종 성과, 선택 정책 요약을 담습니다.
"""

    outputs = {
        "forecast_signal_result.csv": signal_df,
        "threshold_policy_summary.csv": threshold_policy_summary,
        "regime_policy_summary.csv": regime_policy_summary,
        "false_positive_negative_cost.csv": false_positive_negative_cost,
        "latest_decision_signal.csv": latest_decision_signal,
        "feature_importance.csv": feature_importance,
        "backtest_summary.csv": backtest_summary,
    }

    saved_paths = []
    for filename, out_df in outputs.items():
        path = OUTPUT_DIR / filename
        out_df.to_csv(path, index=False, encoding="utf-8-sig")
        saved_paths.append(path)

    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    saved_paths.append(readme_path)

    print_section("Saved Output Files")
    for path in saved_paths:
        print(path.as_posix())

    return saved_paths

# =========================================================
# 8. holdout 분리 / regime leakage 방지용 helper
# =========================================================

# threshold를 고른 데이터와 최종 검증 데이터를 분리해야 기준 선택 편향을 줄일 수 있다.
# 평가구조 / leakage 방지 helper
def split_holdout_temporal(holdout_df, search_ratio=POLICY_SEARCH_RATIO):
    split_idx = int(len(holdout_df) * search_ratio)

    # search / final 둘 다 최소 1행은 남기기
    split_idx = max(1, min(split_idx, len(holdout_df) - 1))

    search_df = holdout_df.iloc[:split_idx].copy().reset_index(drop=True)
    final_df = holdout_df.iloc[split_idx:].copy().reset_index(drop=True)
    return search_df, final_df, split_idx

# KMeans가 뽑은 raw cluster 번호를 운영 해석 가능한 regime 번호(0,1,2)로 다시 매핑한다.
  # 군집 번호를 그냥 숫자로 두지 말고, 해석 가능한 시장국면 이름으로 바꾸자
 # 평가구조 / leakage 방지 helper
def relabel_regimes_by_train_stats(regime_df, raw_regime_col="regime_raw"):
    stats = regime_df.groupby(raw_regime_col)[["ret_3m", "vol_3m"]].mean()

    high_vol_cluster = stats["vol_3m"].idxmax()
    remain = [c for c in stats.index.tolist() if c != high_vol_cluster]
    remain_stats = stats.loc[remain].sort_values("ret_3m")

    low_ret_cluster = remain_stats.index[0]
    high_ret_cluster = remain_stats.index[1]

    mapping = {
        high_vol_cluster: 0,
        low_ret_cluster: 1,
        high_ret_cluster: 2,
    }
    return mapping

# train에서만 scaler와 KMeans를 학습한 뒤, 그 기준으로 train/search/final에 regime를 붙인다.
# 평가구조 / leakage 방지 helper
def attach_regime_without_leakage(train_df, search_df, final_df, regime_features):
    scaler_regime = StandardScaler()
    train_scaled = scaler_regime.fit_transform(train_df[regime_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    train_raw = kmeans.fit_predict(train_scaled)

    train_regime_df = train_df[[DATE_COL] + regime_features].copy()
    train_regime_df["regime_raw"] = train_raw

    mapping = relabel_regimes_by_train_stats(train_regime_df, raw_regime_col="regime_raw")

    train_out = train_df.copy()
    search_out = search_df.copy()
    final_out = final_df.copy()

    search_raw = kmeans.predict(scaler_regime.transform(search_df[regime_features]))
    final_raw = kmeans.predict(scaler_regime.transform(final_df[regime_features]))

    train_out["regime"] = pd.Series(train_raw).map(mapping).values
    search_out["regime"] = pd.Series(search_raw).map(mapping).values
    final_out["regime"] = pd.Series(final_raw).map(mapping).values

    return train_out, search_out, final_out, scaler_regime, kmeans, mapping


'''
C. 데이터 준비층 시황·원가 데이터 정리
원천 데이터를 읽고, 월 기준 정렬/병합하고, 학습 가능한 테이블로 만드는 층
'''
# ============================================================
# [BLOCK] 데이터 로딩 및 월별 가격 시계열 정렬
# [현업 의미] 치즈 구매단가 판단에 필요한 중심 원재료 가격과 보조 원가 신호를 같은 월 기준으로 맞춘다.
# [판단 기준] Date, Price, corn_price, milk_price, 월 시작일 기준 정렬, 미래값 사용 금지
# [산출물] 치즈·옥수수·우유 가격이 월별로 병합된 df
# [수정 포인트] 실무 적용 시 원재료 코드, 계약통화, 단가 기준, 공급사별 가격, 환율·운임·관세 반영 방식을 바꾼다.
# [WHY] 가격 예측과 구매판단은 같은 기준월에서 관측 가능한 정보만 사용해야 미래 정보 누수를 피할 수 있다.
# [ASSUMPTION] 월별 가격 중복은 해당 월 마지막 관측값을 대표값으로 사용하고 보조 원재료 결측은 과거값만 forward fill한다.
# [DESIGN LOGIC] 각 원재료를 월 단위로 정렬한 뒤 동일 Date key로 병합해 원가 신호의 시간축을 통일한다.
# [DATA LINEAGE] data/cheese_price.csv, data/corn_price.csv, data/milk_price.csv가 df와 모델1_학습전.csv로 이어진다.
# [REAL DATA REPLACEMENT] ERP 구매단가, 공급사 견적 이력, 거래소 지표, 환율·물류비, 계약 잔량과 연결한다.
# [INTERVIEW CHECK] 보조 가격의 결측을 미래값으로 채우지 않아 실제 구매판단시점의 정보 제약을 지켰다고 설명한다.
# ============================================================
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

corn_df = pd.read_csv(CORN_CSV_PATH)
corn_df[DATE_COL] = pd.to_datetime(corn_df[DATE_COL])

milk_df = pd.read_csv(MILK_CSV_PATH)
milk_df[DATE_COL] = pd.to_datetime(milk_df[DATE_COL])

# 월 단위 구매판단을 위해 일자 정보를 월 기준 key로 통일한다.
df["ym"] = df[DATE_COL].dt.to_period("M")
corn_df["ym"] = corn_df[DATE_COL].dt.to_period("M")
milk_df["ym"] = milk_df[DATE_COL].dt.to_period("M")

# 같은 월 데이터가 중복될 경우 마지막 값 사용
df = (
    df.sort_values(DATE_COL)
      .groupby("ym", as_index=False)
      .last()
)

corn_df = (
    corn_df.sort_values(DATE_COL)
           .groupby("ym", as_index=False)
           .last()
)

milk_df = (
    milk_df.sort_values(DATE_COL)
           .groupby("ym", as_index=False)
           .last()
)

# period -> timestamp(월 시작일)
df[DATE_COL] = df["ym"].dt.to_timestamp()
corn_df[DATE_COL] = corn_df["ym"].dt.to_timestamp()
milk_df[DATE_COL] = milk_df["ym"].dt.to_timestamp()


# 월 기준 merge
df = df.merge(
    corn_df[[DATE_COL, CORN_PRICE_COL]],
    on=DATE_COL,
    how="left"
).sort_values(DATE_COL).reset_index(drop=True)

df = df.merge(
    milk_df[[DATE_COL, MILK_PRICE_COL]],
    on=DATE_COL,
    how="left"
).sort_values(DATE_COL).reset_index(drop=True)


# corn 월 시계열을 월 단위로 연속 reindex
full_months = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="MS")
df = (
    df.set_index(DATE_COL)
      .reindex(full_months)
      .rename_axis(DATE_COL)
      .reset_index()
)

# cheese 가격은 원래 월별 데이터니까 forward fill 하지 말고 그대로 두는 게 원칙
# corn은 월 구멍이 있으면 과거값으로만 채움 (미래값 사용 금지)
df[CORN_PRICE_COL] = df[CORN_PRICE_COL].ffill()
df[MILK_PRICE_COL] = df[MILK_PRICE_COL].ffill()

missing_corn = df[CORN_PRICE_COL].isna().sum()
missing_milk = df[MILK_PRICE_COL].isna().sum()

print(f"Missing corn_price rows after merge/reindex: {missing_corn}")
print(f"Missing milk_price rows after merge/reindex: {missing_milk}")

if missing_corn > 0:
    print("\nRows with missing corn_price (first 10):")
    print(df.loc[df[CORN_PRICE_COL].isna(), [DATE_COL]].head(10).to_string(index=False))

# ============================================================
# [BLOCK] 가격 변화율 및 상승 target 생성
# [현업 의미] 과거 가격 수준을 구매 리스크 판단에 쓰기 쉬운 변화율로 바꾸고, 모델이 맞힐 가격 상승 이벤트를 정의한다.
# [판단 기준] THRESH_UP, LABEL_MODE, 1개월/3개월 상승률, 향후 3개월 최대 가격 상승 여부
# [산출물] ret, corn_ret, milk_ret, ret_next_1m, ret_next_3m, ret_future_max_3m, y
# [수정 포인트] 실무 적용 시 상승률 기준, target horizon, 계약 갱신주기, 원재료별 가격 민감도를 조정한다.
# [WHY] 구매팀은 가격 자체보다 향후 상승 리스크가 구매 시점 앞당김을 정당화하는지 판단해야 한다.
# [ASSUMPTION] +3% 상승을 구매 검토가 필요한 의미 있는 가격 상승 이벤트로 본다.
# [DESIGN LOGIC] 여러 label mode를 유지해 단기 급등, 3개월 누적 상승, 3개월 내 최대 상승을 같은 코드에서 비교 가능하게 한다.
# [DATA LINEAGE] 월별 가격 df가 상승 label y로 변환되고 이후 X/y 학습 데이터로 이어진다.
# [REAL DATA REPLACEMENT] 실제 구매단가 변동 허용범위, 계약 조항, 가격 전가 가능성, 예산 승인 기준으로 target을 재정의한다.
# [INTERVIEW CHECK] label은 예측 편의를 위한 값이 아니라 구매 검토 필요성을 정의한 업무 기준이라고 설명한다.
# ============================================================
df["ret"] = df[PRICE_COL].pct_change(fill_method=None)  # 치즈 가격의 월별 상승·하락 흐름
df["corn_ret"] = df[CORN_PRICE_COL].pct_change(fill_method=None)  # 옥수수 가격 변동이 치즈 원가 압력으로 전이될 가능성
df["milk_ret"] = df[MILK_PRICE_COL].pct_change(fill_method=None)  # 우유 가격 변동이 치즈 가격 리스크로 연결될 가능성

# 1) strict_1m: 다음 달 수익률이 +3% 이상인가
df["ret_next_1m"] = df[PRICE_COL].shift(-1) / df[PRICE_COL] - 1

# 2) cum_3m: 3개월 뒤 누적수익률이 +3% 이상인가
df["ret_next_3m"] = df[PRICE_COL].shift(-3) / df[PRICE_COL] - 1

# 3) max_3m: 앞으로 3개월 안에 한 번이라도 현재 대비 +3% 이상 간 적이 있는가
future_prices = pd.concat(
    [
        df[PRICE_COL].shift(-1),
        df[PRICE_COL].shift(-2),
        df[PRICE_COL].shift(-3),
    ],
    axis=1
)
df["future_max_3m_price"] = future_prices.max(axis=1)
df["ret_future_max_3m"] = df["future_max_3m_price"] / df[PRICE_COL] - 1

# label mode별로 구매 검토가 필요한 가격 상승 이벤트를 다르게 정의한다.
if LABEL_MODE == "strict_1m":
    df["y"] = (df["ret_next_1m"] >= THRESH_UP).astype(int)
elif LABEL_MODE == "cum_3m":
    df["y"] = (df["ret_next_3m"] >= THRESH_UP).astype(int)
elif LABEL_MODE == "max_3m":
    df["y"] = (df["ret_future_max_3m"] >= THRESH_UP).astype(int)
else:
    raise ValueError("LABEL_MODE must be one of: strict_1m, cum_3m, max_3m")


# ============================================================
# [BLOCK] Lag·rolling·월별 seasonality feature 생성
# [현업 의미] 현재 구매판단시점에 이미 관측 가능한 과거 가격 흐름으로 향후 가격 상승 리스크를 설명한다.
# [판단 기준] 치즈/옥수수/우유 lag, 최근 모멘텀, 변동성, 이동평균 괴리, 가격 위치, 월별 계절성
# [산출물] lag_ret_*, lag_corn_ret_*, lag_milk_ret_*, ret_3m, vol_3m, price_ma6_ratio, month_sin/cos 등 feature
# [수정 포인트] 실무 적용 시 원재료별 리드타임, 가격 전이 시차, 계약 주기, 시즌 수요 패턴에 맞춰 feature를 조정한다.
# [WHY] 구매 판단은 미래 가격을 직접 알 수 없으므로 과거 가격 흐름과 변동성에서 상승 리스크 신호를 추출해야 한다.
# [ASSUMPTION] 공개 예시 데이터에서는 가격 feature만으로 시장 흐름을 설명한다고 단순화한다.
# [DESIGN LOGIC] lag, momentum, volatility, moving average, range position, seasonality를 분리해 가격 흐름의 여러 관점을 보존한다.
# [DATA LINEAGE] df의 가격 컬럼이 feature_cols로 정리되어 LogisticRegression 입력 X로 이어진다.
# [REAL DATA REPLACEMENT] 환율, 운임, 관세, 공급사 견적, 리드타임, 재고 커버, 계약 잔량 등 실무 구매 feature를 추가한다.
# [INTERVIEW CHECK] feature는 가격 예측용 숫자가 아니라 구매자가 관측 가능한 시장 신호를 구조화한 것이라고 설명한다.
# ============================================================
for lag in LAGS:
    df[f"lag_ret_{lag}"] = df["ret"].shift(lag)  # 치즈 가격 상승 압력의 과거 lag 신호
for lag in CORN_LAGS:
    df[f"lag_corn_ret_{lag}"] = df["corn_ret"].shift(lag)  # 옥수수 가격 전이 가능성을 반영한 lag 신호
for lag in MILK_LAGS:
    df[f"lag_milk_ret_{lag}"] = df["milk_ret"].shift(lag)  # 우유 가격 전이 가능성을 반영한 lag 신호


# =========================
# 3-1) 추가 시계열 피처 만들기
# =========================
# 이 블록은 사실 요약형 시계열 feature 묶음이다. 코드상으로는 한 덩어리지만, 의미상 7개 하위 묶음이 있다.
# 1) 최근 모멘텀
df["ret_3m"] = df[PRICE_COL].pct_change(3)  # 최근 3개월 치즈 가격 모멘텀
df["ret_6m"] = df[PRICE_COL].pct_change(6)  # 중기 치즈 가격 방향성

# 2) 변동성
df["vol_3m"] = df["ret"].rolling(3).std()  # 단기 가격 변동성으로 구매 판단 불확실성을 표현
df["vol_6m"] = df["ret"].rolling(6).std()  # 중기 가격 변동성으로 regime 판단의 시장 흔들림을 표현

# 3) 이동평균
df["ma_3"] = df[PRICE_COL].rolling(3).mean()  # 단기 평균 가격 수준
df["ma_6"] = df[PRICE_COL].rolling(6).mean()  # 반년 단위 기준 가격 수준
df["ma_12"] = df[PRICE_COL].rolling(12).mean()  # 연간 기준 평균 가격 수준

# 4) 이동평균 괴리
df["price_ma6_ratio"] = df[PRICE_COL] / df["ma_6"]  # 현재 가격이 최근 평균 대비 얼마나 높은지 보는 과열/저평가 신호

# 5) 추세
df["ma_3_minus_ma_12"] = df["ma_3"] - df["ma_12"]  # 단기 가격이 장기 기준보다 높아지는 추세 압력

# 6) rolling max/min + 최근 범위 내 현재 위치
df["roll_max_6"] = df[PRICE_COL].rolling(6).max()
df["roll_min_6"] = df[PRICE_COL].rolling(6).min()

range_width_6 = df["roll_max_6"] - df["roll_min_6"]
df["range_pos_6"] = np.where(
    range_width_6 == 0,
    np.nan,
    (df[PRICE_COL] - df["roll_min_6"]) / range_width_6
)  # 최근 6개월 가격 범위 안에서 현재 가격이 어느 위치인지 나타내는 구매 타이밍 신호
# 7) milk 관련 feature
df["ret_milk_3m"] = df[MILK_PRICE_COL].pct_change(3)  # 우유 가격의 단기 상승 압력
df["ret_milk_6m"] = df[MILK_PRICE_COL].pct_change(6)  # 우유 가격의 중기 상승 압력

# cheese와 milk의 상대 움직임
df["cheese_milk_spread"] = df["ret"] - df["milk_ret"]  # 치즈 가격과 우유 가격 변동의 괴리로 보는 원가 전이 신호

# =========================
# 4) 계절성 피처 (월)
# =========================
df["month"] = df[DATE_COL].dt.month  # 월별 계절성으로 가격 패턴을 구분하는 기준
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)  # 연중 위치를 순환형 계절 신호로 표현
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)  # month_sin과 함께 월별 반복 패턴을 표현

# ============================================================
# [BLOCK] 모델 학습용 feature table 확정
# [현업 의미] 구매판단시점에 사용 가능한 시장 신호만 남겨 예측 모델 입력 테이블을 확정한다.
# [판단 기준] feature_cols, y, lag/rolling 계산으로 발생한 결측 제거
# [산출물] df_model, X, y
# [수정 포인트] 실무 적용 시 결측 처리 정책, 원재료별 최소 관측기간, 신규 공급사/신규 품목 기준을 정의한다.
# [WHY] 결측이 섞인 판단 행은 threshold 정책 평가를 왜곡할 수 있으므로 학습 가능한 의사결정 행만 남긴다.
# [ASSUMPTION] 결측 행 제거가 데이터 손실보다 모델 입력 안정성에 더 중요하다고 가정한다.
# [DESIGN LOGIC] feature 후보를 명시적으로 나열한 뒤 y와 함께 결측을 제거해 모델 입력 범위를 고정한다.
# [DATA LINEAGE] df가 df_model로 축소되고 X/y 및 모델1_학습전.csv, 학습·평가 결과로 이어진다.
# [REAL DATA REPLACEMENT] 실무에서는 누락 원인별 대체값, 공급사별 가격 공백, 품목 출시 초기 처리 기준이 필요하다.
# [INTERVIEW CHECK] feature_cols가 구매 판단에 쓰인 시장 신호 목록이며 임의 컬럼 자동 선택이 아니라고 설명한다.
# ============================================================
feature_cols = (
    [f"lag_ret_{lag}" for lag in LAGS]
    + [f"lag_corn_ret_{lag}" for lag in CORN_LAGS]
    + [f"lag_milk_ret_{lag}" for lag in MILK_LAGS]
    + [
        "ret_3m", "ret_6m",
        "vol_3m", "vol_6m",
        "price_ma6_ratio",
        "ma_3_minus_ma_12",
        "range_pos_6",
        "month_sin", "month_cos",
        "ret_milk_3m", "ret_milk_6m",
        "cheese_milk_spread",
    ]
)

df_model = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

X = df_model[feature_cols]
y = df_model["y"]

'''
D. 학습/평가 분리층 룰 설계용 구간 vs 진짜 검증 구간 분리
시간순으로 데이터를 나누어 학습용 / 정책탐색용 / 최종평가용을 분리하는 층
C층에서 만든 최종 학습용 테이블 df_model, X, y를 시간순으로 잘라서,
학습용 데이터와 정책 탐색용 데이터와 최종 평가용 데이터를 분리하는 층이다.
'''
# ============================================================
# [BLOCK] 시간순 train/search/final 분리
# [현업 의미] 과거 데이터로 모델을 학습하고, 별도 구간에서 threshold를 고른 뒤, 가장 최근 구간으로 운영 성과를 확인한다.
# [판단 기준] TEST_RATIO, POLICY_SEARCH_RATIO, 시간순 split
# [산출물] train_df, search_df, final_df, X_train, X_search, X_final, y_train, y_search, y_final
# [수정 포인트] 실무 적용 시 구매 회의 기준월, 계약 시즌, 가격 급등기 포함 여부에 맞춰 기간 분할을 바꾼다.
# [WHY] threshold를 고른 구간과 최종 검증 구간을 분리해야 정책 선택 편향을 줄일 수 있다.
# [ASSUMPTION] 과거 80%를 학습에 쓰고 최근 20%를 정책 검증에 쓰는 분할이 포트폴리오 검증에 충분하다고 가정한다.
# [DESIGN LOGIC] 무작위 분할이 아니라 시간순 분할을 사용해 실제 구매판단에서 미래 정보를 알 수 없는 구조를 반영한다.
# [DATA LINEAGE] df_model이 train/search/final 구간으로 나뉘고 이후 모델 학습, threshold 탐색, final 비교에 각각 사용된다.
# [REAL DATA REPLACEMENT] 실제 계약 갱신 캘린더, 공급사 가격 변경 주기, 월별 S&OP freeze 기준으로 split을 설계한다.
# [INTERVIEW CHECK] threshold 탐색용 holdout과 final evaluation을 분리한 이유를 선택 편향 방지로 설명한다.
# ============================================================
split_idx = int(len(df_model) * (1 - TEST_RATIO))

# 시간순 분리여야 한다(시계열은 반드시 시간순이어야 한다.)
train_df = df_model.iloc[:split_idx].copy().reset_index(drop=True)
holdout_df = df_model.iloc[split_idx:].copy().reset_index(drop=True)

# Holdout을 Search / Final로 재분리
search_df, final_df, holdout_search_split_idx = split_holdout_temporal(
    holdout_df,
    search_ratio=POLICY_SEARCH_RATIO
)
# y 분리: y_train, y_search, y_final
y_train = train_df["y"].copy()
y_search = search_df["y"].copy()
y_final = final_df["y"].copy()

# X 분리: X_train, X_search, X_final
X_train = train_df[feature_cols].copy()
X_search = search_df[feature_cols].copy()
X_final = final_df[feature_cols].copy()

df.to_csv("모델1_학습전.csv", index=False, float_format="%.4f")

print("\n===== Data Split =====")
print(f"Train size: {len(train_df)}")
print(f"Holdout size: {len(holdout_df)}")
print(f"Policy-search size: {len(search_df)}")
print(f"Final-eval size: {len(final_df)}")


'''
E. 모델 학습층 상승 위험 확률 계산기 만들기
실제 확률모형(Logistic Regression)을 학습시키는 층
D층에서 분리한 X_train, y_train으로 실제 모델을 학습시키고, 
학습된 모델이 어떤 feature를 얼마나 강하게 쓰는지 1차적으로 확인하는 층이다.
'''
# ============================================================
# [BLOCK] 가격 상승 확률 모델 학습
# [현업 의미] 과거 원가 신호를 이용해 향후 가격 상승 가능성을 확률로 산출하는 구매 리스크 스코어러를 만든다.
# [판단 기준] X_train, y_train, LogisticRegression, feature coefficient
# [산출물] 학습된 model, feature별 계수 확인표
# [수정 포인트] 실무 적용 시 원자재별 모델, 비선형 모델, 공급사/환율/운임 feature, rolling retraining 정책을 검토한다.
# [WHY] 구매 의사결정에는 0/1 예측보다 threshold 정책으로 변환 가능한 확률 점수가 필요하다.
# [ASSUMPTION] 로지스틱 회귀가 feature 방향성과 확률 해석을 보여주는 포트폴리오 모델로 적합하다고 가정한다.
# [DESIGN LOGIC] 해석 가능한 확률 모델을 먼저 학습하고, coefficient를 출력해 어떤 시장 신호가 강하게 반영됐는지 점검한다.
# [DATA LINEAGE] train_df의 feature와 y가 model과 coef_df로 변환되고 이후 proba_search/proba_final로 이어진다.
# [REAL DATA REPLACEMENT] 실제 적용 시 모델 registry, calibration, explainability, 품목군별 backtest 기준이 필요하다.
# [INTERVIEW CHECK] 모델 선택 이유를 복잡도보다 구매 설명 가능성과 threshold 정책 연결성으로 설명한다.
# ============================================================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# =========================
# 7-1) Logistic 계수 확인
# =========================
# 학습된 로지스틱 모델이 어떤 feature를 상대적으로 강하게 반영했는지를 표로 확인한다.
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coef": model.coef_[0],
    "abs_coef": np.abs(model.coef_[0]),
}).sort_values("abs_coef", ascending=False)

print_df_compact(
    coef_df,
    title=f"Logistic Coefficients (top {COEF_TOP_N} by |coef|)",
    max_rows=COEF_TOP_N
)


'''
F. 단일 정책 탐색층 “몇 % 넘으면 구매검토할까?” 찾기
regime 없이, 하나의 threshold로 운영할 경우 어떤 정책이 좋은지 찾는 층
학습된 로지스틱 모델이 뽑은 확률을 바탕으로, single threshold 운영정책 후보를
탐색하고, 보수형/공격형 기준을 골라 final holdout에서 실제 운영 결과로 비교하는 층
'''

# ============================================================
# [BLOCK] 예측 확률 산출 및 단일 threshold 정책 탐색 준비
# [현업 의미] 모델 확률을 그대로 보고 끝내지 않고, 구매 검토 신호로 바꿀 후보 기준을 탐색할 준비를 한다.
# [판단 기준] proba_search, proba_final, PROB_THRESHOLD
# [산출물] search/final 구간의 가격 상승 예측 확률과 기본 0/1 신호
# [수정 포인트] 실무 적용 시 calibration, 담당자 승인 threshold, 원재료별 action 기준을 반영한다.
# [WHY] 예측 확률은 의사결정 언어가 아니므로 구매팀이 실행할 수 있는 signal로 변환해야 한다.
# [ASSUMPTION] search holdout에서 고른 threshold가 final holdout에서 검증 가능한 후보라고 가정한다.
# [DESIGN LOGIC] threshold 탐색 구간과 final 검증 구간의 확률을 먼저 분리 산출해 정책 선택과 평가를 분리한다.
# [DATA LINEAGE] model이 X_search/X_final을 proba_search/proba_final로 변환하고 threshold sweep과 final simulation에 투입한다.
# [REAL DATA REPLACEMENT] 실제 적용 시 구매 담당자 override, 공급사별 action rule, 예산 승인 workflow와 연결한다.
# [INTERVIEW CHECK] 예측 확률을 바로 구매하지 않고 threshold 정책으로 전환한 이유를 설명한다.
# ============================================================
proba_search = model.predict_proba(X_search)[:, 1]  # threshold 탐색에 사용하는 가격 상승 예측 확률
proba_final = model.predict_proba(X_final)[:, 1]  # 최종 운영 성과 검증에 사용하는 가격 상승 예측 확률

# 최종평가 구간을 downstream 호환용 test alias(같은 내용을 별칭으로 넣는것)로 둠
  # final holdout을 test라는 별칭으로 다시 부르는 블록
X_test = X_final.copy()
y_test = y_final.copy()
proba_test = proba_final.copy()
pred_test = (proba_test >= PROB_THRESHOLD).astype(int)  # 기본 threshold 기준 구매 검토 신호
"윗줄 뜻 : 기본 운영기준 0.60을 적용했을 때 final holdout에서 실제 검토 신호가 어떻게 뜨는가"
"그 확률값을 threshold로 잘라서 만든 최종 0/1 판단값"

"search 구간 확률에 대해 여러 threshold를 전부 훑어서,각 threshold별 성능/신호 수/비용을 표로 만든다."
# ===== search holdout에서만 threshold 탐색 =====
# 확률을 다루는 helper
def sweep_thresholds(y_true, y_prob, thresholds=None, months_per_test=None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)

    rows = []
    for t in thresholds:
        # 예측 확률이 후보 threshold 이상이면 가격 상승 리스크가 있다고 보고 구매 검토 신호로 변환한다.
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        signal_count = int(y_pred.sum())
        signals_per_year = calc_signals_per_year_from_count(signal_count, months_per_test)
        total_cost = calc_total_cost(fp=fp, fn=fn, signal_count=signal_count)

        rows.append({
            "threshold": float(t),  # 구매 검토 신호를 발생시키는 가격 상승 확률 기준
            "pred_pos": signal_count,  # 해당 threshold에서 발생한 구매 검토 신호 수
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),  # 가격 상승 포착/오경보/놓침/정상 대기 결과
            "precision": round(precision, 3),  # 신호가 실제 상승으로 이어진 비율
            "recall": round(recall, 3),  # 실제 상승 이벤트 중 신호가 포착한 비율
            "FPR": round(fpr, 3),  # 상승이 없는데 구매 검토를 발생시킨 비율
            "signals_per_year_est": None if signals_per_year is None else round(float(signals_per_year), 2),  # 연간 구매 검토 업무량 추정치
            "total_cost": round(float(total_cost), 3),  # FP/FN과 검토 부담을 합산한 정책 비용
        })
    return pd.DataFrame(rows)

# search holdout에 대해 threshold sweep 결과표를 실제로 생성한다.
df_thr = sweep_thresholds(y_search, proba_search, months_per_test=len(y_search))
print_df_compact(
    df_thr,
    title="Threshold Sweep (Policy Search Holdout)",
    max_rows=THRESHOLD_PREVIEW_N
)

# =========================
# 8-1) 운영용 threshold 추천 (search holdout)
# =========================
# threshold 후보 전체 중에서, 운영 제약을 만족하는 것만 먼저 남긴다.
candidate_thr = df_thr[
    (df_thr["signals_per_year_est"] >= 6) &
    (df_thr["signals_per_year_est"] <= 12) &
    (df_thr["recall"] >= 0.50)
].copy()

# 운영 조건을 만족한 후보 중에서 보수형 기본 운영안을 고른다.
print("\n===== Recommended Threshold Candidates (Policy Search Holdout) =====")
# 운영 제약을 통과한 threshold가 없으면 임의 최적화 대신 사전에 정한 기본 구매 기준을 유지한다.
if len(candidate_thr) == 0:
    print("No threshold matched the operating constraints.")
    best_threshold = PROB_THRESHOLD
else:
    print(candidate_thr.to_string(index=False))

    best_row = candidate_thr.sort_values(
        ["precision", "recall"],
        ascending=[False, False]
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    print("\n===== Recommended Threshold =====")
    print(best_row.to_string())

# 같은 운영 제약 후보 중에서, 이번에는 비용 최소화 기준으로 threshold를 하나 더 고른다.
print("\n===== Cost-Based Threshold Recommendation (Policy Search Holdout) =====")
cost_candidate_thr = df_thr[
    (df_thr["signals_per_year_est"] >= 6) &
    (df_thr["signals_per_year_est"] <= 12) &
    (df_thr["recall"] >= 0.50)
].copy()

# 비용 기준 후보가 없을 때도 정책 기준을 흔들지 않기 위해 기본 threshold로 되돌린다.
if len(cost_candidate_thr) == 0:
    print("No threshold matched the operating constraints for cost evaluation.")
    best_cost_threshold = PROB_THRESHOLD
else:
    best_cost_row = cost_candidate_thr.sort_values(
        ["total_cost", "precision"],
        ascending=[True, False]
    ).iloc[0]

    best_cost_threshold = float(best_cost_row["threshold"])
    print(best_cost_row.to_string())

# =========================
# 8-1-1) Decision Simulation: final holdout single policies
# =========================
# search에서 고른 single policy 후보들을 final holdout에서 실제 운영 결과표로 검증한다.

#보수형 기본 운영안(보통 0.60)을 final holdout에서 돌렸을 때"
# 실제로 어떤 성과가 나왔는지 담은 결과 딕셔너리"
 
sim_single_perf = run_decision_simulation(
    y_true=y_final,
    y_prob=proba_final,
    threshold=best_threshold,
    months_per_test=len(y_final)
)

sim_single_cost = run_decision_simulation(
    y_true=y_final,
    y_prob=proba_final,
    threshold=best_cost_threshold,
    months_per_test=len(y_final)
)

sim_single_060 = run_decision_simulation(
    y_true=y_final,
    y_prob=proba_final,
    threshold=0.60,
    months_per_test=len(y_final)
)

# single policy 시뮬레이션 결과를 사람이 읽을 수 있는 최종 비교표로 정리한다.
sim_rows = [sim_single_perf, sim_single_cost, sim_single_060]
sim_df = pd.DataFrame(sim_rows)

for col in ["precision", "recall", "fpr", "signals_per_year", "total_cost", "cost_per_month"]:
    sim_df[col] = sim_df[col].round(3)

sim_df_display = sim_df.drop_duplicates(
    subset=["policy", "threshold_or_rule"],
    keep="first"
).reset_index(drop=True)

print_section("Decision Simulation: Final Holdout Single Policies")
print(sim_df_display.to_string(index=False))


'''
G. 검증/진단층 이 룰이 일관적인지 확인
모델이 일관적인지, 데이터 구조가 어떤지, 과적합/불안정 징후가 있는지 확인하는 층
좋아 보이는 final 숫자 하나만 믿지 말고, 모델과 정책을 여러 각도에서 따로 검증해야 한다
이 검토 기준이 우연히 한 번 맞은 게 아니라, 실제 운영기준으로 설명 가능한 수준인지 
마지막으로 건강검진하는 층
'''
# 1) Core Model Result 계산
auc = roc_auc_score(y_test, proba_test)
cm = confusion_matrix(y_test, pred_test)#confusion matrix   TN     FP   예측 0
                                         #                   FN     TP   예측 1 
                                         #                  실제0    실제1
# 2) Core Model Result 출력
print_section("Core Model Result")
print(f"Label mode: {LABEL_MODE}")
print(f"Label meaning: {get_label_description(LABEL_MODE, THRESH_UP)}")
print(f"Cheese lags: {LAGS}")
print(f"Corn lags: {CORN_LAGS}")
print(f"Features: {feature_cols}")
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
print(f"Positive rate (train): {y_train.mean():.2%} | (test): {y_test.mean():.2%}")
print(f"ROC-AUC (probability quality): {auc:.4f}")

# 기본 운영안인 0.60 기준을 final holdout에 적용했을 때,
   # 실제 confusion matrix와 classification report를 상세하게 보여준다.
print_section("Threshold 0.60 Final Holdout Detail")
print(proba_test.max())
print(f"Probability threshold: {PROB_THRESHOLD}")
print("\nConfusion Matrix [ [TN FP], [FN TP] ]")
print(cm)
print("\nClassification Report")
print(classification_report(y_test, pred_test, digits=3, zero_division=0))


# =========================
# 8-2) Walk-forward evaluation
# =========================
# 한 번의 final holdout 결과만 보지 않고,
 # 시간을 앞으로 밀어가며 여러 번 학습/평가해서 시기별 안정성을 점검한다.

wf_df = walk_forward_validate(
    X, y,
    start_train=120,
    step=12,
    threshold=PROB_THRESHOLD
)

print_section("Walk-Forward Result")
if len(wf_df) == 0:
    print("Not enough data for walk-forward evaluation.")
else:
    print(wf_df.to_string(index=False))

    print_section("Walk-Forward Average")
    print(wf_df[["auc", "precision", "recall"]].mean().round(4))

# =========================
# 8-3) Correlation / Autocorrelation Check
# =========================

# 몇 개 대표 feature와 y 사이의 상관관계를 빠르게 확인해서,
 # feature 구조가 완전히 이상하게 가지는 않는지 점검한다.
print_section("Correlation Check")
corr_cols = [
    "lag_ret_1",
    "lag_ret_3",
    "ret_3m",
    "ma_3_minus_ma_12",
    "lag_corn_ret_4",
    "y"
]
print(df_model[corr_cols].corr().round(3).to_string())

# 방금 correlation matrix에서 본 걸 한 줄씩 다시 명확히 찍어준다.
print_section("Feature vs Label Correlation")
for col in ["lag_ret_1", "lag_ret_3", "ret_3m", "ma_3_minus_ma_12", "lag_corn_ret_4"]:
    corr_val = df_model[col].corr(df_model["y"])
    print(f"{col} vs y: {corr_val:.3f}")

# 치즈 수익률 시계열 자체가 자기 자신과 어떤 lag 구조를 가지는지 본다.
print_section("Cheese Return Autocorrelation")
for lag in range(1, 7):
    ac = df["ret"].autocorr(lag)
    print(f"ret autocorr lag{lag}: {ac:.3f}")

# final holdout에서 모델이 낸 확률들이 어느 구간에 몰려 있는지 시각적으로 본다.
print_section("Probability Distribution")
plt.hist(proba_test, bins=20)
plt.title("Predicted Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
# 포트폴리오용 CSV 실행에서는 그래프 창을 띄우지 않고 진단용 figure를 닫아 파이프라인 종료를 안정화한다.
plt.close()


'''
H. Regime 실험 및 최종 운영결정층 장세별로 다른 룰을 써볼지 검토 / 실제 운영안 선택 + 최신월 액션 판단
시장 국면을 나누고, 국면별 threshold를 따로 두는 실험을 한 뒤,
최종적으로 single 정책 vs regime 정책을 비교하는 층
즉 H층은 두 가지 엄격한 원칙 위에 서 있다.
regime leakage 금지
regime threshold 선택 편향 금지
그래서:
regime는 train에서만 fit
search에서 threshold 탐색
final에서 최종 비교
이 순서를 지킨다.

'''

# 시장 상태를 나타내는 몇 개 feature로 시장 국면(regime) 을 붙인다.
  # 단, train에서만 학습하고 search/final에는 그대로 예측만 적용한다
# ============================================================
# [BLOCK] 시장 regime 탐지 및 누수 방지 적용
# [현업 의미] 가격 모멘텀·변동성·평균 대비 위치로 시장 국면을 구분해 같은 확률도 국면별로 다르게 해석할 수 있게 한다.
# [판단 기준] ret_3m, vol_3m, price_ma6_ratio, train-only KMeans fit
# [산출물] train/search/final 데이터에 부여된 regime 컬럼
# [수정 포인트] 실무 적용 시 구매 desk의 시장 국면 정의, 변동성 지수, 공급 리스크, 원재료별 regime 기준을 반영한다.
# [WHY] 장세에 따라 같은 상승 확률이 의미하는 구매 위험도가 달라질 수 있으므로 regime별 정책 비교가 필요하다.
# [ASSUMPTION] 세 가지 가격 feature만으로 시장 국면을 3개 cluster로 단순화한다.
# [DESIGN LOGIC] regime는 train에서만 학습하고 search/final에는 예측만 적용해 시장 국면 정보의 미래 누수를 차단한다.
# [DATA LINEAGE] train/search/final feature가 regime 컬럼으로 확장되고 regime threshold search에 투입된다.
# [REAL DATA REPLACEMENT] 실제 원자재 시장 리포트, 공급 차질 지표, 환율·운임 변동성, 재고 커버 정보를 추가한다.
# [INTERVIEW CHECK] regime 정책은 final에서 새로 맞춘 것이 아니라 train 기준 국면을 search/final에 적용한 구조라고 설명한다.
# ============================================================
regime_features = [
    "ret_3m",  # 최근 가격 상승/하락 방향으로 보는 시장 모멘텀
    "vol_3m",  # 단기 가격 변동성으로 보는 구매 판단 불확실성
    "price_ma6_ratio",  # 현재 가격이 최근 평균 대비 높은지 낮은지 보는 가격 위치 신호
]

train_df, search_df, final_df, scaler_regime, kmeans, regime_mapping = attach_regime_without_leakage(
    train_df=train_df,
    search_df=search_df,
    final_df=final_df,
    regime_features=regime_features
)

# df_model에도 regime를 붙여서 마지막 forecast에서 참조 가능하게 함
df_model = pd.concat([train_df, search_df, final_df], axis=0).reset_index(drop=True)

# 붙인 regime가 실제로: 몇 개나 있는지/어떤 특징을 갖는지 를 요약해서 보여준다.
print_section("Regime Counts (Train/Search/Final Combined)")
print(df_model["regime"].value_counts().sort_index())

# “시장 상태를 세 타입으로 나눠봤더니, 
# 하나는 출렁이는 고변동 장, 하나는 눌린 장, 하나는 상대적으로 강한 장처럼 보인다”
print_section("Regime Feature Mean (Train/Search/Final Combined)")
print(
    df_model.groupby("regime")[regime_features]
    .mean()
    .round(4)
    .to_string()
)

# =========================
# 10) Search holdout에서 regime 성능 / threshold 탐색
# =========================
# 기본 threshold 0.60을 regime별로 적용했을 때, 각 regime에서:
 # 확률 품질
 # calibration 관련 점수, precision/recall 이 어떻게 다른지 본다.
 # 같은 0.60 기준이라도, 눌린 장에서는 잘 작동하고 강한 장에서는 다르게 작동할 수 있나"
 
search_eval_df = search_df[[DATE_COL, "y", "regime"]].copy()
search_eval_df = search_eval_df.rename(columns={"y": "y_true"})
search_eval_df["proba"] = proba_search
search_eval_df["pred"] = (search_eval_df["proba"] >= PROB_THRESHOLD).astype(int)

rows = []
for r in sorted(search_eval_df["regime"].dropna().unique()):
    temp = search_eval_df[search_eval_df["regime"] == r]
# 표본이 많은 레짐은 건너뛴다. 레짐0은 5개 이하라서 버림.(홀딩)
    # 표본이 너무 적은 regime는 성과지표가 불안정하므로 정책 판단 근거에서 제외한다.
    if len(temp) < 5:
        continue

    y_true_r = temp["y_true"].values
    y_prob_r = temp["proba"].values
    y_pred_r = temp["pred"].values

    pos_rate = y_true_r.mean()
    pred_rate = y_pred_r.mean()

    try:
        auc_r = roc_auc_score(y_true_r, y_prob_r)
    except:
        auc_r = np.nan

    brier_r = brier_score_loss(y_true_r, y_prob_r)
    ll_r = log_loss(y_true_r, y_prob_r)

    tn, fp, fn, tp = confusion_matrix(y_true_r, y_pred_r, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    rows.append({
        "regime": r,
        "count": len(temp),
        "pos_rate": round(pos_rate, 3),
        "pred_rate": round(pred_rate, 3),
        "auc": round(auc_r, 3) if not np.isnan(auc_r) else None,
        "brier": round(brier_r, 3),
        "logloss": round(ll_r, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    })

regime_result = pd.DataFrame(rows)

print_section("Regime-wise Performance (Policy Search Holdout)")
print(regime_result.to_string(index=False))

# =========================
# 10-1) search holdout에서 regime threshold search
# =========================
# regime 1과 regime 2에 서로 다른 threshold를 줄 경우,
   # 어떤 조합이 운영 조건을 가장 잘 만족하는지 search holdout에서 탐색한다.
# 국면별 threshold 조합을 전부 돌려보는 블록
df_eval = search_eval_df[[DATE_COL, "y_true", "proba", "regime"]].copy()

result_df, filtered_df = run_threshold_search(
    df=df_eval,
    th_grid_reg1=np.arange(0.45, 0.81, 0.05),
    th_grid_reg2=np.arange(0.55, 0.91, 0.05),
    date_col=DATE_COL,
    y_col="y_true",
    proba_col="proba",
    regime_col="regime"
)

# regime threshold search 결과를 보여주고, hard filter 통과 후보 중 최상위 조합을 고른다
print_df_compact(
    result_df,
    title="Regime Threshold Search (Policy Search Holdout): All Results",
    max_rows=REGIME_SEARCH_PREVIEW_N
)
# “국면별 검토 기준을 여러 조합으로 돌려본 결과, 가장 그럴듯한 운영안은 reg1=0.55, reg2=0.60 이다”
print_section("Regime Threshold Search (Policy Search Holdout): Passed Hard Filter")
# 국면별 threshold 조합이 운영 제약을 만족하지 못하면 무리하게 차등 정책을 채택하지 않고 기본 기준을 유지한다.
if len(filtered_df) == 0:
    print("No threshold combination passed the hard conditions.")
    best_regime_th1 = 0.60
    best_regime_th2 = 0.60
else:
    print(filtered_df.head(REGIME_SEARCH_PREVIEW_N).to_string(index=False))
    if len(filtered_df) > REGIME_SEARCH_PREVIEW_N:
        print(f"... ({len(filtered_df) - REGIME_SEARCH_PREVIEW_N} more rows)")

    best_combo = filtered_df.iloc[0]
    best_regime_th1 = float(best_combo["th_reg1"])
    best_regime_th2 = float(best_combo["th_reg2"])

    print_section("Best Regime Threshold Combination")
    print(best_combo.to_string())

# =========================
# 10-2) Final holdout에서 single vs regime 비교
# =========================
# search에서 고른 regime policy를, final holdout에서 single policy 두 개와 직접 비교한다.
# 국면별 차등 대응안이 정말 더 좋은가? 아니면 그냥 단순한 single 기준 0.60이 더 운영하기 쉬운가?
final_eval_df = final_df[[DATE_COL, "y", "regime"]].copy()
final_eval_df = final_eval_df.rename(columns={"y": "y_true"})
final_eval_df["proba"] = proba_final

sim_single_perf_final = run_decision_simulation(
    y_true=y_final,
    y_prob=proba_final,
    threshold=best_threshold,
    months_per_test=len(y_final)
)

sim_single_cost_final = run_decision_simulation(
    y_true=y_final,
    y_prob=proba_final,
    threshold=best_cost_threshold,
    months_per_test=len(y_final)
)

sim_regime_best = run_regime_decision_simulation(
    df_eval=final_eval_df,
    th_reg1=best_regime_th1,
    th_reg2=best_regime_th2,
    date_col=DATE_COL,
    y_col="y_true",
    proba_col="proba",
    regime_col="regime"
)

sim_compare_df = pd.DataFrame([
    sim_single_perf_final,
    sim_single_cost_final,
    sim_regime_best
])

round_cols = [
    "precision", "recall", "fpr", "signals_per_year",
    "total_cost", "cost_per_month",
    "reg1_precision", "reg2_precision",
    "reg1_fpr", "reg2_fpr"
]
for col in round_cols:
    if col in sim_compare_df.columns:
        sim_compare_df[col] = sim_compare_df[col].round(3)

sim_compare_df_display = sim_compare_df.drop_duplicates(
    subset=["policy", "threshold_or_rule"],
    keep="first"
).reset_index(drop=True)

print_section("Decision Simulation: Final Holdout Single vs Regime Policy")
print(sim_compare_df_display.to_string(index=False))



# ============================================================
# [BLOCK] 마지막 월 기준 구매 검토 action 출력
# [현업 의미] 최신 가격 관측월의 상승 확률을 구매팀이 이해할 수 있는 BUY/ACCELERATE 또는 HOLD/WAIT 판단으로 변환한다.
# [판단 기준] last_row, next_up_proba, PROB_THRESHOLD
# [산출물] 마지막 월 상승 확률, threshold 대비 차이, 최종 구매 검토 action
# [수정 포인트] 실무 적용 시 action을 발주 검토, 계약 선매입, 공급사 견적 요청, 관망 등 세분화한다.
# [WHY] 모델 결과는 확률로 끝나지 않고 실제 구매 검토 여부로 이어져야 포트폴리오의 의사결정 가치가 드러난다.
# [ASSUMPTION] 마지막 df_model 행이 현재 사용 가능한 최신 구매판단시점이라고 가정한다.
# [DESIGN LOGIC] 동일 feature_cols로 최신 행을 평가하고 기본 threshold와 비교해 단순하고 설명 가능한 action을 출력한다.
# [DATA LINEAGE] df_model 최신 행이 next_up_proba와 decision으로 변환되어 콘솔 최종 판단에 출력된다.
# [REAL DATA REPLACEMENT] 실제 최신 가격 feed, 오픈 PO, 재고 커버, 계약 잔량, 승인 workflow와 연결한다.
# [INTERVIEW CHECK] 최종 문장은 예측 확률을 구매 검토 신호로 바꾼다는 프로젝트 목적을 보여주는 지점이다.
# ============================================================
last_row = df_model.iloc[[-1]]
last_X = last_row[feature_cols]

next_up_proba = model.predict_proba(last_X)[0, 1]
label_desc = get_label_description(LABEL_MODE, THRESH_UP)
forecast_header = get_forecast_header(LABEL_MODE)

print_section(forecast_header.replace("=====", "").strip())
print(f"Label mode used for training: {LABEL_MODE}")
print(f"Label meaning: {label_desc}")
print(f"Last available month: {last_row[DATE_COL].iloc[0].date()}")

print(f"P({label_desc}) = {float(next_up_proba):.6f}")

print(f"Raw probability: {next_up_proba!r}")
print(f"Raw probability exact: {float(next_up_proba):.15f}")
print(f"Rounded probability: {float(next_up_proba):.6f}")
print(f"Probability threshold exact: {float(PROB_THRESHOLD):.6f}")
print(f"Probability minus threshold: {float(next_up_proba - PROB_THRESHOLD):.6f}")

# 상승 확률이 구매 검토 기준 이상이면 선매입·계약 협상·견적 확인 같은 조기 대응 검토 대상으로 본다.
decision = bool(next_up_proba >= PROB_THRESHOLD)
print(f"Decision flag: {decision}")


print(f"Decision (threshold {PROB_THRESHOLD}): {'BUY/ACCELERATE' if decision else 'HOLD/WAIT'}")

print_section("Label Distribution (full data)")
print(df_model["y"].value_counts(normalize=True).sort_index())

# ============================================================
# [BLOCK] 포트폴리오 PDF용 CSV 산출
# [현업 의미] 콘솔에서 확인한 모델·정책 결과를 구매 검토 테이블, 정책 요약, 최신 의사결정 파일로 저장한다.
# [판단 기준] final holdout의 실제 성과, single/regime 정책별 total_cost, 최신 월 예측 확률
# [산출물] outputs/forecast_signal_result.csv 외 6개 CSV와 outputs/README.md
# [수정 포인트] 실무 적용 시 산출물 컬럼에 구매 금액, 공급사, 계약 만료월, 재고 커버월을 추가한다.
# [WHY] PDF 포트폴리오에서는 콘솔 로그보다 재현 가능한 CSV 산출물이 모델의 의사결정 연결성을 더 명확히 보여준다.
# [ASSUMPTION] final holdout에서 비용이 가장 낮은 정책을 CSV의 최종 판단 기준으로 사용한다.
# [DESIGN LOGIC] 기존 계산 결과를 재사용해 저장만 수행하고 모델 로직, threshold 탐색 기준, 평가 계산식은 변경하지 않는다.
# [DATA LINEAGE] sim_compare_df_display와 final_df가 outputs 폴더의 구매 판단 산출물로 이어진다.
# [REAL DATA REPLACEMENT] 회의 자료 자동화 시 BI 대시보드, ERP 구매 품의, S&OP 회의 템플릿과 연결한다.
# [INTERVIEW CHECK] "예측 확률을 구매 검토 신호와 비용 비교 산출물로 변환했다"는 흐름을 설명한다.
# ============================================================
selected_policy_row = sim_compare_df_display.sort_values(
    ["total_cost", "precision"],
    ascending=[True, False]
).iloc[0]

save_outputs(
    final_df=final_df,
    proba_final=proba_final,
    coef_df=coef_df,
    sim_df_display=sim_df_display,
    sim_compare_df_display=sim_compare_df_display,
    best_threshold=best_threshold,
    best_regime_th1=best_regime_th1,
    best_regime_th2=best_regime_th2,
    train_size=len(train_df),
    policy_search_size=len(search_df),
    final_eval_size=len(final_df),
    selected_policy_row=selected_policy_row,
    latest_row=last_row,
    latest_proba=next_up_proba,
    label_desc=label_desc,
)
