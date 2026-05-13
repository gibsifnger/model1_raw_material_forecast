# 터미널에 pip install scikit-learn 이거 해주세요~
# 터미널에 pip install matplotlib 이거 해주세요~
import pandas as pd
import numpy as np
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss

"""
A. 설정층 운영 원칙/판정 기준 정의
모델/라벨/threshold/cost/출력형식의 전역 룰을 잡는 층
"""
# =========================
# 0) 설정
# =========================
# A-1. 파일/컬럼/문제정의 상수 묶음
CSV_PATH = "data/cheese_price.csv"
CORN_CSV_PATH = "data/corn_price.csv"
MILK_CSV_PATH = "data/milk_price.csv"

DATE_COL = "Date"
PRICE_COL = "Price"
CORN_PRICE_COL = "corn_price"
MILK_PRICE_COL = "milk_price"

THRESH_UP = 0.03
HORIZON = 1

# A-2. 시계열 입력 범위/분할 규칙 상수 묶음
LAGS = [1, 2, 3, 6]             # cheese용 lag
CORN_LAGS = [2, 3, 4, 6]        # corn용 lag
MILK_LAGS = [1, 2, 3, 6]        # milk용 lag
TEST_RATIO = 0.2
POLICY_SEARCH_RATIO = 0.5  # holdout 중 앞쪽은 threshold 탐색, 뒤쪽은 최종 평가

PROB_THRESHOLD = 0.6

LABEL_MODE = "max_3m"
# "strict_1m" : 다음 달 +3% 이상
# "cum_3m"    : 3개월 뒤 누적수익률 +3% 이상
# "max_3m"    : 앞으로 3개월 안에 한 번이라도 +3% 이상

# =========================
# Cost Function 설정값
# =========================
# A-3. 비용 기준 상수 묶음
REVIEW_COST = 1.0
FP_EXTRA_COST = 1.5
FN_COST = 4.0

# =========================
# 출력 가독성 설정 (계산 결과에는 영향 없음)
# =========================
# A-4. 출력 형식 상수 묶음(결과를 얼마나 요약해서 보여줄것인지)
COEF_TOP_N = 10
THRESHOLD_PREVIEW_N = 10
REGIME_SEARCH_PREVIEW_N = 10

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

# A-9. 검증 helper: walk_forward_validate
   # 시계열 데이터를 한 번만 train/test로 자르지 않고 시간을 앞으로 밀어가며 여러 번 재검증하는 함수다.
   # 시간을 밀어가며 같은 모델을 여러 번 다시 학습/평가해서 안정성을 보는 함수
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

# =========================================================
# 1. 기본 성능지표 계산
# =========================================================
# 실제값과 예측값을 받아서 분류 성능의 가장 기본 뼈대를 계산한다.
    # 즉 TP/TN/FP/FN과 precision/recall/fpr를 한 번에 뽑는 기본 계산기다.
# 신호/성과를 다루는 helper
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


# =========================================================
# 3. regime별 threshold 적용
# =========================================================
#regime별로 다른 threshold를 적용해서 최종 신호 y_pred를 만든다.
# 확률을 다루는 helper
def apply_regime_threshold(df, th_reg1, th_reg2,
                           proba_col="proba", regime_col="regime"):
    out = df.copy()

    def decide(row):
        if row[regime_col] == 1:
            return int(row[proba_col] >= th_reg1)
        elif row[regime_col] == 2:
            return int(row[proba_col] >= th_reg2)
        else:
            # regime 0 등 나머지는 일단 신호 없음 처리
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

# =========================================================
# 5-1. cost function 계산
# =========================================================
# 정책 결과를 총 기대비용으로 환산한다.
# 신호/성과를 다루는 helper
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
# =========================
# 1) 데이터 로드 & 정렬
# =========================
# 치즈/옥수수/우유 원본 파일을 읽어서, 월 단위로 정렬되고 서로 맞물리는 하나의 기준 테이블로 만든다.
df = pd.read_csv(CSV_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

corn_df = pd.read_csv(CORN_CSV_PATH)
corn_df[DATE_COL] = pd.to_datetime(corn_df[DATE_COL])

milk_df = pd.read_csv(MILK_CSV_PATH)
milk_df[DATE_COL] = pd.to_datetime(milk_df[DATE_COL])

# 월 단위 키 통일 (해당 월의 1일로 맞춤)
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

# =========================
# 2) 수익률 & 라벨(y) 만들기
# =========================
# 원본 가격 레벨을 변화율 정보로 바꾸고, 동시에 모델이 맞혀야 할 **정답 라벨 y**를 만든다.
df["ret"] = df[PRICE_COL].pct_change(fill_method=None)
df["corn_ret"] = df[CORN_PRICE_COL].pct_change(fill_method=None)
df["milk_ret"] = df[MILK_PRICE_COL].pct_change(fill_method=None)

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

if LABEL_MODE == "strict_1m":
    df["y"] = (df["ret_next_1m"] >= THRESH_UP).astype(int)
elif LABEL_MODE == "cum_3m":
    df["y"] = (df["ret_next_3m"] >= THRESH_UP).astype(int)
elif LABEL_MODE == "max_3m":
    df["y"] = (df["ret_future_max_3m"] >= THRESH_UP).astype(int)
else:
    raise ValueError("LABEL_MODE must be one of: strict_1m, cum_3m, max_3m")


# =========================
# 3) lag 피처 만들기 (t 시점에서 과거만 사용)
# =========================
# 현재 시점에서 과거 몇 개월의 수익률 정보를 feature로 만든다.
for lag in LAGS:
    df[f"lag_ret_{lag}"] = df["ret"].shift(lag)
for lag in CORN_LAGS:
    df[f"lag_corn_ret_{lag}"] = df["corn_ret"].shift(lag)    
for lag in MILK_LAGS:
    df[f"lag_milk_ret_{lag}"] = df["milk_ret"].shift(lag)


# =========================
# 3-1) 추가 시계열 피처 만들기
# =========================
# 이 블록은 사실 요약형 시계열 feature 묶음이다. 코드상으로는 한 덩어리지만, 의미상 7개 하위 묶음이 있다.
# 1) 최근 모멘텀
df["ret_3m"] = df[PRICE_COL].pct_change(3) #3개월전 대비 퍼센트 변화율 계산
df["ret_6m"] = df[PRICE_COL].pct_change(6)

# 2) 변동성
df["vol_3m"] = df["ret"].rolling(3).std() #rolling(3) = 최근 3개 구간씩 묶음(그달 포함)
df["vol_6m"] = df["ret"].rolling(6).std() #std() = 그 3개 구간 수익율의 표준편차 계산

# 3) 이동평균
df["ma_3"] = df[PRICE_COL].rolling(3).mean()
df["ma_6"] = df[PRICE_COL].rolling(6).mean()
df["ma_12"] = df[PRICE_COL].rolling(12).mean()

# 4) 이동평균 괴리
df["price_ma6_ratio"] = df[PRICE_COL] / df["ma_6"]

# 5) 추세
df["ma_3_minus_ma_12"] = df["ma_3"] - df["ma_12"]

# 6) rolling max/min + 최근 범위 내 현재 위치
df["roll_max_6"] = df[PRICE_COL].rolling(6).max()
df["roll_min_6"] = df[PRICE_COL].rolling(6).min()

range_width_6 = df["roll_max_6"] - df["roll_min_6"]
df["range_pos_6"] = np.where(
    range_width_6 == 0,
    np.nan,
    (df[PRICE_COL] - df["roll_min_6"]) / range_width_6
)
# 7) milk 관련 feature
df["ret_milk_3m"] = df[MILK_PRICE_COL].pct_change(3)
df["ret_milk_6m"] = df[MILK_PRICE_COL].pct_change(6)

# cheese와 milk의 상대 움직임
df["cheese_milk_spread"] = df["ret"] - df["milk_ret"]

# =========================
# 4) 계절성 피처 (월)
# =========================
df["month"] = df[DATE_COL].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# =========================
# 5) 결측 제거 (shift/rolling로 생긴 NaN 제거)
# =========================
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
# =========================
# 6) 시간순 Train / Holdout / Search / Final 분리
# =========================
# train vs holdout 그리고  holdout 안에서 search vs final
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
# =========================
# 7) 모델 학습 (Logistic Regression)
# =========================
#D층에서 분리한 X_train, y_train으로 실제 모델을 학습시키고, 
# 학습된 모델이 어떤 feature를 얼마나 강하게 쓰는지 1차적으로 확인하는 층이다.
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

# =========================
# 8) 예측 & 정책 탐색 (search holdout)
# =========================
# threshold 탐색은 확률 벡터가 있어야 가능하다.
   # 이 구간에서 처음으로 “학습된 모델”이 실제 평가 구간에 대해 확률을 뱉는다.
    # proba_search는 threshold 탐색으로
      # proba_final은 final single policy 시뮬레이션으로 넘어간다.
proba_search = model.predict_proba(X_search)[:, 1]
proba_final = model.predict_proba(X_final)[:, 1]

# 최종평가 구간을 downstream 호환용 test alias(같은 내용을 별칭으로 넣는것)로 둠
  # final holdout을 test라는 별칭으로 다시 부르는 블록
X_test = X_final.copy()
y_test = y_final.copy()
proba_test = proba_final.copy()
pred_test = (proba_test >= PROB_THRESHOLD).astype(int)
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
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        signal_count = int(y_pred.sum())
        signals_per_year = calc_signals_per_year_from_count(signal_count, months_per_test)
        total_cost = calc_total_cost(fp=fp, fn=fn, signal_count=signal_count)

        rows.append({
            "threshold": float(t),
            "pred_pos": signal_count,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "FPR": round(fpr, 3),
            "signals_per_year_est": None if signals_per_year is None else round(float(signals_per_year), 2),
            "total_cost": round(float(total_cost), 3),
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
plt.show()


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
# =========================
# 9) Regime Detection (KMeans, train fit only)
# =========================
#레짐으로 판단하는 세가지 피처 이건 내가 설정한거임.! 최근흐름, 변동성, 평균대비가격위치
regime_features = [
    "ret_3m",
    "vol_3m",
    "price_ma6_ratio",
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



# =========================
# 11) 마지막 달 기준 forecast 출력
# =========================
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

decision = bool(next_up_proba >= PROB_THRESHOLD)
print(f"Decision flag: {decision}")


print(f"Decision (threshold {PROB_THRESHOLD}): {'BUY/ACCELERATE' if decision else 'HOLD/WAIT'}")

print_section("Label Distribution (full data)")
print(df_model["y"].value_counts(normalize=True).sort_index())