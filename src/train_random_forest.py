import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def find_best_f1_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx]

def evaluate_predictions(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Threshold": float(threshold)
    }

def prepare_features(df):
    print("Preparing tabular and graph features...")
    ts = pd.to_datetime(df["Timestamp"], format="mixed")
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute

    df["log_amount_paid"] = np.log1p(df["Amount Paid"].astype(float))
    df["log_amount_received"] = np.log1p(df["Amount Received"].astype(float))
    df["currency_mismatch"] = (df["Receiving Currency"] != df["Payment Currency"]).astype(int)

    for cat_col in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        df[cat_col + "_code"] = pd.Categorical(df[cat_col]).codes

    tabular_cols = [
        "hour",
        "minute",
        "log_amount_paid",
        "log_amount_received",
        "currency_mismatch",
        "Receiving Currency_code",
        "Payment Currency_code",
        "Payment Format_code"
    ]

    emb_cols = [col for col in df.columns if col.startswith("emb_")]
    graph_cols = ["anomaly_score"] + emb_cols
    hybrid_cols = tabular_cols + graph_cols

    return df, tabular_cols, hybrid_cols

def save_evaluation_plots(df_results, feature_importances, hybrid_cols, out_dir="reports/figures"):
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Ablation Study Bar Chart (Baseline Tabular RF vs Hybrid GAE + RF)
    metrics_to_plot = ["Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    base_vals = [df_results.loc[0, m] for m in metrics_to_plot]
    hyb_vals = [df_results.loc[1, m] for m in metrics_to_plot]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline (Pure Tabular RF)", color="#7f7f7f")
    rects2 = ax.bar(x + width/2, hyb_vals, width, label="Proposed Hybrid (GAE + RF)", color="#1f77b4")

    ax.set_ylabel("Score")
    ax.set_title("Figure 4: Ablation Study - Pure Tabular RF vs Proposed Hybrid GAE + RF")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig4_path = os.path.join(out_dir, "fig4_ablation_performance_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # Plot 2: Top 12 Feature Importances in Hybrid Model
    idx_sorted = np.argsort(feature_importances)[::-1][:12]
    top_names = [hybrid_cols[i] for i in idx_sorted][::-1]
    top_vals = feature_importances[idx_sorted][::-1]
    colors = ["#d62728" if ("emb_" in name or name == "anomaly_score") else "#1f77b4" for name in top_names]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.barh(top_names, top_vals, color=colors, alpha=0.85)
    ax.set_xlabel("Relative Importance Score")
    ax.set_title("Figure 5: Top 12 Feature Importances in Hybrid GAE + RF (Red = GAE Features)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig5_path = os.path.join(out_dir, "fig5_top_feature_importances.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    print(f"Saved evaluation figures to {fig4_path} and {fig5_path}")

def run_stage5_random_forest(
    fused_csv="data/processed/ibm_aml_fused_features.csv",
    metrics_out_csv="reports/stage5_evaluation_metrics.csv"
):
    print("Starting Stage 5: Supervised Classification (Random Forest) and Ablation Study")
    t_start = time.time()

    if not os.path.exists(fused_csv):
        raise FileNotFoundError(f"Fused dataset not found: {fused_csv}")

    print(f"Loading fused dataset from {fused_csv}...")
    df = pd.read_csv(fused_csv, sep=";")
    df, tabular_cols, hybrid_cols = prepare_features(df)

    y = df["Is Laundering"].values.astype(int)
    X_tab = df[tabular_cols].values.astype(np.float32)
    X_hyb = df[hybrid_cols].values.astype(np.float32)

    # Split dataset into 80% train and 20% test (stratified to preserve fraud ratio)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )

    y_train, y_test = y[train_idx], y[test_idx]

    # Undersample majority class in training set (1:20 ratio) to avoid RAM bottleneck and improve minority learning
    pos_train_idx = train_idx[y_train == 1]
    neg_train_idx = train_idx[y_train == 0]
    rng = np.random.default_rng(42)
    sampled_neg_idx = rng.choice(neg_train_idx, size=min(len(neg_train_idx), len(pos_train_idx) * 25), replace=False)
    balanced_train_idx = np.concatenate([pos_train_idx, sampled_neg_idx])
    rng.shuffle(balanced_train_idx)

    print(f"Total Transactions: {len(df):,} | Test Set: {len(test_idx):,} (Fraud in Test: {int(y_test.sum())})")
    print(f"Training on balanced subset: {len(balanced_train_idx):,} rows ({len(pos_train_idx)} fraud, {len(sampled_neg_idx)} normal)")

    # 1. Baseline Model: Pure Tabular Random Forest
    print("\nTraining Model 1: Baseline Random Forest (Tabular Features Only)...")
    t0 = time.time()
    rf_baseline = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_baseline.fit(X_tab[balanced_train_idx], y[balanced_train_idx])
    base_probs = rf_baseline.predict_proba(X_tab[test_idx])[:, 1]
    base_thresh = find_best_f1_threshold(y_test, base_probs)
    base_metrics = evaluate_predictions(y_test, base_probs, base_thresh)
    base_metrics["Model"] = "Baseline (Tabular RF)"
    base_metrics["Train_Time_Sec"] = round(time.time() - t0, 2)

    # 2. Proposed Model: Two-Stage Hybrid (GAE Features + Tabular RF)
    print("Training Model 2: Proposed Hybrid Framework (GAE Features + Tabular RF)...")
    t1 = time.time()
    rf_hybrid = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf_hybrid.fit(X_hyb[balanced_train_idx], y[balanced_train_idx])
    hyb_probs = rf_hybrid.predict_proba(X_hyb[test_idx])[:, 1]
    hyb_thresh = find_best_f1_threshold(y_test, hyb_probs)
    hyb_metrics = evaluate_predictions(y_test, hyb_probs, hyb_thresh)
    hyb_metrics["Model"] = "Proposed Hybrid (GAE + RF)"
    hyb_metrics["Train_Time_Sec"] = round(time.time() - t1, 2)

    cols_order = [
        "Model", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC",
        "TP", "FP", "FN", "TN", "Threshold", "Train_Time_Sec"
    ]
    df_results = pd.DataFrame([base_metrics, hyb_metrics])[cols_order]

    os.makedirs(os.path.dirname(metrics_out_csv), exist_ok=True)
    df_results.to_csv(metrics_out_csv, index=False)

    save_evaluation_plots(df_results, rf_hybrid.feature_importances_, hybrid_cols)

    f1_gain = (hyb_metrics["F1_Score"] - base_metrics["F1_Score"]) * 100
    pr_gain = (hyb_metrics["PR_AUC"] - base_metrics["PR_AUC"]) * 100

    print("\n___STAGE 5 SUMMARY REPORT___")
    print(df_results[["Model", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]].to_string(index=False))
    print(f"\nF1-Score Improvement : {f1_gain:+.2f}% points")
    print(f"PR-AUC Improvement   : {pr_gain:+.2f}% points")
    print(f"Metrics CSV Saved    : {metrics_out_csv}")
    print(f"Total Runtime        : {time.time() - t_start:.2f} seconds")
    print("Stage 5 completed. Full Two-Stage Hybrid pipeline finished.")

if __name__ == "__main__":
    run_stage5_random_forest()
