import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score,
    log_loss,
)
from time import perf_counter

from logistic_regression import LogisticRegression


sns.set_theme(style="whitegrid", palette="muted")
SEED = 42
rng = np.random.default_rng(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_dataset(n_samples=2000, n_features=20, n_informative=10,
                  class_sep=1.0, noise_flip=0.0, random_state=SEED):
    """Wrapper around make_classification with optional label-flip noise."""
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=max(1, (n_features - n_informative) // 2),
        n_clusters_per_class=1, class_sep=class_sep,
        flip_y=noise_flip, random_state=random_state,
    )
    return X, y


def _split_scale(X, y, test_size=0.2, random_state=SEED):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    return scaler.fit_transform(X_tr), scaler.transform(X_te), y_tr, y_te


def _fit_eval(model, X_tr, y_tr, X_te, y_te):
    t0 = perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = perf_counter() - t0

    proba = model.predict_proba(X_te)
    pred  = model.predict(X_te)
    return {
        "fit_time":  fit_time,
        "accuracy":  accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall":    recall_score(y_te, pred, zero_division=0),
        "f1":        f1_score(y_te, pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, proba),
        "avg_prec":  average_precision_score(y_te, proba),
        "log_loss":  log_loss(y_te, proba),
        "proba":     proba,
        "pred":      pred,
        "losses":    model.losses_,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1 – convergence & training dynamics
# ──────────────────────────────────────────────────────────────────────────────

def exp_convergence(lr_grid=(0.001, 0.01, 0.05, 0.1), n_iter=2000):
    """Compare loss curves for different learning rates."""
    X, y = _make_dataset()
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)

    results = {}
    for lr in lr_grid:
        model = LogisticRegression(lr=lr, n_iter=n_iter)
        model.fit(X_tr, y_tr)
        results[lr] = model.losses_

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for lr, losses in results.items():
        ax.plot(losses, label=f"lr={lr}")
    ax.set(xlabel="Iteration", ylabel="Binary cross-entropy",
           title="Training loss — learning rate comparison")
    ax.legend()

    ax = axes[1]
    for lr, losses in results.items():
        ax.plot(losses, label=f"lr={lr}")
    ax.set_yscale("log")
    ax.set(xlabel="Iteration", ylabel="Loss (log scale)",
           title="Training loss (log scale) — convergence speed")
    ax.legend()

    fig.suptitle("Experiment 1 · Convergence dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2 – threshold analysis (ROC, PR, operating point)
# ──────────────────────────────────────────────────────────────────────────────

def exp_threshold_analysis():
    """ROC + PR curves and the effect of decision threshold on all metrics."""
    X, y = _make_dataset()
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)

    model = LogisticRegression(lr=0.05, n_iter=2000)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)

    fpr, tpr, roc_thresh = roc_curve(y_te, proba)
    prec, rec, pr_thresh  = precision_recall_curve(y_te, proba)

    thresholds = np.linspace(0.01, 0.99, 200)
    accs, precs, recs, f1s = [], [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        accs.append(accuracy_score(y_te, pred))
        precs.append(precision_score(y_te, pred, zero_division=0))
        recs.append(recall_score(y_te, pred, zero_division=0))
        f1s.append(f1_score(y_te, pred, zero_division=0))

    best_t = thresholds[np.argmax(f1s)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.fill_between(fpr, tpr, alpha=0.1)
    ax.set(xlabel="FPR", ylabel="TPR",
           title=f"ROC curve  (AUC = {roc_auc_score(y_te, proba):.3f})")

    ax = axes[1]
    ax.plot(rec, prec, lw=2, color="darkorange")
    ax.fill_between(rec, prec, alpha=0.1, color="darkorange")
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"PR curve  (AP = {average_precision_score(y_te, proba):.3f})")

    ax = axes[2]
    ax.plot(thresholds, accs,  label="Accuracy")
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, recs,  label="Recall")
    ax.plot(thresholds, f1s,   label="F1", lw=2.5)
    ax.axvline(best_t, color="red", linestyle="--",
               label=f"Best F1 threshold ({best_t:.2f})")
    ax.set(xlabel="Decision threshold", ylabel="Score",
           title="Metrics vs. decision threshold")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 2 · Threshold analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3 – cross-validated performance + confidence intervals
# ──────────────────────────────────────────────────────────────────────────────

def exp_cross_validation(n_splits=10):
    """StratifiedKFold — distribution of metrics with 95 % CI."""
    X, y = _make_dataset(n_samples=3000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fold_scores  = {m: [] for m in metric_names}

    for tr_idx, te_idx in skf.split(X_scaled, y):
        model = LogisticRegression(lr=0.05, n_iter=1000)
        model.fit(X_scaled[tr_idx], y[tr_idx])
        proba = model.predict_proba(X_scaled[te_idx])
        pred  = model.predict(X_scaled[te_idx])
        y_te  = y[te_idx]

        fold_scores["accuracy"].append(accuracy_score(y_te, pred))
        fold_scores["precision"].append(precision_score(y_te, pred, zero_division=0))
        fold_scores["recall"].append(recall_score(y_te, pred, zero_division=0))
        fold_scores["f1"].append(f1_score(y_te, pred, zero_division=0))
        fold_scores["roc_auc"].append(roc_auc_score(y_te, proba))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    data_list = [fold_scores[m] for m in metric_names]
    bp = ax.boxplot(data_list, patch_artist=True, labels=metric_names, widths=0.5)
    palette = sns.color_palette("muted", len(metric_names))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax.set(ylabel="Score", title=f"{n_splits}-fold CV — metric distributions")

    ax = axes[1]
    means = [np.mean(fold_scores[m]) for m in metric_names]
    # 95 % CI via t-distribution
    cis   = [stats.t.interval(0.95, df=n_splits - 1,
                               loc=np.mean(fold_scores[m]),
                               scale=stats.sem(fold_scores[m]))
             for m in metric_names]
    errs  = [m - ci[0] for m, ci in zip(means, cis)]

    x = np.arange(len(metric_names))
    bars = ax.bar(x, means, yerr=errs, capsize=6,
                  color=palette, edgecolor="white", width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_ylim(max(0, min(means) - 0.1), 1.05)
    ax.set(ylabel="Score", title="Mean ± 95% CI")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Experiment 3 · Cross-validated performance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 4 – sensitivity to dataset difficulty
# ──────────────────────────────────────────────────────────────────────────────

def exp_dataset_difficulty():
    """Sweep class_sep (separability) and label noise independently."""
    sep_grid   = np.linspace(0.3, 3.0, 8)
    noise_grid = np.linspace(0.0, 0.3, 8)

    def _score(n_samples=1500, n_features=20, class_sep=1.0, noise_flip=0.0):
        X, y = _make_dataset(n_samples=n_samples, n_features=n_features,
                             class_sep=class_sep, noise_flip=noise_flip)
        X_tr, X_te, y_tr, y_te = _split_scale(X, y)
        model = LogisticRegression(lr=0.05, n_iter=1000)
        model.fit(X_tr, y_tr)
        return {
            "roc_auc": roc_auc_score(y_te, model.predict_proba(X_te)),
            "f1":      f1_score(y_te, model.predict(X_te), zero_division=0),
        }

    sep_results   = [_score(class_sep=s) for s in sep_grid]
    noise_results = [_score(noise_flip=n) for n in noise_grid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(sep_grid, [r["roc_auc"] for r in sep_results], marker="o", label="ROC-AUC")
    ax.plot(sep_grid, [r["f1"]      for r in sep_results], marker="s", label="F1")
    ax.set(xlabel="class_sep", ylabel="Score",
           title="Performance vs. class separability")
    ax.legend()

    ax = axes[1]
    ax.plot(noise_grid * 100, [r["roc_auc"] for r in noise_results], marker="o", label="ROC-AUC")
    ax.plot(noise_grid * 100, [r["f1"]      for r in noise_results], marker="s", label="F1")
    ax.set(xlabel="Label noise (%)", ylabel="Score",
           title="Performance vs. label noise")
    ax.legend()

    fig.suptitle("Experiment 4 · Dataset difficulty", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 5 – confusion matrix + probability calibration
# ──────────────────────────────────────────────────────────────────────────────

def exp_confusion_and_calibration():
    """Confusion matrix heatmap and reliability (calibration) diagram."""
    X, y = _make_dataset(n_samples=3000)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)

    model = LogisticRegression(lr=0.05, n_iter=2000)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    pred  = model.predict(X_te)

    # Calibration: bin predicted proba, compare to fraction of positives
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(proba, bins[1:-1])
    bin_mean_pred = [proba[bin_ids == b].mean() if (bin_ids == b).any() else np.nan
                     for b in range(n_bins)]
    bin_frac_pos  = [y_te[bin_ids == b].mean()  if (bin_ids == b).any() else np.nan
                     for b in range(n_bins)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    cm = confusion_matrix(y_te, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"], ax=ax, linewidths=0.5)
    for (i, j), raw in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.7, f"n={raw}", ha="center", va="center",
                fontsize=8, color="gray")
    ax.set(title="Confusion matrix (row-normalised)")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(bin_mean_pred, bin_frac_pos, marker="o", lw=2, label="Model")
    ax.fill_between(bin_mean_pred, bin_frac_pos, bin_mean_pred,
                    alpha=0.15, color="steelblue", label="Gap")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title="Reliability diagram (calibration)", xlim=(0, 1), ylim=(0, 1))
    ax.legend()

    fig.suptitle("Experiment 5 · Confusion matrix & calibration", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 6 – scalability (n_samples × n_features)
# ──────────────────────────────────────────────────────────────────────────────

def exp_scalability():
    """Fit-time and AUC heatmap over a (n_samples, n_features) grid."""
    sample_grid  = [200, 500, 1000, 2000, 5000]
    feature_grid = [5, 10, 20, 50, 100]

    time_mat = np.zeros((len(sample_grid), len(feature_grid)))
    auc_mat  = np.zeros_like(time_mat)

    for i, n in enumerate(sample_grid):
        for j, d in enumerate(feature_grid):
            X, y = _make_dataset(n_samples=n, n_features=d,
                                 n_informative=max(2, d // 2))
            X_tr, X_te, y_tr, y_te = _split_scale(X, y)
            model = LogisticRegression(lr=0.05, n_iter=500)
            t0 = perf_counter()
            model.fit(X_tr, y_tr)
            time_mat[i, j] = perf_counter() - t0
            auc_mat[i, j]  = roc_auc_score(y_te, model.predict_proba(X_te))

    row_labels = [str(n) for n in sample_grid]
    col_labels = [str(d) for d in feature_grid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(time_mat, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=col_labels, yticklabels=row_labels,
                ax=axes[0], linewidths=0.4)
    axes[0].set(xlabel="n_features", ylabel="n_samples", title="Fit time (s)")

    sns.heatmap(auc_mat, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=col_labels, yticklabels=row_labels,
                ax=axes[1], linewidths=0.4, vmin=0.5, vmax=1.0)
    axes[1].set(xlabel="n_features", ylabel="n_samples", title="ROC-AUC")

    fig.suptitle("Experiment 6 · Scalability heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_LinearRegression():
    print("Running logistic regression evaluation …\n")

    experiments = [
        ("Convergence dynamics",          exp_convergence),
        ("Threshold analysis",            exp_threshold_analysis),
        ("Cross-validated performance",   exp_cross_validation),
        ("Dataset difficulty",            exp_dataset_difficulty),
        ("Confusion matrix & calibration",exp_confusion_and_calibration),
        ("Scalability",                   exp_scalability),
    ]

    for name, fn in experiments:
        print(f"  -> {name} …")
        fig = fn()
        # fig.savefig(f"eval_{fn.__name__}.png", dpi=140, bbox_inches="tight")
        # print(f"    saved  eval_{fn.__name__}.png")

    print("\nDone. Call plt.show() or open the saved PNGs.")
    plt.show()

def main():
    evaluate_LinearRegression()


if __name__ == "__main__":
    main()