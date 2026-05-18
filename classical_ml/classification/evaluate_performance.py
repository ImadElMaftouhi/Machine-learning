import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.figure as mplfig
import seaborn as sns
import argparse

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score, log_loss,
)
from time import perf_counter

from logistic_regression import LogisticRegression
from knn import KNN
from naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

sns.set_theme(style="whitegrid", palette="muted")
SEED = 42

# Number of classes used per type
N_CLASSES = {"binary": 2, "multinomial": 4, "ordinal": 5}

VALID_TYPES = ("binary", "multinomial", "ordinal", "all")


# ----------------------
# Helpers
# ----------------------

def _make_dataset(cls_type, n_samples=2000, n_features=20, n_informative=10,
                  class_sep=1.0, noise_flip=0.0, random_state=SEED):
    n_cls        = N_CLASSES[cls_type]
    n_informative = max(n_informative, n_cls)           # sklearn constraint
    n_redundant   = max(0, (n_features - n_informative) // 2)
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        n_clusters_per_class=1, n_classes=n_cls,
        class_sep=class_sep, flip_y=noise_flip,
        random_state=random_state,
    )
    return X, y


def _split_scale(X, y, test_size=0.2, random_state=SEED):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te


def _compute_metrics(cls_type, y_te, pred, proba):
    """Return a flat dict of scalar metrics, adapted to cls_type."""
    avg = "binary" if cls_type == "binary" else "macro"

    base = {
        "accuracy":  accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, average=avg, zero_division=0),
        "recall":    recall_score(y_te, pred, average=avg, zero_division=0),
        "f1":        f1_score(y_te, pred, average=avg, zero_division=0),
        "log_loss":  log_loss(y_te, proba),
    }

    if cls_type == "binary":
        base["roc_auc"]  = roc_auc_score(y_te, proba)
        base["avg_prec"] = average_precision_score(y_te, proba)
    else:
        base["roc_auc"] = roc_auc_score(
            y_te, proba, multi_class="ovr", average="macro"
        )
        if cls_type == "ordinal":
            # Classes are assumed to be consecutive integers 0..K-1
            classes  = np.unique(y_te)
            pred_idx = np.searchsorted(classes, pred)
            true_idx = np.searchsorted(classes, y_te)
            base["ordinal_mae"] = float(np.mean(np.abs(pred_idx - true_idx)))

    return base


def _fit_eval(cls_type, model, X_tr, y_tr, X_te, y_te):
    t0 = perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = perf_counter() - t0

    proba = model.predict_proba(X_te)
    pred  = model.predict(X_te)
    return {"fit_time": fit_time, **_compute_metrics(cls_type, y_te, pred, proba),
            "proba": proba, "pred": pred, "losses": model.losses_}


def _new_model(cls_type, lr=0.05, n_iter=1000):
    return LogisticRegression(lr=lr, n_iter=n_iter, type=cls_type)


# ----------------------
# Experiment 1 – convergence & training dynamics
# ----------------------

def exp_convergence(cls_type, lr_grid=(0.001, 0.01, 0.05, 0.1), n_iter=2000):
    X, y   = _make_dataset(cls_type)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)

    results = {}
    for lr in lr_grid:
        m = _new_model(cls_type, lr=lr, n_iter=n_iter)
        m.fit(X_tr, y_tr)
        results[lr] = m.losses_

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, yscale in zip(axes, ("linear", "log")):
        for lr, losses in results.items():
            ax.plot(losses, label=f"lr={lr}")
        ax.set_yscale(yscale)
        ax.set(xlabel="Iteration",
               ylabel="Loss" + (" (log scale)" if yscale == "log" else ""),
               title=f"Training loss — lr comparison ({yscale})")
        ax.legend()

    fig.suptitle(f"Exp 1 · Convergence  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Experiment 2 – decision boundary / class confidence analysis
# ----------------------

def exp_decision_analysis(cls_type):
    X, y   = _make_dataset(cls_type)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)
    model  = _new_model(cls_type, lr=0.05, n_iter=2000)
    model.fit(X_tr, y_tr)
    proba  = model.predict_proba(X_te)
    pred   = model.predict(X_te)
    classes = np.unique(y_te)
    K       = len(classes)

    if cls_type == "binary":
        # ── ROC / PR / threshold sweep ────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_te, proba)
        pr_p, pr_r, _ = precision_recall_curve(y_te, proba)
        thresholds  = np.linspace(0.01, 0.99, 200)
        accs, precs, recs, f1s = [], [], [], []
        for t in thresholds:
            p = (proba >= t).astype(int)
            accs.append(accuracy_score(y_te, p))
            precs.append(precision_score(y_te, p, zero_division=0))
            recs.append(recall_score(y_te, p, zero_division=0))
            f1s.append(f1_score(y_te, p, zero_division=0))
        best_t = thresholds[np.argmax(f1s)]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].plot(fpr, tpr, lw=2)
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].fill_between(fpr, tpr, alpha=0.1)
        axes[0].set(xlabel="FPR", ylabel="TPR",
                    title=f"ROC  (AUC={roc_auc_score(y_te, proba):.3f})")

        axes[1].plot(pr_r, pr_p, lw=2, color="darkorange")
        axes[1].fill_between(pr_r, pr_p, alpha=0.1, color="darkorange")
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title=f"PR  (AP={average_precision_score(y_te, proba):.3f})")

        axes[2].plot(thresholds, accs,  label="Accuracy")
        axes[2].plot(thresholds, precs, label="Precision")
        axes[2].plot(thresholds, recs,  label="Recall")
        axes[2].plot(thresholds, f1s,   label="F1", lw=2.5)
        axes[2].axvline(best_t, color="red", ls="--",
                        label=f"Best-F1 t={best_t:.2f}")
        axes[2].set(xlabel="Threshold", ylabel="Score",
                    title="Metrics vs. decision threshold")
        axes[2].legend(fontsize=8)

    else:
        # ── Per-class OvR ROC + confidence histogram + per-class F1 ──────────
        Y_bin  = np.asarray(label_binarize(y_te, classes=classes))  # (n, K)
        palette = sns.color_palette("muted", K)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Per-class ROC curves
        ax = axes[0]
        for k in range(K):
            fpr, tpr, _ = roc_curve(Y_bin[:, k], proba[:, k])
            auc = roc_auc_score(Y_bin[:, k], proba[:, k])
            ax.plot(fpr, tpr, color=palette[k], label=f"class {classes[k]}  AUC={auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set(xlabel="FPR", ylabel="TPR", title="Per-class OvR ROC curves")
        ax.legend(fontsize=8)

        # Confidence histogram (max predicted probability)
        ax = axes[1]
        max_conf = proba.max(axis=1)
        correct  = (pred == y_te)
        ax.hist(max_conf[correct],  bins=25, alpha=0.6, label="Correct",   color="steelblue")
        ax.hist(max_conf[~correct], bins=25, alpha=0.6, label="Incorrect", color="crimson")
        ax.set(xlabel="Max predicted probability", ylabel="Count",
               title="Confidence distribution: correct vs. incorrect")
        ax.legend()

        # Per-class F1
        ax = axes[2]
        f1s = np.asarray(f1_score(y_te, pred, average=None, labels=classes, zero_division=0))
        bars = ax.bar(range(K), f1s, color=palette, edgecolor="white", width=0.6)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"class {c}" for c in classes])
        ax.set(ylabel="F1 score", title="Per-class F1 scores", ylim=(0, 1.05))
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Exp 2 · Decision analysis  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Experiment 3 – cross-validated performance + confidence intervals
# ----------------------

def exp_cross_validation(cls_type, n_splits=10):
    X, y = _make_dataset(cls_type, n_samples=3000)
    X_sc = StandardScaler().fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    metric_names = (["accuracy", "precision", "recall", "f1", "roc_auc"]
                    + (["ordinal_mae"] if cls_type == "ordinal" else []))
    fold_scores  = {m: [] for m in metric_names}

    for tr_idx, te_idx in skf.split(X_sc, y):
        m = _new_model(cls_type)
        m.fit(X_sc[tr_idx], y[tr_idx])
        metrics = _compute_metrics(cls_type, y[te_idx],
                                   m.predict(X_sc[te_idx]),
                                   m.predict_proba(X_sc[te_idx]))
        for name in metric_names:
            fold_scores[name].append(metrics[name])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = sns.color_palette("muted", len(metric_names))

    ax = axes[0]
    bp = ax.boxplot([fold_scores[m] for m in metric_names],
                    patch_artist=True, labels=metric_names, widths=0.5)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax.set(ylabel="Score", title=f"{n_splits}-fold CV — metric distributions")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    means = [np.mean(fold_scores[m]) for m in metric_names]
    cis   = [stats.t.interval(0.95, df=n_splits - 1,
                               loc=np.mean(fold_scores[m]),
                               scale=stats.sem(fold_scores[m]))
             for m in metric_names]
    errs  = [mu - ci[0] for mu, ci in zip(means, cis)]
    x = np.arange(len(metric_names))
    bars = ax.bar(x, means, yerr=errs, capsize=6, color=palette,
                  edgecolor="white", width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(metric_names, rotation=15)
    ax.set(ylabel="Score", title="Mean ± 95% CI")
    for bar, mu in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mu:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Exp 3 · Cross-validation  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Experiment 4 – sensitivity to dataset difficulty
# ----------------------

def exp_dataset_difficulty(cls_type):
    sep_grid   = np.linspace(0.3, 3.0, 8)
    noise_grid = np.linspace(0.0, 0.25, 8)

    def _score(class_sep=1.0, noise_flip=0.0):
        X, y = _make_dataset(cls_type, n_samples=1500, class_sep=class_sep,
                             noise_flip=noise_flip)
        X_tr, X_te, y_tr, y_te = _split_scale(X, y)
        m = _new_model(cls_type)
        m.fit(X_tr, y_tr)
        metrics = _compute_metrics(cls_type, y_te, m.predict(X_te),
                                   m.predict_proba(X_te))
        return metrics

    sep_results   = [_score(class_sep=s)   for s in sep_grid]
    noise_results = [_score(noise_flip=nf) for nf in noise_grid]

    n_panels = 3 if cls_type == "ordinal" else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    ax = axes[0]
    ax.plot(sep_grid, [r["roc_auc"] for r in sep_results], marker="o", label="ROC-AUC")
    ax.plot(sep_grid, [r["f1"]      for r in sep_results], marker="s", label="F1 (macro)")
    ax.set(xlabel="class_sep", ylabel="Score",
           title="Performance vs. class separability")
    ax.legend()

    ax = axes[1]
    ax.plot(noise_grid * 100, [r["roc_auc"] for r in noise_results], marker="o", label="ROC-AUC")
    ax.plot(noise_grid * 100, [r["f1"]      for r in noise_results], marker="s", label="F1 (macro)")
    ax.set(xlabel="Label noise (%)", ylabel="Score",
           title="Performance vs. label noise")
    ax.legend()

    if cls_type == "ordinal":
        ax = axes[2]
        ax.plot(sep_grid,       [r["ordinal_mae"] for r in sep_results],
                marker="o", label="vs sep", color="purple")
        ax2 = ax.twinx()
        ax2.plot(noise_grid * 100, [r["ordinal_mae"] for r in noise_results],
                 marker="s", ls="--", label="vs noise", color="darkorange")
        ax.set(xlabel="class_sep / label noise (%)", ylabel="Ordinal MAE",
               title="Ordinal MAE vs. difficulty")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    fig.suptitle(f"Exp 4 · Dataset difficulty  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Experiment 5 – confusion matrix + calibration
# ----------------------

def exp_confusion_and_calibration(cls_type):
    X, y = _make_dataset(cls_type, n_samples=3000)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)
    model = _new_model(cls_type, lr=0.05, n_iter=2000)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)
    pred  = model.predict(X_te)
    classes = np.unique(y_te)
    K = len(classes)

    n_cols = 3 if cls_type == "ordinal" else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    ax = axes[0]
    cm      = confusion_matrix(y_te, pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels  = [str(c) for c in classes]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5)
    for (i, j), raw in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.72, f"n={raw}", ha="center", va="center",
                fontsize=7, color="gray")
    ax.set(title="Confusion matrix (row-normalised)",
           xlabel="Predicted", ylabel="True")

    # ── Calibration (reliability diagram) ────────────────────────────────────
    ax    = axes[1]
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)
    palette = sns.color_palette("muted", K)

    if cls_type == "binary":
        p1d = proba
        bin_ids   = np.digitize(p1d, bins[1:-1])
        mean_pred = [p1d[bin_ids == b].mean()  if (bin_ids == b).any() else np.nan for b in range(n_bins)]
        frac_pos  = [y_te[bin_ids == b].mean() if (bin_ids == b).any() else np.nan for b in range(n_bins)]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label="Model")
        ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color="steelblue")
    else:
        # One OvR reliability curve per class
        Y_bin = np.asarray(label_binarize(y_te, classes=classes))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        for k in range(K):
            pk       = proba[:, k]
            bin_ids  = np.digitize(pk, bins[1:-1])
            mean_pred = [pk[bin_ids == b].mean()       if (bin_ids == b).any() else np.nan for b in range(n_bins)]
            frac_pos  = [Y_bin[bin_ids == b, k].mean() if (bin_ids == b).any() else np.nan for b in range(n_bins)]
            ax.plot(mean_pred, frac_pos, marker="o", lw=1.5,
                    color=palette[k], label=f"class {classes[k]}")

    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)

    # ── Ordinal error distribution ────────────────────────────────────────────
    if cls_type == "ordinal":
        ax = axes[2]
        pred_idx = np.searchsorted(classes, pred)
        true_idx = np.searchsorted(classes, y_te)
        errors   = pred_idx - true_idx
        unique_e, counts = np.unique(errors, return_counts=True)
        ax.bar(unique_e, counts / len(errors), color="mediumpurple",
               edgecolor="white", width=0.6)
        ax.axvline(0, color="black", ls="--", alpha=0.5)
        ax.set(xlabel="Predicted class − True class (ordinal steps)",
               ylabel="Proportion", title=f"Ordinal error distribution  (MAE={np.mean(np.abs(errors)):.3f})")
        ax.set_xticks(unique_e)

    fig.suptitle(f"Exp 5 · Confusion & calibration  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Experiment 6 – scalability (n_samples × n_features)
# ----------------------

def exp_scalability(cls_type):
    sample_grid  = [200, 500, 1000, 2000, 5000]
    feature_grid = [5, 10, 20, 50, 100]
    n_cls = N_CLASSES[cls_type]

    time_mat = np.zeros((len(sample_grid), len(feature_grid)))
    auc_mat  = np.zeros_like(time_mat)

    for i, n in enumerate(sample_grid):
        for j, d in enumerate(feature_grid):
            n_inf = max(n_cls, min(d, d // 2))
            X, y  = _make_dataset(cls_type, n_samples=n, n_features=d,
                                  n_informative=n_inf)
            X_tr, X_te, y_tr, y_te = _split_scale(X, y)
            m = _new_model(cls_type, n_iter=500)
            t0 = perf_counter()
            m.fit(X_tr, y_tr)
            time_mat[i, j] = perf_counter() - t0
            proba = m.predict_proba(X_te)
            if cls_type == "binary":
                auc_mat[i, j] = roc_auc_score(y_te, proba)
            else:
                auc_mat[i, j] = roc_auc_score(y_te, proba,
                                               multi_class="ovr", average="macro")

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
    axes[1].set(xlabel="n_features", ylabel="n_samples",
                title="ROC-AUC (macro OvR)")

    fig.suptitle(f"Exp 6 · Scalability  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Logistic Regression Entry point
# ----------------------

def evaluate_LogisticRegression(cls_type: str = "binary") -> None:
    if cls_type not in VALID_TYPES:
        raise ValueError(f"cls_type must be one of {VALID_TYPES}. Got {cls_type!r}.")
    if cls_type == "all":
        for t in VALID_TYPES[:-1]:  # exclude "all"
            evaluate_LogisticRegression(cls_type=t)
        return
    
    print(f"Evaluating LogisticRegression  [type={cls_type}]  "
          f"({N_CLASSES[cls_type]} classes)\n")

    experiments = [
        ("Convergence dynamics",        lambda: exp_convergence(cls_type)),
        ("Decision analysis",           lambda: exp_decision_analysis(cls_type)),
        ("Cross-validated performance", lambda: exp_cross_validation(cls_type)),
        ("Dataset difficulty",          lambda: exp_dataset_difficulty(cls_type)),
        ("Confusion & calibration",     lambda: exp_confusion_and_calibration(cls_type)),
        ("Scalability",                 lambda: exp_scalability(cls_type)),
    ]

    for name, fn in experiments:
        print(f"  -> {name} …")
        fig = fn()
        # fname = f"eval_{cls_type}_{name.lower().replace(' ', '_')}.png"
        # fig.savefig(fname, dpi=140, bbox_inches="tight")
        # print(f"     saved {fname}")

    print("\nDone.")
    plt.show()

# ----------------------
# KNN helpers
# ----------------------

def _knn_eval(cls_type: str, model: KNN, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit-then-evaluate helper for KNN. Records predict time instead of fit time."""
    model.fit(X_tr, y_tr)
    t0    = perf_counter()
    proba = model.predict_proba(X_te)
    pred_time = perf_counter() - t0
    pred  = model.predict(X_te)
    return {"pred_time": pred_time,
            **_compute_metrics(cls_type, y_te, pred, proba),
            "proba": proba, "pred": pred}


# ----------------------
# KNN Experiment 1 – k sweep (bias-variance tradeoff)
# ----------------------

def exp_knn_k_sweep(cls_type: str, k_max: int = 40):
    """
    Vary k from 1 to k_max and track train/test accuracy plus AUC and F1.
    Small k → high variance (overfits noise).
    Large k → high bias (decision boundary too smooth).
    """
    X, y = _make_dataset(cls_type, n_samples=1500)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)

    k_grid = list(range(1, k_max + 1))
    train_accs, test_accs, test_f1s, test_aucs = [], [], [], []

    for k in k_grid:
        m = KNN(k=k)
        m.fit(X_tr, y_tr)
        train_accs.append(m.score(X_tr, y_tr))
        proba   = m.predict_proba(X_te)
        pred    = m.predict(X_te)
        metrics = _compute_metrics(cls_type, y_te, pred, proba)
        test_accs.append(metrics["accuracy"])
        test_f1s.append(metrics["f1"])
        test_aucs.append(metrics["roc_auc"])

    best_k = k_grid[int(np.argmax(test_f1s))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(k_grid, train_accs, ls="--", label="Train accuracy")
    ax.plot(k_grid, test_accs,  label="Test accuracy")
    ax.axvline(best_k, color="red", ls=":", alpha=0.7, label=f"Best k={best_k}")
    ax.fill_between(k_grid,
                    [tr - te for tr, te in zip(train_accs, test_accs)],
                    alpha=0.08, color="red", label="Overfit gap")
    ax.set(xlabel="k", ylabel="Accuracy", title="Train vs. test accuracy — variance at k=1")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(k_grid, test_f1s,  marker=".", ms=4, label="F1 (macro)")
    ax.plot(k_grid, test_aucs, marker=".", ms=4, label="ROC-AUC")
    ax.axvline(best_k, color="red", ls=":", alpha=0.7, label=f"Best k={best_k}")
    ax.set(xlabel="k", ylabel="Score", title="Test metrics vs. k")
    ax.legend(fontsize=9)

    fig.suptitle(f"KNN Exp 1 · k sweep  [{cls_type}]", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# KNN Experiment 2 – distance metric + weighting comparison
# ----------------------

def exp_knn_distance_comparison(cls_type: str, k: int = 5):
    """Compare euclidean / manhattan / cosine and uniform / distance weighting."""
    X, y = _make_dataset(cls_type, n_samples=1500)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)

    distances   = ["euclidean", "manhattan", "cosine"]
    weight_opts = ["uniform", "distance"]
    metric_cols = ["accuracy", "f1", "roc_auc"]
    palette     = sns.color_palette("muted", len(distances))

    # ── left: metric comparison by distance ──
    results_dist: dict[str, dict] = {}
    for dist in distances:
        m = KNN(k=k, distance=dist)
        r = _knn_eval(cls_type, m, X_tr, y_tr, X_te, y_te)
        results_dist[dist] = r

    # ── right: uniform vs distance-weighted ──
    results_w: dict[str, dict] = {}
    for w in weight_opts:
        m = KNN(k=k, weights=w)
        r = _knn_eval(cls_type, m, X_tr, y_tr, X_te, y_te)
        results_w[w] = r

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax  = axes[0]
    x   = np.arange(len(metric_cols))
    bar_w = 0.25
    for i, dist in enumerate(distances):
        vals = [results_dist[dist][mc] for mc in metric_cols]
        bars = ax.bar(x + i * bar_w, vals, width=bar_w, label=dist, color=palette[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(metric_cols)
    ax.set(ylabel="Score", ylim=(0, 1.1),
           title=f"Distance metric comparison  (k={k})")
    ax.legend()

    ax   = axes[1]
    x2   = np.arange(len(metric_cols))
    bar_w2 = 0.35
    pal2 = sns.color_palette("muted", 2)
    for i, w in enumerate(weight_opts):
        vals = [results_w[w][mc] for mc in metric_cols]
        bars = ax.bar(x2 + i * bar_w2, vals, width=bar_w2, label=w, color=pal2[i])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x2 + bar_w2 / 2)
    ax.set_xticklabels(metric_cols)
    ax.set(ylabel="Score", ylim=(0, 1.1),
           title=f"Uniform vs. distance-weighted voting  (k={k})")
    ax.legend()

    fig.suptitle(f"KNN Exp 2 · Distance & weighting  [{cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# KNN Experiment 3 – confusion matrix + cross-validated metrics
# ----------------------

def exp_knn_confusion(cls_type: str, k: int = 5):
    X, y = _make_dataset(cls_type, n_samples=2000)
    X_tr, X_te, y_tr, y_te = _split_scale(X, y)
    r       = _knn_eval(cls_type, KNN(k=k), X_tr, y_tr, X_te, y_te)
    pred    = r["pred"]
    classes = np.unique(y_te)

    # Cross-validation
    n_splits    = 8
    metric_names = ["accuracy", "f1", "roc_auc"]
    fold_scores  = {m: [] for m in metric_names}
    X_sc = StandardScaler().fit_transform(X)
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, te_idx in skf.split(X_sc, y):
        m = KNN(k=k)
        m.fit(X_sc[tr_idx], y[tr_idx])
        proba_cv = m.predict_proba(X_sc[te_idx])
        pred_cv  = m.predict(X_sc[te_idx])
        cv_m = _compute_metrics(cls_type, y[te_idx], pred_cv, proba_cv)
        for name in metric_names:
            fold_scores[name].append(cv_m[name])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette   = sns.color_palette("muted", len(metric_names))

    # Confusion matrix
    ax = axes[0]
    cm      = confusion_matrix(y_te, pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=[str(c) for c in classes],
                yticklabels=[str(c) for c in classes],
                ax=ax, linewidths=0.5)
    for (i, j), raw in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.72, f"n={raw}", ha="center", va="center",
                fontsize=7, color="gray")
    ax.set(title=f"Confusion matrix  (k={k})", xlabel="Predicted", ylabel="True")

    # CV bar chart with CI
    ax    = axes[1]
    means = [np.mean(fold_scores[m]) for m in metric_names]
    cis   = [stats.t.interval(0.95, df=n_splits - 1,
                               loc=np.mean(fold_scores[m]),
                               scale=stats.sem(fold_scores[m]))
             for m in metric_names]
    errs  = [mu - ci[0] for mu, ci in zip(means, cis)]
    bars  = ax.bar(metric_names, means, yerr=errs, capsize=6,
                   color=palette, edgecolor="white")
    ax.set(ylabel="Score", title=f"{n_splits}-fold CV — mean ± 95% CI",
           ylim=(0, 1.05))
    for bar, mu in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mu:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"KNN Exp 3 · Confusion & CV  [{cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# KNN Experiment 4 – scalability (predict time, not fit time)
# ----------------------

def exp_knn_scalability(cls_type: str, k: int = 5):
    """
    KNN fit is O(1) — just stores data.
    Predict is O(n_train * d) per query, so predict time is what scales badly.
    """
    sample_grid  = [100, 500, 1000, 3000, 5000]
    feature_grid = [5, 10, 20, 50, 100]
    n_cls = N_CLASSES[cls_type]

    pred_time_mat = np.zeros((len(sample_grid), len(feature_grid)))
    auc_mat       = np.zeros_like(pred_time_mat)

    for i, n in enumerate(sample_grid):
        for j, d in enumerate(feature_grid):
            n_inf = max(n_cls, min(d, d // 2))
            X, y  = _make_dataset(cls_type, n_samples=n, n_features=d,
                                  n_informative=n_inf)
            X_tr, X_te, y_tr, y_te = _split_scale(X, y)
            m = KNN(k=k)
            m.fit(X_tr, y_tr)
            t0    = perf_counter()
            proba = m.predict_proba(X_te)
            pred_time_mat[i, j] = perf_counter() - t0
            pred = m.predict(X_te)
            auc_mat[i, j] = _compute_metrics(cls_type, y_te, pred, proba)["roc_auc"]

    row_labels = [str(n) for n in sample_grid]
    col_labels = [str(d) for d in feature_grid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(pred_time_mat, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=col_labels, yticklabels=row_labels,
                ax=axes[0], linewidths=0.4)
    axes[0].set(xlabel="n_features", ylabel="n_train_samples",
                title="Predict time (s)  — grows with n_train × d")

    sns.heatmap(auc_mat, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=col_labels, yticklabels=row_labels,
                ax=axes[1], linewidths=0.4, vmin=0.5, vmax=1.0)
    axes[1].set(xlabel="n_features", ylabel="n_train_samples", title="ROC-AUC")

    fig.suptitle(f"KNN Exp 4 · Scalability  [{cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# KNN entry point
# ----------------------

def evaluate_KNN(cls_type: str = "binary") -> None:
    if cls_type not in VALID_TYPES:
        raise ValueError(f"cls_type must be one of {VALID_TYPES}. Got {cls_type!r}.")
    if cls_type == "all":
        for t in VALID_TYPES[:-1]:
            evaluate_KNN(cls_type=t)
        return

    print(f"Evaluating KNN  [type={cls_type}]  ({N_CLASSES[cls_type]} classes)\n")

    experiments = [
        ("k sweep",                    lambda: exp_knn_k_sweep(cls_type)),
        ("Distance & weighting",       lambda: exp_knn_distance_comparison(cls_type)),
        ("Confusion & CV",             lambda: exp_knn_confusion(cls_type)),
        ("Scalability",                lambda: exp_knn_scalability(cls_type)),
    ]

    for name, fn in experiments:
        print(f"  -> {name} …")
        fig = fn()
        # fname = f"eval_knn_{cls_type}_{name.lower().replace(' & ', '_').replace(' ', '_')}.png"
        # fig.savefig(fname, dpi=140, bbox_inches="tight")
        # print(f"     saved {fname}")

    print("\nDone.")
    plt.show()

# ----------------------
# Naive Bayes helpers
# ----------------------

VALID_NB_VARIANTS = ("gaussian", "bernoulli", "multinomial", "all")


def _new_nb_model(nb_variant: str, **kwargs):
    if nb_variant == "gaussian":
        return GaussianNB(**kwargs)
    elif nb_variant == "bernoulli":
        return BernoulliNB(**kwargs)
    elif nb_variant == "multinomial":
        return MultinomialNB(**kwargs)
    raise ValueError(f"Unknown nb_variant: {nb_variant!r}")


def _prepare_nb_data(X_tr, X_te, nb_variant: str):
    """
    Each NB variant requires a different data contract.
    Data arrives already scaled (zero-mean, unit-variance).

    gaussian    — continuous as-is
    bernoulli   — binary: binarize at 0 (above mean → 1)
    multinomial — non-negative counts: shift so min=0, then scale
    """
    if nb_variant == "gaussian":
        return X_tr, X_te
    elif nb_variant == "bernoulli":
        return (X_tr > 0).astype(float), (X_te > 0).astype(float)
    else:  # multinomial
        shift = X_tr.min()           # shift on train stats only (no leakage)
        return X_tr - shift, X_te - shift


def _nb_eval(cls_type: str, model, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit-then-evaluate for NB. Records fit time (NB has no losses_)."""
    t0 = perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = perf_counter() - t0
    proba = model.predict_proba(X_te)
    pred  = model.predict(X_te)
    return {"fit_time": fit_time,
            **_compute_metrics(cls_type, y_te, pred, proba),
            "proba": proba, "pred": pred}


# ----------------------
# NB Experiment 1 – variant comparison
# ----------------------

def exp_nb_variant_comparison(cls_type: str) -> mplfig.Figure:
    """
    Evaluate all three NB variants on the same dataset.
    Each variant sees data transformed to match its input contract.
    Shows which assumptions fit the (continuous) data best.
    """
    X, y = _make_dataset(cls_type, n_samples=2000)
    X_tr_raw, X_te_raw, y_tr, y_te = _split_scale(X, y)

    variants    = ["gaussian", "bernoulli", "multinomial"]
    metric_cols = ["accuracy", "f1", "roc_auc"]
    palette     = sns.color_palette("muted", len(variants))
    results: dict[str, dict] = {}

    for v in variants:
        X_tr, X_te = _prepare_nb_data(X_tr_raw, X_te_raw, v)
        r = _nb_eval(cls_type, _new_nb_model(v), X_tr, y_tr, X_te, y_te)
        results[v] = r

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── grouped bar: metrics ──────────────────────────────────────────────────
    ax    = axes[0]
    x     = np.arange(len(metric_cols))
    bar_w = 0.25
    for i, v in enumerate(variants):
        vals = [results[v][mc] for mc in metric_cols]
        bars = ax.bar(x + i * bar_w, vals, width=bar_w, label=v, color=palette[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(metric_cols)
    ax.set(ylabel="Score", ylim=(0, 1.15), title="Metrics by variant")
    ax.legend()

    # ── fit time bar ──────────────────────────────────────────────────────────
    ax = axes[1]
    times = [results[v]["fit_time"] for v in variants]
    bars  = ax.bar(variants, times, color=palette, edgecolor="white")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, t + max(times) * 0.01,
                f"{t*1000:.2f} ms", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Fit time (s)", title="Fit time by variant")

    fig.suptitle(f"NB Exp 1 · Variant comparison  [{cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# NB Experiment 2 – smoothing parameter sweep
# ----------------------

def exp_nb_smoothing(nb_variant: str, cls_type: str) -> mplfig.Figure:
    """
    Sweep the smoothing parameter and observe AUC + log-loss.

    gaussian    → var_smoothing  (prevents zero-variance division)
    bernoulli /
    multinomial → alpha          (Laplace smoothing on count estimates)

    Higher smoothing = stronger prior toward uniform → higher bias, lower variance.
    """
    X, y = _make_dataset(cls_type, n_samples=2000)
    X_tr_raw, X_te_raw, y_tr, y_te = _split_scale(X, y)
    X_tr, X_te = _prepare_nb_data(X_tr_raw, X_te_raw, nb_variant)

    if nb_variant == "gaussian":
        param_name  = "var_smoothing"
        param_grid  = np.logspace(-12, 0, 30)
    else:
        param_name  = "alpha"
        param_grid  = np.logspace(-3, 2, 30)

    aucs, llosses = [], []
    for val in param_grid:
        m = _new_nb_model(nb_variant, **{param_name: val})
        r = _nb_eval(cls_type, m, X_tr, y_tr, X_te, y_te)
        aucs.append(r["roc_auc"])
        llosses.append(r["log_loss"])

    best_idx = int(np.argmax(aucs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogx(param_grid, aucs, marker=".", ms=5)
    ax.axvline(param_grid[best_idx], color="red", ls="--",
               label=f"Best {param_name}={param_grid[best_idx]:.2e}")
    ax.set(xlabel=param_name, ylabel="ROC-AUC",
           title=f"ROC-AUC vs. {param_name}")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.semilogx(param_grid, llosses, marker=".", ms=5, color="darkorange")
    ax.axvline(param_grid[best_idx], color="red", ls="--",
               label=f"Best AUC at {param_grid[best_idx]:.2e}")
    ax.set(xlabel=param_name, ylabel="Log-loss",
           title=f"Log-loss vs. {param_name}")
    ax.legend(fontsize=9)

    fig.suptitle(f"NB Exp 2 · Smoothing sweep  [{nb_variant} · {cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# NB Experiment 3 – independence assumption violation
# ----------------------

def exp_nb_independence(nb_variant: str, cls_type: str) -> mplfig.Figure:
    """
    Sweep n_redundant (correlated features) to show how correlation degrades NB.
    Compares NB against LogisticRegression, which doesn't assume independence.

    As n_redundant rises, features are more correlated → the independence
    assumption breaks down → NB confidence estimates become overconfident
    → AUC and F1 drop faster than for LR.
    """
    n_features   = 20
    n_cls        = N_CLASSES[cls_type]
    redundant_grid = list(range(0, n_features - n_cls, 2))

    nb_aucs, nb_f1s = [], []
    lr_aucs, lr_f1s = [], []

    for n_red in redundant_grid:
        n_inf = max(n_cls, n_features - n_red)
        X, y  = make_classification(
            n_samples=1500, n_features=n_features,
            n_informative=n_inf, n_redundant=n_red,
            n_clusters_per_class=1, n_classes=n_cls,
            random_state=SEED,
        )
        X_tr_raw, X_te_raw, y_tr, y_te = _split_scale(X, y)

        # NB
        X_tr, X_te = _prepare_nb_data(X_tr_raw, X_te_raw, nb_variant)
        r_nb = _nb_eval(cls_type, _new_nb_model(nb_variant), X_tr, y_tr, X_te, y_te)
        nb_aucs.append(r_nb["roc_auc"]); nb_f1s.append(r_nb["f1"])

        # Logistic Regression baseline
        r_lr = _nb_eval(cls_type,
                        LogisticRegression(lr=0.05, n_iter=500, type=cls_type),
                        X_tr_raw, y_tr, X_te_raw, y_te)
        lr_aucs.append(r_lr["roc_auc"]); lr_f1s.append(r_lr["f1"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(redundant_grid, nb_aucs, marker="o", label=f"NaiveBayes ({nb_variant})")
    ax.plot(redundant_grid, lr_aucs, marker="s", ls="--", label="LogisticRegression")
    ax.set(xlabel="n_redundant (correlated features)", ylabel="ROC-AUC",
           title="ROC-AUC vs. feature correlation")
    ax.legend()

    ax = axes[1]
    ax.plot(redundant_grid, nb_f1s, marker="o", label=f"NaiveBayes ({nb_variant})")
    ax.plot(redundant_grid, lr_f1s, marker="s", ls="--", label="LogisticRegression")
    ax.set(xlabel="n_redundant (correlated features)", ylabel="F1",
           title="F1 vs. feature correlation")
    ax.legend()

    fig.suptitle(f"NB Exp 3 · Independence assumption  [{nb_variant} · {cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# NB Experiment 4 – confusion matrix + calibration
# ----------------------

def exp_nb_confusion_calibration(nb_variant: str, cls_type: str) -> mplfig.Figure:
    """
    Confusion matrix + reliability diagram.

    NB is famously overconfident: the independence assumption causes each
    feature's evidence to be counted as if it were independent, so predicted
    probabilities cluster near 0 and 1 even when the true confidence is moderate.
    The reliability diagram makes this visible.
    """
    X, y = _make_dataset(cls_type, n_samples=3000)
    X_tr_raw, X_te_raw, y_tr, y_te = _split_scale(X, y)
    X_tr, X_te = _prepare_nb_data(X_tr_raw, X_te_raw, nb_variant)
    r = _nb_eval(cls_type, _new_nb_model(nb_variant), X_tr, y_tr, X_te, y_te)
    proba, pred = r["proba"], r["pred"]
    classes = np.unique(y_te)
    K = len(classes)

    n_cols = 3 if cls_type == "ordinal" else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    palette = sns.color_palette("muted", K)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    ax = axes[0]
    cm      = confusion_matrix(y_te, pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=[str(c) for c in classes],
                yticklabels=[str(c) for c in classes],
                ax=ax, linewidths=0.5)
    for (i, j), raw in np.ndenumerate(cm):
        ax.text(j + 0.5, i + 0.72, f"n={raw}", ha="center", va="center",
                fontsize=7, color="gray")
    ax.set(title=f"Confusion matrix  [{nb_variant}]",
           xlabel="Predicted", ylabel="True")

    # ── Reliability diagram ───────────────────────────────────────────────────
    ax     = axes[1]
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    if cls_type == "binary":
        bin_ids   = np.digitize(proba, bins[1:-1])
        mean_pred = [proba[bin_ids == b].mean()  if (bin_ids == b).any() else np.nan
                     for b in range(n_bins)]
        frac_pos  = [y_te[bin_ids == b].mean()   if (bin_ids == b).any() else np.nan
                     for b in range(n_bins)]
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=nb_variant)
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.15, color="steelblue", label="Miscalibration gap")
    else:
        Y_bin = np.asarray(label_binarize(y_te, classes=classes))
        for k in range(K):
            pk       = proba[:, k]
            bin_ids  = np.digitize(pk, bins[1:-1])
            mean_pred = [pk[bin_ids == b].mean()        if (bin_ids == b).any() else np.nan
                         for b in range(n_bins)]
            frac_pos  = [Y_bin[bin_ids == b, k].mean()  if (bin_ids == b).any() else np.nan
                         for b in range(n_bins)]
            ax.plot(mean_pred, frac_pos, marker="o", lw=1.5,
                    color=palette[k], label=f"class {classes[k]}")

    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title="Reliability diagram — NB tends to overconfidence",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)

    # ── Ordinal error distribution ────────────────────────────────────────────
    if cls_type == "ordinal":
        ax = axes[2]
        pred_idx = np.searchsorted(classes, pred)
        true_idx = np.searchsorted(classes, y_te)
        errors   = pred_idx - true_idx
        unique_e, counts = np.unique(errors, return_counts=True)
        ax.bar(unique_e, counts / len(errors), color="mediumpurple",
               edgecolor="white", width=0.6)
        ax.axvline(0, color="black", ls="--", alpha=0.5)
        ax.set(xlabel="Predicted − True class", ylabel="Proportion",
               title=f"Ordinal error  (MAE={np.mean(np.abs(errors)):.3f})")
        ax.set_xticks(unique_e)

    fig.suptitle(f"NB Exp 4 · Confusion & calibration  [{nb_variant} · {cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# NB Experiment 5 – scalability (fit time heatmap)
# ----------------------

def exp_nb_scalability(nb_variant: str, cls_type: str) -> mplfig.Figure:
    """
    NB fit is O(n * d) — one pass to accumulate sufficient statistics.
    This makes it extremely fast compared to gradient-based models.
    Predict is O(K * d) per sample — independent of training set size.
    """
    sample_grid  = [200, 500, 1000, 5000, 20000]
    feature_grid = [5, 10, 20, 50, 100]
    n_cls = N_CLASSES[cls_type]

    fit_time_mat = np.zeros((len(sample_grid), len(feature_grid)))
    auc_mat      = np.zeros_like(fit_time_mat)

    for i, n in enumerate(sample_grid):
        for j, d in enumerate(feature_grid):
            n_inf = max(n_cls, min(d, d // 2))
            X, y  = _make_dataset(cls_type, n_samples=n, n_features=d,
                                  n_informative=n_inf)
            X_tr_raw, X_te_raw, y_tr, y_te = _split_scale(X, y)
            X_tr, X_te = _prepare_nb_data(X_tr_raw, X_te_raw, nb_variant)
            r = _nb_eval(cls_type, _new_nb_model(nb_variant),
                         X_tr, y_tr, X_te, y_te)
            fit_time_mat[i, j] = r["fit_time"]
            auc_mat[i, j]      = r["roc_auc"]

    row_labels = [str(n) for n in sample_grid]
    col_labels = [str(d) for d in feature_grid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(fit_time_mat, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=col_labels, yticklabels=row_labels,
                ax=axes[0], linewidths=0.4)
    axes[0].set(xlabel="n_features", ylabel="n_samples",
                title="Fit time (s)  — O(n·d), single pass")

    sns.heatmap(auc_mat, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=col_labels, yticklabels=row_labels,
                ax=axes[1], linewidths=0.4, vmin=0.5, vmax=1.0)
    axes[1].set(xlabel="n_features", ylabel="n_samples", title="ROC-AUC")

    fig.suptitle(f"NB Exp 5 · Scalability  [{nb_variant} · {cls_type}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ----------------------
# Naive Bayes entry point
# ----------------------

def evaluate_NaiveBayes(nb_variant: str = "gaussian", cls_type: str = "binary") -> None:
    if nb_variant not in VALID_NB_VARIANTS:
        raise ValueError(f"nb_variant must be one of {VALID_NB_VARIANTS}. Got {nb_variant!r}.")
    if cls_type not in VALID_TYPES:
        raise ValueError(f"cls_type must be one of {VALID_TYPES}. Got {cls_type!r}.")

    if nb_variant == "all":
        for v in VALID_NB_VARIANTS[:-1]:
            evaluate_NaiveBayes(nb_variant=v, cls_type=cls_type)
        return
    if cls_type == "all":
        for t in VALID_TYPES[:-1]:
            evaluate_NaiveBayes(nb_variant=nb_variant, cls_type=t)
        return

    print(f"Evaluating NaiveBayes  [variant={nb_variant} · type={cls_type}]  "
          f"({N_CLASSES[cls_type]} classes)\n")

    experiments = [
        ("Variant comparison",       lambda: exp_nb_variant_comparison(cls_type)),
        ("Smoothing sweep",          lambda: exp_nb_smoothing(nb_variant, cls_type)),
        ("Independence assumption",  lambda: exp_nb_independence(nb_variant, cls_type)),
        ("Confusion & calibration",  lambda: exp_nb_confusion_calibration(nb_variant, cls_type)),
        ("Scalability",              lambda: exp_nb_scalability(nb_variant, cls_type)),
    ]

    for name, fn in experiments:
        print(f"  -> {name} …")
        fig = fn()
        # fname = (f"eval_nb_{nb_variant}_{cls_type}_"
        #          f"{name.lower().replace(' & ', '_').replace(' ', '_')}.png")
        # fig.savefig(fname, dpi=140, bbox_inches="tight")
        # print(f"     saved {fname}")

    print("\nDone.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate classification algorithms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo", default="logistic_regression",
        choices=["logistic_regression", "knn", "naive_bayes"],
        help="Algorithm to evaluate.",
    )
    parser.add_argument(
        "--type", dest="cls_type", default="binary",
        choices=list(VALID_TYPES),
        help="Classification mode (number of classes / label structure).",
    )
    parser.add_argument(
        "--nb_variant", default="gaussian",
        choices=list(VALID_NB_VARIANTS),
        help="Naive Bayes variant (only used when --algo=naive_bayes).",
    )
    args = parser.parse_args()

    if args.algo == "logistic_regression":
        evaluate_LogisticRegression(cls_type=args.cls_type)
    elif args.algo == "knn":
        evaluate_KNN(cls_type=args.cls_type)
    elif args.algo == "naive_bayes":
        evaluate_NaiveBayes(nb_variant=args.nb_variant, cls_type=args.cls_type)
    else:
        raise ValueError(f"Unsupported algorithm {args.algo!r}.")

if __name__ == "__main__":
    main()