"""
Live visualisations of polynomial-regression training and behaviour.

Two modes:

    1. live_gd_train     — animate gradient-descent training of a polynomial
                           regressor on a 1D non-linear target. Four live panels:
                              (a) data + fitted curve (updates every iteration)
                              (b) loss curve (log scale, growing in real time)
                              (c) coefficient bar chart
                              (d) live metrics readout (iter, loss, train R², ||β||₂)

    2. live_degree_sweep — animate the polynomial degree growing from 1 → N
                           on the same data, using a closed-form solver so the
                           fit at each frame is the optimal least-squares fit.
                           Reveals the bias-variance transition visually.

Both modes use matplotlib's interactive mode and update individual artist
handles, so rendering stays fast on a laptop. No extra dependencies.

Run:
    python live_graph.py                 # default: live GD training
    python live_graph.py degree-sweep    # the degree-sweep animation
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


# ---------------------------------------------------------------- data + features
def make_data(n=150, noise=0.35, x_range=(-2.0, 2.0), seed=42):
    """Smooth non-linear ground truth: y = sin(1.5x) + 0.5x + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_range[0], x_range[1], n)
    y_clean = np.sin(1.5 * x) + 0.5 * x
    y = y_clean + rng.normal(0.0, noise, n)
    return x, y


def polynomial_features_1d(x, degree, include_bias=True):
    """Build [1, x, x^2, ..., x^d] for a 1D vector x."""
    cols = [np.ones_like(x)] if include_bias else []
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


# ============================================================================
# Mode 1 — live GD training
# ============================================================================
def live_gd_train(degree=10, n_samples=150, noise=0.35,
                  penalty=0.01, n_iter=3000,
                  render_every=5, pause=0.001, seed=42):
    """Animate polynomial-regression training with gradient descent.

    Args
    ----
    degree         polynomial degree
    penalty        ridge λ (set 0.0 for plain OLS-GD)
    n_iter         max GD iterations
    render_every   redraw the figure every K iterations (1 = every step,
                   higher = faster training, less frequent redraw)
    pause          plt.pause() seconds between frames (controls perceived speed)
    """
    x, y = make_data(n_samples, noise, seed=seed)
    Phi = polynomial_features_1d(x, degree)
    n, m = Phi.shape

    # Bias mask: column 0 is the intercept and is NOT regularized.
    bias_mask = np.ones(m); bias_mask[0] = 0.0

    # Safe lr from the Lipschitz bound L = 2*(||Φ||_F²/n + λ).
    L = 2.0 * (np.linalg.norm(Phi, "fro") ** 2 / n + penalty)
    lr = 1.0 / L

    coef = np.zeros(m)

    # Dense grid for the fitted curve plot
    x_grid = np.linspace(-2.5, 2.5, 400)
    Phi_grid = polynomial_features_1d(x_grid, degree)
    y_truth = np.sin(1.5 * x_grid) + 0.5 * x_grid

    # ---------------- figure setup ----------------
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Live Polynomial-Regression GD Training  "
        f"(degree={degree},  n={n_samples},  λ={penalty},  lr={lr:.2e})",
        fontsize=12, fontweight="bold",
    )

    # (a) data + fitted curve
    ax_fit = axes[0, 0]
    ax_fit.scatter(x, y, s=18, color="#444", alpha=0.55, label="train")
    ax_fit.plot(x_grid, y_truth, "k--", alpha=0.55, linewidth=1.3, label="ground truth")
    (fit_line,) = ax_fit.plot(x_grid, Phi_grid @ coef,
                              color="#E91E63", linewidth=2.2, label="current fit")
    ax_fit.set(xlabel="x", ylabel="y", title="Fitted Curve")
    ax_fit.legend(fontsize=9, loc="upper left")
    ax_fit.set_ylim(y.min() - 1.0, y.max() + 1.0)
    ax_fit.set_xlim(-2.7, 2.7)

    # (b) live loss
    ax_loss = axes[0, 1]
    (loss_line,) = ax_loss.plot([], [], color="#2196F3", linewidth=1.6)
    ax_loss.set(xlabel="iteration", ylabel="loss", title="Training Loss")
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(1, n_iter)

    # (c) coefficient bars
    ax_coef = axes[1, 0]
    bars = ax_coef.bar(range(m), coef, color="#4CAF50", edgecolor="white")
    ax_coef.axhline(0.0, color="gray", linewidth=0.7)
    ax_coef.set(xlabel="coefficient index", ylabel="value",
                title="Coefficients   (β₀ = intercept)")
    ax_coef.set_xticks(range(m))

    # (d) metrics readout
    ax_text = axes[1, 1]
    ax_text.axis("off")
    text_handle = ax_text.text(
        0.04, 0.5, "", transform=ax_text.transAxes,
        fontsize=13, family="monospace", verticalalignment="center",
    )
    ax_text.set_title("Live Metrics", loc="left", fontweight="bold")

    fig.tight_layout()

    losses = []
    tss = float(((y - y.mean()) ** 2).sum())

    # ---------------- training loop ----------------
    try:
        for it in range(1, n_iter + 1):
            residual = Phi @ coef - y
            grad = (2.0 / n) * (Phi.T @ residual) + 2.0 * penalty * (bias_mask * coef)
            coef -= lr * grad

            reg = penalty * float(np.sum((bias_mask * coef) ** 2))
            loss = float(residual @ residual) / n + reg

            if not np.isfinite(loss):
                print(f"\n[!] loss diverged at iter {it}; stopping.")
                break
            losses.append(loss)

            if it % render_every == 0 or it == 1:
                # update fitted curve
                fit_line.set_ydata(Phi_grid @ coef)

                # update loss curve
                loss_line.set_data(range(1, len(losses) + 1), losses)
                ax_loss.relim(); ax_loss.autoscale_view(scalex=False, scaley=True)

                # update coefficient bars
                for b, v in zip(bars, coef):
                    b.set_height(v)
                ax_coef.relim(); ax_coef.autoscale_view(scalex=False, scaley=True)

                # metrics
                rss = float(((Phi @ coef - y) ** 2).sum())
                r2 = 1.0 - rss / tss
                text_handle.set_text(
                    f"iteration : {it:>6d} / {n_iter}\n\n"
                    f"loss      : {loss:>12.4e}\n"
                    f"train RSS : {rss:>12.4e}\n"
                    f"train R²  : {r2:>12.4f}\n"
                    f"|β|₂      : {np.linalg.norm(coef):>12.4f}\n"
                    f"max |β|   : {np.abs(coef).max():>12.4f}"
                )

                fig.canvas.draw_idle()
                plt.pause(pause)
    except KeyboardInterrupt:
        print("\n[i] interrupted by user — leaving final frame on screen.")

    plt.ioff()
    plt.show()


# ============================================================================
# Mode 2 — animate degree sweep
# ============================================================================
def live_degree_sweep(max_degree=14, n_samples=150, noise=0.35,
                      penalty=0.0, pause=0.6, seed=42):
    """Animate the polynomial degree increasing 1 → max_degree.

    At each frame, the optimal least-squares (or ridge) fit at that degree is
    shown together with train/test R² in the title. This visualises the
    bias→variance transition without any iterative training.
    """
    from sklearn.model_selection import train_test_split

    x, y = make_data(n_samples, noise, seed=seed)
    X = x.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed,
    )
    x_tr, x_te = X_tr.ravel(), X_te.ravel()

    x_grid = np.linspace(-2.5, 2.5, 400)
    y_truth = np.sin(1.5 * x_grid) + 0.5 * x_grid

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(x_tr, y_tr, s=22, color="#444", alpha=0.55, label="train")
    ax.scatter(x_te, y_te, s=22, color="#2196F3", alpha=0.7,
               marker="x", label="test")
    ax.plot(x_grid, y_truth, "k--", alpha=0.55, linewidth=1.3, label="ground truth")
    (fit_line,) = ax.plot(x_grid, np.zeros_like(x_grid),
                          color="#E91E63", linewidth=2.2, label="fit")
    ax.set(xlabel="x", ylabel="y")
    ax.set_ylim(y.min() - 1.5, y.max() + 1.5)
    ax.set_xlim(-2.7, 2.7)
    ax.legend(loc="upper left", fontsize=10)

    try:
        for d in range(1, max_degree + 1):
            Phi_tr   = polynomial_features_1d(x_tr,   d)
            Phi_te   = polynomial_features_1d(x_te,   d)
            Phi_grid = polynomial_features_1d(x_grid, d)

            # Ridge-stabilised closed-form (Cholesky); equivalent to OLS at λ=0
            # but more robust to high-degree conditioning.
            m = Phi_tr.shape[1]
            I = np.eye(m); I[0, 0] = 0.0
            beta = la.solve(
                Phi_tr.T @ Phi_tr + penalty * I,
                Phi_tr.T @ y_tr,
                assume_a="pos",
            )

            r2_tr = 1.0 - ((Phi_tr @ beta - y_tr) ** 2).sum() \
                       / ((y_tr - y_tr.mean()) ** 2).sum()
            r2_te = 1.0 - ((Phi_te @ beta - y_te) ** 2).sum() \
                       / ((y_te - y_te.mean()) ** 2).sum()

            fit_line.set_ydata(Phi_grid @ beta)
            ax.set_title(
                f"degree = {d:>2d}    "
                f"train R² = {r2_tr:6.3f}    "
                f"test R² = {r2_te:6.3f}",
                fontsize=12, fontweight="bold",
            )
            fig.canvas.draw_idle()
            plt.pause(pause)
    except KeyboardInterrupt:
        print("\n[i] interrupted by user — leaving final frame on screen.")

    plt.ioff()
    plt.show()


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gd"

    if mode in ("gd", "train", "default"):
        live_gd_train(
            degree=10, n_samples=150, noise=0.35,
            penalty=0.01, n_iter=3000,
            render_every=5, pause=0.001,
        )
    elif mode in ("degree-sweep", "sweep", "degree"):
        live_degree_sweep(
            max_degree=14, n_samples=150, noise=0.35,
            penalty=0.0, pause=0.7,
        )
    else:
        print(f"unknown mode: {mode!r}\n"
              f"usage:  python live_graph.py [gd | degree-sweep]")
        sys.exit(2)
