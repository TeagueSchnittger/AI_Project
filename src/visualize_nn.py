"""
Generates NN training visualizations and saves them to data/.
Run from project root:
    python -m src.visualize_nn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

from src.nn_scorer import generate_training_data

SEED = 42
OUT_DIR = "data"


def train_with_history(seed=SEED):
    """Retrain from scratch and record loss at every iteration."""
    X, y = generate_training_data(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=400,
        random_state=seed,
        warm_start=False,
    )

    # Collect loss curve by training one epoch at a time
    train_losses, val_losses = [], []
    model.max_iter = 1
    model.warm_start = True
    for _ in range(400):
        model.fit(X_train, y_train)
        train_losses.append(mean_squared_error(y_train, model.predict(X_train)))
        val_losses.append(mean_squared_error(y_test, model.predict(X_test)))

    y_pred = model.predict(X_test)
    return train_losses, val_losses, y_test, y_pred


def plot_loss_curve(train_losses, val_losses, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train MSE", linewidth=1.5)
    ax.plot(epochs, val_losses,   label="Validation MSE", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Neural Network Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_pred_vs_actual(y_test, y_pred, path):
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Scatter: predicted vs actual ---
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="steelblue")
    lims = [0, 1]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual Score (utility label)")
    ax.set_ylabel("Predicted Score (NN output)")
    ax.set_title(f"Predicted vs Actual\nR² = {r2:.4f},  MSE = {mse:.5f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Residuals histogram ---
    ax = axes[1]
    residuals = y_pred - y_test
    ax.hist(residuals, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (predicted − actual)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Retraining to collect loss history...")
    train_losses, val_losses, y_test, y_pred = train_with_history()

    plot_loss_curve(train_losses, val_losses,
                    os.path.join(OUT_DIR, "nn_loss_curve.png"))
    plot_pred_vs_actual(y_test, y_pred,
                        os.path.join(OUT_DIR, "nn_pred_vs_actual.png"))
    print("\nDone. Add to report with:")
    print("  \\includegraphics[width=0.85\\textwidth]{nn_loss_curve.png}")
    print("  \\includegraphics[width=0.85\\textwidth]{nn_pred_vs_actual.png}")
