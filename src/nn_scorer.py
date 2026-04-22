import numpy as np
import pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

MODEL_PATH = "data/scorer_model.pkl"


def _utility_score(dist_norm, temp_norm, pop_norm, w_dist, w_temp, w_pop):
    """
    Computes the rule-based multiplicative utility score used to generate training labels.

    Parameters:
        dist_norm (float): Normalized city distance [0, 1].
        temp_norm (float): Normalized city temperature [0, 1].
        pop_norm  (float): Normalized city population [0, 1].
        w_dist    (float): User's distance preference [0, 1].
        w_temp    (float): User's temperature preference [0, 1].
        w_pop     (float): User's population preference [0, 1].

    Returns:
        float: Raw utility score normalized to approximately [0, 1].
    """
    d_m = 1 - abs(dist_norm - w_dist)
    t_m = 1 - abs(temp_norm - w_temp)
    p_m = 1 - abs(pop_norm - w_pop)
    raw = (d_m + 0.01) * (t_m + 0.01) * (p_m + 0.01)
    return raw / (1.01 ** 3)


def generate_training_data(n=12000, noise=0.015, seed=42):
    """
    Generates labelled (features, score) pairs that span the full [0, 1] output range.

    Pure random inputs cluster scores around ~0.3; instead samples are drawn from
    three buckets so the NN sees the full distribution:
      - 'good match' : city attributes close to user preferences  → high score
      - 'bad match'  : city attributes far from user preferences  → low score
      - 'random'     : unconstrained                             → middle range

    Parameters:
        n     (int):   Total number of samples to generate. Default 12000.
        noise (float): Gaussian noise std added to labels to prevent overfitting. Default 0.015.
        seed  (int):   Random seed for reproducibility. Default 42.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X — feature matrix of shape (n, 6), dtype float32.
            y — label vector of shape (n,), min-max normalized to [0, 1], dtype float32.
    """
    rng = np.random.default_rng(seed)
    rows, targets = [], []
    per_bucket = n // 3

    # Bucket 1 — good match: sample a preference vector, put city near it
    prefs = rng.random((per_bucket, 3))
    noise_v = rng.normal(0, 0.1, (per_bucket, 3))
    city_feats = np.clip(prefs + noise_v, 0, 1)
    for i in range(per_bucket):
        f = np.concatenate([city_feats[i], prefs[i]])
        rows.append(f)
        targets.append(_utility_score(*f))

    # Bucket 2 — bad match: city attributes far from preferences
    prefs = rng.random((per_bucket, 3))
    far = 1.0 - prefs + rng.normal(0, 0.08, (per_bucket, 3))
    city_feats = np.clip(far, 0, 1)
    for i in range(per_bucket):
        f = np.concatenate([city_feats[i], prefs[i]])
        rows.append(f)
        targets.append(_utility_score(*f))

    # Bucket 3 — random
    rand_feats = rng.random((n - 2 * per_bucket, 6))
    for f in rand_feats:
        rows.append(f)
        targets.append(_utility_score(*f))

    X = np.array(rows, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)

    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min + 1e-9)
    y += rng.normal(0, noise, len(y)).astype(np.float32)
    y = np.clip(y, 0, 1)
    return X, y


def train(model_path=MODEL_PATH, seed=42):
    """
    Trains an MLPRegressor on synthetic preference data and saves the model to disk.

    Uses an 80/20 train/test split and prints final MSE and R² on the test set.

    Parameters:
        model_path (str): Path to save the trained model. Default 'data/scorer_model.pkl'.
        seed       (int): Random seed for reproducibility. Default 42.

    Returns:
        MLPRegressor: The trained model.
    """
    X, y = generate_training_data(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=400,
        random_state=seed,
        verbose=False,
    )

    print(f"Training on {len(X_train)} samples, validating on {len(X_test)}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Final test MSE : {mean_squared_error(y_test, y_pred):.5f}")
    print(f"Final test R²  : {r2_score(y_test, y_pred):.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {model_path}")
    return model


def load_model(model_path=MODEL_PATH):
    """
    Loads a trained MLPRegressor from disk.

    Parameters:
        model_path (str): Path to the saved model. Default 'data/scorer_model.pkl'.

    Returns:
        MLPRegressor: Loaded model ready for inference.
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(model, dist_norm, temp_norm, pop_norm, w_dist, w_temp, w_pop):
    """
    Runs a single inference to predict a city's match score.

    Parameters:
        model     (MLPRegressor): Trained model.
        dist_norm (float):        Normalized city distance [0, 1].
        temp_norm (float):        Normalized city temperature [0, 1].
        pop_norm  (float):        Normalized city population [0, 1].
        w_dist    (float):        User's distance preference [0, 1].
        w_temp    (float):        User's temperature preference [0, 1].
        w_pop     (float):        User's population preference [0, 1].

    Returns:
        float: Predicted match score in [0, 1] — higher means better match.
    """
    x = np.array([[dist_norm, temp_norm, pop_norm, w_dist, w_temp, w_pop]], dtype=np.float32)
    return float(np.clip(model.predict(x)[0], 0, 1))
