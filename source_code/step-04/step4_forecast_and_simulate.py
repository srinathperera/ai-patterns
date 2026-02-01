"""Inference pipeline: load saved models and score unverified data."""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# -------------------------- Config --------------------------
CONFIG = {
    "target_column": "verified_pattern",
    "predict_mode": True,
    "labeled_data_path": "/datasets/labeled_data.csv",
    "embeddings_path": "/datasets/embeddings.csv",
    "min_samples_per_class": 20,
    "n_splits": 5,
    "random_state": 42,
    "none_label": "none",
    "model_dir": "artifacts/models",
    "output_dir": "artifacts/outputs",
}

VOTE_WEIGHTS = {"LogReg": 1.06, "SVC": 1.01, "KNN": 0.93}


# -------------------------- Utilities --------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_labels(y):
    label_encoder = LabelEncoder()
    y_values = y.values if isinstance(y, pd.Series) else y
    encoded = label_encoder.fit_transform(y_values)
    return label_encoder, encoded


def get_model_classes(fitted_model):
    if hasattr(fitted_model, "classes_"):
        return fitted_model.classes_
    if hasattr(fitted_model, "steps") and fitted_model.steps:
        return fitted_model.steps[-1][1].classes_
    raise AttributeError("Model does not expose classes_.")


def align_probabilities(prob_matrix, model_classes, global_classes):
    aligned = np.zeros((prob_matrix.shape[0], len(global_classes)))
    for class_id, class_label in enumerate(model_classes):
        global_index = np.where(global_classes == class_label)[0][0]
        aligned[:, global_index] = prob_matrix[:, class_id]
    return aligned


def resolve_weights(custom_weights=None):
    if custom_weights is None:
        return VOTE_WEIGHTS
    return {**VOTE_WEIGHTS, **custom_weights}


def load_and_merge_data(config):
    labeled = pd.read_csv(config["labeled_data_path"])
    embeddings = pd.read_csv(config["embeddings_path"])
    merged = pd.merge(labeled, embeddings, on="file")
    return merged


def split_verified_sets(data, target_column, min_samples):
    verified = data[~data[target_column].isna()]
    counts = verified[target_column].value_counts()
    keep_labels = counts[counts >= min_samples].index
    verified_filtered = verified[verified[target_column].isin(keep_labels)]
    unverified = data[data[target_column].isna()]
    return verified_filtered, unverified


def weighted_vote_predict(prob_results, label_encoder, weights):
    weight_map = resolve_weights(weights)
    predictions = []
    scores = []

    total_rows = len(next(iter(prob_results.values()))) if prob_results else 0
    for row_index in range(total_rows):
        score_vector = None
        for name, probs_df in prob_results.items():
            weighted = weight_map.get(name, 0) * probs_df.iloc[row_index].astype(float).values
            score_vector = weighted if score_vector is None else score_vector + weighted
        best_index = int(np.argmax(score_vector))
        predictions.append(label_encoder.inverse_transform([best_index])[0])
        scores.append(float(score_vector[best_index]))

    return predictions, scores


def load_artifacts(model_dir: Path):
    models = {}
    for name in ("LogReg", "SVC", "KNN"):
        model_path = model_dir / f"{name.lower()}_model.joblib"
        if model_path.exists():
            models[name] = joblib.load(model_path)
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    global_classes = np.array(load_json(model_dir / "classes.json"))
    feature_list = load_json(model_dir / "feature_list.json")
    return models, label_encoder, global_classes, feature_list


def run_prediction_pipeline(config=CONFIG):
    model_dir = Path(config["model_dir"])
    output_dir = ensure_dir(Path(config["output_dir"]))
    models, label_encoder, global_classes, feature_list = load_artifacts(model_dir)

    data = load_and_merge_data(config)
    _, unverified_df = split_verified_sets(
        data,
        target_column=config["target_column"],
        min_samples=config["min_samples_per_class"],
    )
    if unverified_df.empty:
        print("No unverified data to predict.")
        return

    X_predict = unverified_df[feature_list]
    prob_map = {}
    for name, model in models.items():
        probs = model.predict_proba(X_predict.values)
        model_classes = get_model_classes(model)
        aligned = align_probabilities(probs, model_classes, global_classes)
        prob_map[name] = pd.DataFrame(aligned, columns=global_classes)

    predictions, scores = weighted_vote_predict(prob_map, label_encoder, weights=None)

    pd.DataFrame(
        {
            "file": unverified_df["file"],
            "weighted_pred": predictions,
            "weighted_score": scores,
            "code_summary": unverified_df.get("code_summary"),
        }
    ).to_csv(output_dir / "unverified_weighted_predictions.csv", index=False)
    print(f"Saved predictions to {output_dir / 'unverified_weighted_predictions.csv'}")


if __name__ == "__main__":
    run_prediction_pipeline()
