import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "Crop_recommendation.csv"
MODEL_PATH = "crop_recommendation_model.joblib"


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at '{path}'")

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("Expected dataset to contain a 'label' column")

    return df


def prepare_data(df: pd.DataFrame):
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y


def build_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
        ]
    )


def train_model(X: pd.DataFrame, y: pd.Series):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model training completed")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    return model, label_encoder


def save_model(model, label_encoder, feature_names, path: str):
    payload = {"pipeline": model, "label_encoder": label_encoder, "feature_names": list(feature_names)}
    joblib.dump(payload, path)
    print(f"Saved trained model to '{path}'")


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'")
    return joblib.load(path)


def predict_crop(sample: dict, model_payload: dict) -> str:
    model = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]

    feature_names = model_payload.get("feature_names")
    sample_df = pd.DataFrame([sample])
    if feature_names is not None:
        missing = [c for c in feature_names if c not in sample_df.columns]
        if missing:
            raise ValueError(f"Missing feature(s) for prediction: {missing}")
        # Reorder columns to match training
        sample_df = sample_df[feature_names]

    prediction = model.predict(sample_df)
    return label_encoder.inverse_transform(prediction)[0]


def main():
    df = load_dataset(DATA_PATH)
    X, y = prepare_data(df)
    model, label_encoder = train_model(X, y)
    save_model(model, label_encoder, X.columns, MODEL_PATH)

    example_input = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.8,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.0,
    }

    saved_payload = load_model(MODEL_PATH)
    predicted_crop = predict_crop(example_input, saved_payload)
    print(f"\nExample prediction for sample:\n{example_input}\n=> Recommended crop: {predicted_crop}")


if __name__ == "__main__":
    main()
