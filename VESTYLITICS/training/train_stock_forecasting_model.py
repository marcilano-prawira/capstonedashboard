"""
train_stock_forecasting_model.py
--------------------------------------------------
Train stock forecasting model using
TensorFlow Functional API.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(data_path: str):

    file_path = Path(data_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully\n")

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    X = df.drop(columns=["target"])

    y = df["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            shuffle=False
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# =========================================================
# BUILD MODEL
# =========================================================

def build_model(input_shape):

    # Functional API Input
    inputs = tf.keras.Input(
        shape=(input_shape,)
    )

    # Hidden Layer 1
    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(inputs)

    # Hidden Layer 2
    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    # Output Layer
    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    # Build Model
    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    # Compile Model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    model,
    X_train,
    y_train
):

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    return history


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = (
        model.predict(X_test)
        > 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.4f}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

def save_model(model):

    output_dir = Path("../models")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(
        output_dir /
        "stock_forecasting_model.keras"
    )

    print(
        "\nModel saved successfully."
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    data_path = (
        "../data/processed/"
        "processed_stock_dataset.csv"
    )

    # Load Dataset
    df = load_dataset(data_path)

    print(df.head())

    # Prepare Data
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = prepare_data(df)

    # Build Model
    model = build_model(
        input_shape=X_train.shape[1]
    )

    # Show Summary
    model.summary()

    # Train Model
    train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save Model
    save_model(model)


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":

    print(
        "\nTraining Stock Forecasting Model...\n"
    )

    main()