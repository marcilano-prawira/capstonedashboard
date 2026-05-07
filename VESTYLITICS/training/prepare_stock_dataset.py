"""
prepare_stock_dataset.py
--------------------------------------------------
Prepare stock dataset for AI forecasting model.

This script:
1. Loads feature-engineered stock dataset
2. Selects important features
3. Creates prediction target
4. Handles missing values
5. Scales features
6. Saves processed dataset
"""

from pathlib import Path

import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(data_path: str) -> pd.DataFrame:

    file_path = Path(data_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully\n")

    return df


# =========================================================
# CREATE TARGET
# =========================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stock movement target.

    Target:
    1 = next day price goes up
    0 = next day price goes down
    """

    df["target"] = (
        df["Close"].shift(-1)
        > df["Close"]
    ).astype(int)

    return df


# =========================================================
# SELECT FEATURES
# =========================================================

def select_features(df: pd.DataFrame):

    selected_features = [
        "Close",
        "Volume"
    ]

    # Optional engineered features
    optional_features = [
        "RSI",
        "MACD",
        "Volatility",
        "MA_20",
        "EMA_20"
    ]

    for feature in optional_features:

        if feature in df.columns:

            selected_features.append(feature)

    print(
        "\nSelected Features:\n",
        selected_features
    )

    return df[selected_features + ["target"]]


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.dropna()

    return df


# =========================================================
# SCALE FEATURES
# =========================================================

def scale_features(df: pd.DataFrame):

    scaler = MinMaxScaler()

    feature_columns = [
        col for col in df.columns
        if col != "target"
    ]

    df[feature_columns] = scaler.fit_transform(
        df[feature_columns]
    )

    return df


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: str
):

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_file, index=False)

    print(
        f"\nProcessed dataset saved to:\n"
        f"{output_path}"
    )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    data_path = (
    r"D:\SEMESTER 6\DBS\VESTYLITICS"
    r"\VESTYLITICS\data\stock_data"
    r"\lq45_feature_engineering.csv"
    )
    

    output_path = (
        "/data/processed/"
        "processed_stock_dataset.csv"
    )

    # Load dataset
    df = load_dataset(data_path)

    print(df.head())

    print("\nDataset Shape:", df.shape)

    # Create target
    df = create_target(df)

    # Select features
    df = select_features(df)

    # Remove missing values
    df = clean_dataset(df)

    # Scale features
    df = scale_features(df)

    # Save processed dataset
    save_dataset(df, output_path)

    print("\nPipeline Completed Successfully\n")


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":

    main()