"""
baseline_financial_awareness.py
--------------------------------------------------
Baseline Financial Awareness System

This script:
1. Loads financial profile data
2. Performs feature engineering
3. Generates financial readiness status
4. Generates financial insight/reasons
5. Saves processed results

Output Labels:
- Ready
- Fair
- Not Ready
"""

from pathlib import Path
from typing import Tuple

import pandas as pd


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load financial profile dataset.

    Parameters
    ----------
    data_path : str
        Path to CSV dataset

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """

    file_path = Path(data_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully.\n")

    return df


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_financial_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create financial ratio features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """

    # Prevent division by zero
    df["monthly_income"] = (
        df["monthly_income"]
        .replace(0, 1)
    )

    df["monthly_expense"] = (
        df["monthly_expense"]
        .replace(0, 1)
    )

    # Expense ratio
    df["expense_ratio"] = (
        df["monthly_expense"]
        / df["monthly_income"]
    )

    # Saving ratio
    df["saving_ratio"] = (
        df["savings"]
        / df["monthly_income"]
    )

    # Emergency fund ratio
    df["emergency_ratio"] = (
        df["emergency_fund"]
        / df["monthly_expense"]
    )

    return df


# =========================================================
# FINANCIAL READINESS ANALYSIS
# =========================================================

def analyze_financial_readiness(
    row: pd.Series
) -> pd.Series:
    """
    Analyze user financial readiness.

    Parameters
    ----------
    row : pd.Series
        Single dataframe row

    Returns
    -------
    pd.Series
        Financial status and reasons
    """

    reasons = []

    score = 0

    # =====================================================
    # EMERGENCY FUND CHECK
    # =====================================================

    if row["emergency_ratio"] < 3:

        reasons.append(
            "Emergency fund is below 3x monthly expenses"
        )

    else:

        score += 1

        reasons.append(
            "Emergency fund level is healthy"
        )

    # =====================================================
    # EXPENSE RATIO CHECK
    # =====================================================

    if row["expense_ratio"] > 0.8:

        reasons.append(
            "Monthly expenses are too high compared to income"
        )

    else:

        score += 1

        reasons.append(
            "Expenses are well managed"
        )

    # =====================================================
    # SAVING RATIO CHECK
    # =====================================================

    if row["saving_ratio"] < 0.1:

        reasons.append(
            "Savings ratio is still low"
        )

    else:

        score += 1

        reasons.append(
            "Savings ratio is healthy"
        )

    # =====================================================
    # FINAL STATUS
    # =====================================================

    if score == 3:
        status = "Ready"

    elif score == 2:
        status = "Fair"

    else:
        status = "Not Ready"

    return pd.Series([
        status,
        " | ".join(reasons)
    ])


# =========================================================
# APPLY ANALYSIS
# =========================================================

def apply_financial_analysis(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply readiness analysis to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Processed dataframe
    """

    df[["status", "reasons"]] = df.apply(
        analyze_financial_readiness,
        axis=1
    )

    return df


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_output(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Save processed dataframe to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe

    output_path : str
        Output CSV path
    """

    output_file = Path(output_path)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_file, index=False)

    print(
        f"\nResults saved to: {output_path}"
    )


# =========================================================
# MAIN EXECUTION
# =========================================================

def main() -> None:
    """
    Main execution pipeline.
    """

    # Paths
    data_path = (
        "../data/financial_data/"
        "financial_profile.csv"
    )

    output_path = (
        "../data/processed/"
        "financial_awareness_result.csv"
    )

    try:

        # Load dataset
        df = load_dataset(data_path)

        print(df.head())

        # Feature engineering
        df = create_financial_features(df)

        # Financial analysis
        df = apply_financial_analysis(df)

        # Preview result
        print(
            "\nFinancial Readiness Results\n"
        )

        print(
            df[
                [
                    "monthly_income",
                    "monthly_expense",
                    "savings",
                    "emergency_fund",
                    "status",
                    "reasons"
                ]
            ].head()
        )

        # Save output
        save_output(df, output_path)

    except Exception as error:

        print(
            f"\n[ERROR] {error}"
        )


# =========================================================
# RUN SCRIPT
# =========================================================

if __name__ == "__main__":

    print(
        "\nRunning Financial Awareness Analysis...\n"
    )

    main()