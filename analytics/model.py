from pathlib import Path
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "loan_approval_data.csv"
)


def load_model_data():
    df = pd.read_csv(DATA_PATH)

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns

    numerical_cols = df.select_dtypes(
        include=["number"]
    ).columns

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    df[numerical_cols] = num_imputer.fit_transform(
        df[numerical_cols]
    )

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    df[categorical_cols] = cat_imputer.fit_transform(
        df[categorical_cols]
    )

    education_encoder = LabelEncoder()
    loan_encoder = LabelEncoder()

    df["Education_Level"] = (
        education_encoder.fit_transform(
            df["Education_Level"]
        )
    )

    df["Loan_Approved"] = (
        loan_encoder.fit_transform(
            df["Loan_Approved"]
        )
    )

    ohe_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category",
    ]

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    )

    encoded = ohe.fit_transform(
        df[ohe_cols]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(
            ohe_cols
        ),
        index=df.index,
    )

    df = pd.concat(
        [
            df.drop(columns=ohe_cols),
            encoded_df,
        ],
        axis=1,
    )

    df = df.drop(
        "Applicant_ID",
        axis=1,
    )

    df["DTI_Ratio_sq"] = (
        df["DTI_Ratio"] ** 2
    )

    df["Credit_Score_sq"] = (
        df["Credit_Score"] ** 2
    )

    X = df.drop(
        columns=[
            "Loan_Approved",
            "Credit_Score",
            "DTI_Ratio",
        ]
    )

    y = df["Loan_Approved"]

    return X, y


def evaluate_random_forest():
    X, y = load_model_data()

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    y_pred = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    return {
        "model": model,
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "confusion_matrix": matrix.tolist(),
        "feature_count": X.shape[1],
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
    }