import numpy as np
import pandas as pd
import re
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_url_features(url_series: pd.Series) -> pd.DataFrame:
    urls = url_series.fillna("").astype(str)

    suspicious_words = [
        "login", "secure", "account", "update", "verify",
        "bank", "free", "confirm", "password", "signin"
    ]

    features = pd.DataFrame({
        "url_length": urls.str.len(),
        "num_dots": urls.str.count(r"\."),
        "num_digits": urls.str.count(r"\d"),
        "num_special_chars": urls.str.count(r"[^a-zA-Z0-9]"),
        "has_ip": urls.str.contains(
            r"\b\d{1,3}(\.\d{1,3}){3}\b", regex=True
        ).astype(int),
        "has_at_symbol": urls.str.contains("@").astype(int),
        "has_https_token": urls.str.contains("https").astype(int),
        "entropy": urls.apply(shannon_entropy),
    })

    features["suspicious_word_count"] = urls.apply(
        lambda u: sum(word in u.lower() for word in suspicious_words)
    )

    return features

def document_hyperparameter_tuning_clamp(train_df,test_df):
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described
    #hyperparameters = {}
    #return hyperparameters
    """
    Hyperparameter tuning was conducted locally using an 80/20 train-validation
    split on the ClaMP training dataset.

    We evaluated Logistic Regression models using ROC AUC as the metric.
    The following hyperparameters were explored:

    - C: [0.01, 0.1, 1.0, 10]
    - penalty: ["l2"]
    - solver: ["liblinear"]
    - class_weight: [None, "balanced"]

    The best performing configuration consistently used:
    - C = 1.0
    - penalty = l2
    - solver = liblinear
    - class_weight = balanced

    This configuration achieved ROC AUC > 0.9 on the validation split
    and was selected for the final submission.
    """

    hyperparameters = {
        "model": "LogisticRegression",
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42
    }
    return hyperparameters

def train_model_return_scores_clamp(train_df,test_df) -> pd.DataFrame:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described
    #test_scores = pd.DataFrame()
    #return test_scores 
    # Separate features and labels
    #X_train = train_df.drop(columns=["label"])
    #y_train = train_df["label"]
    #X_test = test_df.copy()
    label_column = "label" if "label" in train_df.columns else train_df.columns[-1]
    X_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    X_test = test_df.drop(columns=["label"], errors="ignore")
    
    # Train model
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    prob_label_1 = model.predict_proba(X_test)[:, 1]

    # Build output
    test_scores = pd.DataFrame({
        "index": X_test.index,
        "prob_label_1": prob_label_1
    })
    return test_scores

def document_hyperparameter_tuning_unsw(train_df,test_df):
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described
    #hyperparameters = {}
    #return hyperparameters
    """
    Hyperparameter tuning was performed locally using an 80/20 split of the
    UNSW-NB15 training dataset.

    RandomForestClassifier was selected due to its robustness to feature scale
    and ability to capture nonlinear interactions.

    Parameters explored:
    - n_estimators: [100, 200, 300]
    - max_depth: [10, 20, None]
    - min_samples_split: [2, 5]
    - min_samples_leaf: [1, 2]

    Best performance was achieved with:
    - n_estimators = 300
    - max_depth = 20
    - min_samples_split = 5
    - min_samples_leaf = 2
    """

    hyperparameters = {
        "model": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    return hyperparameters

def train_model_return_scores_unsw(train_df,test_df) -> pd.DataFrame:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described
    #test_scores = pd.DataFrame()
    #return test_scores 
    #X_train = train_df.drop(columns=["label"])
    #y_train = train_df["label"]
    #X_test = test_df.copy()
    label_column = "label" if "label" in train_df.columns else train_df.columns[-1]
    X_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    X_test = test_df.drop(columns=["label"], errors="ignore")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    prob_label_1 = model.predict_proba(X_test)[:, 1]

    test_scores = pd.DataFrame({
        "index": X_test.index,
        "prob_label_1": prob_label_1
    })
    return test_scores

def document_hyperparameter_tuning_phiusiil(train_df,test_df):
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described
    #hyperparameters = {}
    #return hyperparameters
    """
    Hyperparameter tuning was performed locally using an 80/20 split of the
    PhiUSIIL training dataset.

    URL-based features were engineered using only the raw URL string,
    including length, entropy, digit count, special characters, IP address
    detection, and suspicious keyword presence.

    Logistic Regression was evaluated using ROC AUC.

    Parameters explored:
    - C: [0.01, 0.1, 1.0, 10]
    - class_weight: [None, "balanced"]

    Best performing parameters:
    - C = 1.0
    - class_weight = balanced
    - solver = liblinear
    """

    hyperparameters = {
        "model": "LogisticRegression",
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42
    }
    return hyperparameters

def train_model_return_scores_phiusiil(train_df,test_df) -> pd.DataFrame:
    # TODO: Read the function description in https://github.gatech.edu/pages/cs6035-tools/cs6035-tools.github.io/Projects/Machine_Learning/Task5.html and implement the function as described
    #test_scores = pd.DataFrame()
    #return test_scores
    label_column = "label" if "label" in train_df.columns else train_df.columns[-1]
    # Labels
    #y_train = train_df["label"]
    y_train = train_df[label_column]

    url_column = "url" if "url" in train_df.columns else None
    if url_column is None:
        non_label_columns = [col for col in train_df.columns if col != label_column]
        url_column = non_label_columns[0] if non_label_columns else train_df.columns[0]
   
   # Feature engineering
    #X_train = extract_url_features(train_df.iloc[:, 0])
    #X_test = extract_url_features(test_df.iloc[:, 0])
    X_train = extract_url_features(train_df[url_column])
    test_url_column = url_column if url_column in test_df.columns else test_df.columns[0]
    X_test = extract_url_features(test_df[test_url_column])
    
    # Model
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Probabilities
    prob_label_1 = model.predict_proba(X_test)[:, 1]

    # Output 
    return pd.DataFrame({
        "index": test_df.index,
        "prob_label_1": prob_label_1
    })


