"""Train-only estimators.

Input: feature frames from features.build_features and a train label vector.
Output: positive-class probabilities. Logistic regression stays sparse.
Depends on: config.py
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import config


def _columns(bundle):
    return config.numeric_columns(bundle), config.categorical_columns(bundle)


def make_logit(bundle, C):
    numeric, categorical = _columns(bundle)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )
    classifier = LogisticRegression(
        C=C,
        solver="liblinear",
        max_iter=1000,
    )
    return make_pipeline(preprocessor, classifier)


def make_hgb(bundle):
    """Used only when run_experiments.py is started with --hgb."""
    numeric, categorical = _columns(bundle)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                categorical,
            ),
        ]
    )
    categorical_idx = list(range(len(numeric), len(numeric) + len(categorical)))
    classifier = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=config.SEED,
        categorical_features=categorical_idx,
    )
    return make_pipeline(preprocessor, classifier)


def positive_proba(model, features):
    return model.predict_proba(features)[:, 1]
