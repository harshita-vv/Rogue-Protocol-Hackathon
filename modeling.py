"""Train-only estimators.

Input: feature frames from features.build_features and a train label vector.
Output: positive-class probabilities. Logistic regression stays sparse.
LightGBM is optional and is constructed only when the import succeeds.
Depends on: config.py
"""

import pandas as pd
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


def lightgbm_available():
    """Return (True, version) or (False, reason). Never installs anything."""
    try:
        import lightgbm
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, getattr(lightgbm, "__version__", "unknown")


def _category_map(frame, categorical):
    mapping = {}
    for column in categorical:
        values = frame[column].astype("string").fillna("MISSING")
        mapping[column] = pd.Index(values.unique())
    return mapping


def _apply_categories(frame, mapping):
    encoded = frame.copy()
    for column, categories in mapping.items():
        values = encoded[column].astype("string").fillna("MISSING")
        encoded[column] = pd.Categorical(values, categories=categories)
    return encoded


class _LightGBMModel:
    """Categories are taken from the train rows passed to fit_lightgbm."""

    def __init__(self, model, categories):
        self.model = model
        self.categories = categories

    def predict_proba(self, features):
        return self.model.predict_proba(_apply_categories(features, self.categories))


def fit_lightgbm(x_train, y_train, bundle):
    available, detail = lightgbm_available()
    if not available:
        raise RuntimeError(detail)
    import lightgbm as lgb

    categorical = config.categorical_columns(bundle)
    categories = _category_map(x_train, categorical)
    model = lgb.LGBMClassifier(**config.LGB_PARAMS)
    model.fit(
        _apply_categories(x_train, categories),
        y_train,
        categorical_feature=categorical,
    )
    return _LightGBMModel(model, categories)
