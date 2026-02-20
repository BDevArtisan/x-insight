from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


ID_COLS = {"pat_ipp"}

SHAPIRO_MAX_SAMPLE = 5000
NORMALITY_PVALUE = 0.05


def _test_normality(series: pd.Series) -> bool:
    values = series.dropna().values
    if len(values) < 20:
        return False
    sample = values if len(values) <= SHAPIRO_MAX_SAMPLE else np.random.choice(values, SHAPIRO_MAX_SAMPLE, replace=False)
    _, p = stats.shapiro(sample)
    return p > NORMALITY_PVALUE


class BasePreprocessor:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

        self.binary_cols: List[str] = []
        self.ohe_cols: List[str] = []
        self.continuous_cols: List[str] = []

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.ohe: Optional[OneHotEncoder] = None
        self.ohe_feature_names: List[str] = []
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}

        self.feature_names_out: List[str] = []
        self.fitted: bool = False

    def _classify_columns(self, df: pd.DataFrame) -> None:
        self.binary_cols = []
        self.ohe_cols = []
        self.continuous_cols = []

        for col in df.columns:
            if col in ID_COLS:
                continue
            dtype = df[col].dtype
            nunique = df[col].nunique(dropna=True)

            if dtype == bool or (dtype == object and nunique == 2) or (dtype in [np.int64, np.float64] and nunique == 2):
                self.binary_cols.append(col)
            elif dtype == object and nunique > 2:
                self.ohe_cols.append(col)
            elif dtype == object and nunique <= 2:
                self.binary_cols.append(col)
            else:
                self.continuous_cols.append(col)

    def fit_transform(self, df: pd.DataFrame) -> tuple:
        self._classify_columns(df)

        work = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore").copy()

        work = work[self.binary_cols + self.ohe_cols + self.continuous_cols]

        for col in self.binary_cols:
            work[col] = work[col].astype(str).replace("nan", np.nan)
            le = LabelEncoder()
            non_null = work[col].dropna()
            le.fit(non_null)
            work[col] = work[col].map(lambda v, le=le: le.transform([v])[0] if v not in (np.nan, None, "nan", "None") and not (isinstance(v, float) and np.isnan(v)) else np.nan)
            work[col] = pd.to_numeric(work[col], errors="coerce")
            self.label_encoders[col] = le

        if self.ohe_cols:
            for col in self.ohe_cols:
                work[col] = work[col].astype(str).replace({"nan": np.nan, "None": np.nan})
            self.ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            ohe_array = self.ohe.fit_transform(work[self.ohe_cols].fillna("__missing__"))
            self.ohe_feature_names = list(self.ohe.get_feature_names_out(self.ohe_cols))
            ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_feature_names, index=work.index)
            work = work.drop(columns=self.ohe_cols).join(ohe_df)

        imputed_array = self.imputer.fit_transform(work)
        work = pd.DataFrame(imputed_array, columns=work.columns, index=work.index)

        for col in self.continuous_cols:
            if _test_normality(work[col]):
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            work[col] = scaler.fit_transform(work[[col]])
            self.scalers[col] = scaler

        self.feature_names_out = list(work.columns)
        self.fitted = True
        return work.values, work, self.feature_names_out

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform.")

        work = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore").copy()
        work = work[[c for c in self.binary_cols + self.ohe_cols + self.continuous_cols if c in work.columns]]

        for col in self.binary_cols:
            if col not in work.columns:
                continue
            le = self.label_encoders[col]
            work[col] = work[col].astype(str).replace("nan", np.nan)
            known = set(le.classes_)
            work[col] = work[col].map(lambda v, le=le, known=known: le.transform([v])[0] if v in known else np.nan)
            work[col] = pd.to_numeric(work[col], errors="coerce")

        if self.ohe_cols and self.ohe is not None:
            filled = work[self.ohe_cols].fillna("__missing__").astype(str)
            ohe_array = self.ohe.transform(filled)
            ohe_df = pd.DataFrame(ohe_array, columns=self.ohe_feature_names, index=work.index)
            work = work.drop(columns=self.ohe_cols).join(ohe_df)

        imputed_array = self.imputer.transform(work)
        work = pd.DataFrame(imputed_array, columns=work.columns, index=work.index)

        for col, scaler in self.scalers.items():
            if col in work.columns:
                work[col] = scaler.transform(work[[col]])

        work = work.reindex(columns=self.feature_names_out, fill_value=0.0)
        return work.values

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform.")

        work = pd.DataFrame(X, columns=self.feature_names_out)

        for col, scaler in self.scalers.items():
            if col in work.columns:
                work[col] = scaler.inverse_transform(work[[col]])

        return work

    def save(self, filepath: str) -> None:
        joblib.dump(self.__dict__, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BasePreprocessor":
        data = joblib.load(filepath)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        return obj


class NotSelectedPreprocessor(BasePreprocessor):
    """Preprocessor for FHU-patient_descriptive_not_selected.xlsx."""
    pass


class SelectedPreprocessor(BasePreprocessor):
    """Preprocessor for FHU-patients_selected.xlsx."""
    pass
