import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data.preprocessing import BasePreprocessor, NotSelectedPreprocessor, SelectedPreprocessor


@pytest.fixture
def simple_df():
    np.random.seed(42)
    n = 80
    return pd.DataFrame({
        'pat_ipp': range(n),
        'age': np.random.uniform(20, 80, n),
        'sex': np.random.choice(['M', 'F'], n),
        'score': np.random.normal(50, 10, n),
        'flag': np.random.choice([0.0, 1.0], n),
        'category': np.random.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture
def simple_df_with_nulls(simple_df):
    df = simple_df.copy()
    df.loc[[0, 5, 10], 'age'] = np.nan
    df.loc[[2, 7], 'score'] = np.nan
    df.loc[[3], 'sex'] = np.nan
    return df


class TestBasePreprocessor:
    def test_id_col_dropped(self, simple_df):
        p = BasePreprocessor()
        _, out, names = p.fit_transform(simple_df)
        assert 'pat_ipp' not in names

    def test_output_shape(self, simple_df):
        p = BasePreprocessor()
        X, out, names = p.fit_transform(simple_df)
        assert X.shape[0] == len(simple_df)
        assert X.shape[1] == len(names)
        assert out.shape == X.shape

    def test_no_nulls_after_imputation(self, simple_df_with_nulls):
        p = BasePreprocessor()
        X, out, names = p.fit_transform(simple_df_with_nulls)
        assert not np.isnan(X).any()
        assert not out.isnull().any().any()

    def test_binary_encoding(self, simple_df):
        p = BasePreprocessor()
        _, out, names = p.fit_transform(simple_df)
        assert 'sex' in names
        assert set(out['sex'].dropna().round().unique()).issubset({0.0, 1.0})
        assert 'flag' in names
        assert set(out['flag'].dropna().round().unique()).issubset({0.0, 1.0})

    def test_ohe_columns(self, simple_df):
        p = BasePreprocessor()
        _, out, names = p.fit_transform(simple_df)
        ohe_cols = [c for c in names if c.startswith('category_')]
        assert len(ohe_cols) > 0

    def test_continuous_column_scaled(self, simple_df):
        p = BasePreprocessor()
        X, out, names = p.fit_transform(simple_df)
        assert 'age' in names
        assert 'score' in names

    def test_column_classification(self, simple_df):
        p = BasePreprocessor()
        p.fit_transform(simple_df)
        assert 'sex' in p.binary_cols or 'flag' in p.binary_cols
        assert 'category' in p.ohe_cols
        assert 'age' in p.continuous_cols or 'score' in p.continuous_cols

    def test_transform_after_fit(self, simple_df):
        p = BasePreprocessor()
        X_fit, _, _ = p.fit_transform(simple_df)
        X_tr = p.transform(simple_df)
        np.testing.assert_array_almost_equal(X_fit, X_tr, decimal=5)

    def test_transform_before_fit_raises(self, simple_df):
        p = BasePreprocessor()
        with pytest.raises(ValueError):
            p.transform(simple_df)

    def test_inverse_transform(self, simple_df):
        p = BasePreprocessor()
        X, _, names = p.fit_transform(simple_df)
        df_inv = p.inverse_transform(X)
        assert list(df_inv.columns) == names
        assert df_inv.shape[0] == len(simple_df)

    def test_save_load(self, simple_df, tmp_path):
        p1 = BasePreprocessor()
        X1, _, _ = p1.fit_transform(simple_df)
        path = str(tmp_path / 'pp.pkl')
        p1.save(path)
        p2 = BasePreprocessor.load(path)
        X2 = p2.transform(simple_df)
        np.testing.assert_array_almost_equal(X1, X2, decimal=5)

    def test_fitted_flag(self, simple_df):
        p = BasePreprocessor()
        assert not p.fitted
        p.fit_transform(simple_df)
        assert p.fitted


class TestSubclasses:
    def test_not_selected_preprocessor(self, simple_df):
        p = NotSelectedPreprocessor()
        X, out, names = p.fit_transform(simple_df)
        assert X.shape[0] == len(simple_df)

    def test_selected_preprocessor(self, simple_df):
        p = SelectedPreprocessor()
        X, out, names = p.fit_transform(simple_df)
        assert X.shape[0] == len(simple_df)

    def test_custom_n_neighbors(self, simple_df):
        p = NotSelectedPreprocessor(n_neighbors=3)
        assert p.n_neighbors == 3
        X, _, _ = p.fit_transform(simple_df)
        assert not np.isnan(X).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
