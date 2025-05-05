from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Кастомный фильтр коррелированных признаков
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            mask = corr_matrix.where(upper)
            self.to_drop_ = [column for column in mask.columns if any(mask[column] > self.threshold)]
        return self

    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop_, errors='ignore')


def remove_and_fill_nans(data):
    return data.fillna(data.median())

def simple_preprocess_df(data):
    # data = feature_engineer(data)
    # data = remove_zeros_and_fill_remained_with_mean(data, threshold=30)
    return data

def preprocess_df(data):
    data = feature_engineering_df_preprocessor(data)
    # data = remove_zeros_and_fill_remained_with_mean(data, threshold=30)
    return data


def feature_engineering_df_preprocessor(df):
    df = df.copy()

    # 1. Логарифмические и корневые преобразования признаков
    for col in ['MolWt', 'LabuteASA', 'TPSA', 'NumRotatableBonds']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            df[f'sqrt_{col}'] = np.sqrt(df[col])

    # 2. Отношения и взаимодействия
    if all(c in df.columns for c in ['MolLogP', 'MolWt']):
        df['LogP_per_weight'] = df['MolLogP'] / (df['MolWt'] + 1)
    if all(c in df.columns for c in ['NumHDonors', 'NumHAcceptors']):
        df['H_don_acc_ratio'] = df['NumHDonors'] / (df['NumHAcceptors'] + 1)
    if all(c in df.columns for c in ['RingCount', 'HeavyAtomCount']):
        df['Ring_density'] = df['RingCount'] / (df['HeavyAtomCount'] + 1)

    # 3. Суммы фрагментов (группировка fr_*)
    nitro_cols = [c for c in df.columns if c in ['fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho']]
    if nitro_cols:
        df['fr_nitro_total'] = df[nitro_cols].sum(axis=1)

    amine_cols = [c for c in df.columns if c in ['fr_NH0', 'fr_NH1', 'fr_NH2']]
    if amine_cols:
        df['fr_amine_total'] = df[amine_cols].sum(axis=1)

    fr_cols = [c for c in df.columns if c.startswith('fr_')]
    if fr_cols:
        df['fr_total'] = df[fr_cols].sum(axis=1)

    # 4. PCA по PEOE_VSA* (если есть)
    vsa_cols = [c for c in df.columns if c.startswith('PEOE_VSA')]
    if len(vsa_cols) >= 3:
        pca = PCA(n_components=2)
        vsa_pca = pca.fit_transform(df[vsa_cols].fillna(0))
        df['PEOE_VSA_PC1'] = vsa_pca[:, 0]
        df['PEOE_VSA_PC2'] = vsa_pca[:, 1]

    return df


import numpy as np
import pandas as pd

def add_log_features(df, cols):
    for col in cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df

def add_interaction_features(df):
    if "MolWt" in df.columns and "TPSA" in df.columns:
        df["MolWt_x_TPSA"] = df["MolWt"] * df["TPSA"]
    if "MolLogP" in df.columns and "MolWt" in df.columns:
        df["LogP_per_weight"] = df["MolLogP"] / (df["MolWt"] + 1e-6)
    return df

def add_mass_normalized_features(df):
    if "MolWt" in df.columns:
        df["NumRings_per_weight"] = df["RingCount"] / (df["MolWt"] + 1e-6)
        df["NumHAcceptors_per_weight"] = df["NumHAcceptors"] / (df["MolWt"] + 1e-6)
    return df

def add_fragment_aggregates(df):
    df['fr_total'] = df[[c for c in df.columns if c.startswith('fr_')]].sum(axis=1)
    df['fr_nitro_total'] = df[[c for c in df.columns if c in ['fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho']]].sum(axis=1)
    df['fr_amine_total'] = df[[c for c in df.columns if c in ['fr_NH0', 'fr_NH1', 'fr_NH2']]].sum(axis=1)
    df['fr_aromatic_total'] = df[[c for c in df.columns if 'arom' in c]].sum(axis=1)
    df['fr_halogen_total'] = df[[c for c in df.columns if 'halogen' in c]].sum(axis=1)
    return df

def add_block_stats(df, prefix):
    block_cols = [col for col in df.columns if col.startswith(prefix)]
    if block_cols:
        df[f"{prefix}_sum"] = df[block_cols].sum(axis=1)
        df[f"{prefix}_std"] = df[block_cols].std(axis=1)
    return df

def binarize_fragments(df):
    for col in [c for c in df.columns if c.startswith("fr_")]:
        df[f"{col}_bin"] = (df[col] > 0).astype(int)
    return df





def preprocess_ic50_regression(df):
    df = df.copy()
    df['IC50'] = np.log1p(df['IC50'])

    df = add_log_features(df, ['MolWt', 'TPSA', 'LabuteASA', 'MolLogP'])
    df = add_interaction_features(df)
    df = add_mass_normalized_features(df)
    df = add_fragment_aggregates(df)
    df = add_block_stats(df, "PEOE_VSA")
    return df

def preprocess_cc50_regression(df):
    df = df.copy()
    df['CC50'] = np.log1p(df['CC50'])

    df = add_log_features(df, ['MolWt', 'TPSA', 'LabuteASA', 'MolLogP'])
    df = add_interaction_features(df)
    df = add_mass_normalized_features(df)
    df = add_fragment_aggregates(df)
    df = add_block_stats(df, "SlogP_VSA")
    return df

def preprocess_si_regression(df):
    df = df.copy()

    # Пересчёт SI вручную и логарифмирование
    df['SI'] = df['CC50'] / (df['IC50'] + 1e-8)
    df['SI'] = np.log1p(df['SI'])  # используем ТОЛЬКО этот признак

    # Логарифмирование ключевых признаков
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP', 'NumRotatableBonds'])

    # Интерактивные признаки (например, произведения и соотношения)
    df = add_interaction_features(df)

    # Агрегаты по фрагментам
    df = add_fragment_aggregates(df)

    # Статистика по блокам признаков
    df = add_block_stats(df, "EState_VSA")

    # Нормализация по массе (например, для интерпретируемости и масштабирования)
    df = add_mass_normalized_features(df)


    return df


def preprocess_ic50_classification(df):
    df = df.copy()
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])
    df = add_interaction_features(df)
    df = add_fragment_aggregates(df)
    df = binarize_fragments(df)
    return df

def preprocess_cc50_classification(df):
    df = df.copy()
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])
    df = add_interaction_features(df)
    df = add_fragment_aggregates(df)
    df = binarize_fragments(df)
    return df

def preprocess_si_median_classification(df):
    df = df.copy()
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])
    df = add_interaction_features(df)
    df = add_fragment_aggregates(df)
    df = add_block_stats(df, "PEOE_VSA")
    return df

def preprocess_si_gt8_classification(df):
    df = df.copy()
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])
    df = add_interaction_features(df)
    df = add_fragment_aggregates(df)
    df = add_block_stats(df, "SlogP_VSA")
    return df






































#
#
#
# def preprocess_ic50_regression(df):
#     df = df.copy()
#     df['IC50'] = np.log1p(df['IC50'])
#     for col in ['MolWt', 'TPSA', 'LabuteASA', 'MolLogP']:
#         if col in df:
#             df[f'log_{col}'] = np.log1p(df[col])
#     return df
#
# def preprocess_cc50_regression(df):
#     df = df.copy()
#     df['CC50'] = np.log1p(df['CC50'])
#     for col in ['MolWt', 'TPSA', 'LabuteASA', 'MolLogP']:
#         if col in df:
#             df[f'log_{col}'] = np.log1p(df[col])
#     # Группы фрагментов
#     df['fr_nitro_total'] = df[[c for c in df.columns if c in ['fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho']]].sum(axis=1)
#     df['fr_amine_total'] = df[[c for c in df.columns if c in ['fr_NH0', 'fr_NH1', 'fr_NH2']]].sum(axis=1)
#     return df
#
# def preprocess_si_regression(df):
#     df = df.copy()
#     df['SI'] = np.log1p(df['SI'])
#     df['SI_ratio'] = df['CC50'] / (df['IC50'] + 1e-8)
#     df['log_SI_ratio'] = np.log1p(df['SI_ratio'])
#     for col in ['MolWt', 'TPSA', 'MolLogP', 'NumRotatableBonds']:
#         if col in df:
#             df[f'log_{col}'] = np.log1p(df[col])
#     df['fr_total'] = df[[c for c in df.columns if c.startswith('fr_')]].sum(axis=1)
#     return df
#
# def preprocess_si_regression_2(df):
#     df = df.copy()
#     df['SI'] = df['CC50'] / (df['IC50'] + 1e-8)
#     df['SI'] = np.log1p(df['SI'])
#     for col in ['MolWt', 'TPSA', 'MolLogP', 'NumRotatableBonds']:
#         if col in df:
#             df[f'log_{col}'] = np.log1p(df[col])
#     df['fr_total'] = df[[c for c in df.columns if c.startswith('fr_')]].sum(axis=1)
#     return df
#
# def preprocess_ic50_classification(df):
#     df = df.copy()
#     # df['IC50_bin'] = (df['IC50'] > df['IC50'].median()).astype(int)
#     for col in ['MolWt', 'TPSA', 'MolLogP']:
#         df[f'log_{col}'] = np.log1p(df[col])
#     return df
#
# def preprocess_cc50_classification(df):
#     df = df.copy()
#     # df['CC50_bin'] = (df['CC50'] > df['CC50'].median()).astype(int)
#     for col in ['MolWt', 'TPSA', 'MolLogP']:
#         df[f'log_{col}'] = np.log1p(df[col])
#     df['fr_nitro_total'] = df[[c for c in df.columns if c in ['fr_nitro', 'fr_nitro_arom']]].sum(axis=1)
#     return df
#
# def preprocess_si_median_classification(df):
#     df = df.copy()
#     # df['SI_bin_median'] = (df['SI'] > df['SI'].median()).astype(int)
#     for col in ['MolWt', 'TPSA', 'MolLogP']:
#         df[f'log_{col}'] = np.log1p(df[col])
#     df['fr_total'] = df[[c for c in df.columns if c.startswith('fr_')]].sum(axis=1)
#     return df
#
# def preprocess_si_gt8_classification(df):
#     df = df.copy()
#     # df['SI_bin_8'] = (df['SI'] > 8).astype(int)
#     for col in ['MolWt', 'TPSA', 'MolLogP']:
#         df[f'log_{col}'] = np.log1p(df[col])
#     df['fr_total'] = df[[c for c in df.columns if c.startswith('fr_')]].sum(axis=1)
#     return df