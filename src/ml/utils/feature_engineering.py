from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Кастомный фильтр для удаления сильно коррелированных признаков
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold  # Порог корреляции
        self.to_drop_ = []  # Список признаков для удаления

    def fit(self, X, y=None):
        # Вычисляем корреляционную матрицу и определяем признаки, превышающие порог
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
            upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            mask = corr_matrix.where(upper)
            self.to_drop_ = [column for column in mask.columns if any(mask[column] > self.threshold)]
        return self

    def transform(self, X):
        # Удаляем выбранные признаки
        return pd.DataFrame(X).drop(columns=self.to_drop_, errors='ignore')

# Заполняем пропущенные значения медианой
def remove_and_fill_nans(data):
    return data.fillna(data.median())

# Пустой препроцессинг (для совместимости)
def simple_preprocess_df(data):
    return data

# Основной препроцессинг датафрейма с фильтрацией выбросов и преобразованиями
def preprocess_df(df):
    df = feature_engineering_df_preprocessor(df)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    df = df.dropna()
    df.rename(columns={'CC50, mM': 'CC50'}, inplace=True)
    df.rename(columns={'IC50, mM': 'IC50'}, inplace=True)
    df = df[df['IC50'] < 1001]  # Фильтрация по IC50
    df['SI'] = df['CC50'] / df['IC50']  # Расчёт SI
    df = df[df['SI'] < 1001]  # Фильтрация по SI
    return df

# Более строгая фильтрация выбросов
def strict_delete_outliners_preprocess_df(data):
    data = feature_engineering_df_preprocessor(data)
    if "Unnamed: 0" in data.columns:
        data.drop(columns=["Unnamed: 0"], inplace=True)
    data = data.dropna()
    data.rename(columns={'CC50, mM': 'CC50'}, inplace=True)
    data.rename(columns={'IC50, mM': 'IC50'}, inplace=True)
    data = data[data['IC50'] < 400]
    data = data[data['CC50'] < 1500]
    data['SI'] = data['CC50'] / data['IC50']
    data = data[data['SI'] < 60]
    return data

# Расширенный препроцессинг: извлечение новых признаков
def feature_engineering_df_preprocessor(df):
    df = df.copy()

    # Логарифмы и корни признаков
    for col in ['MolWt', 'LabuteASA', 'TPSA', 'NumRotatableBonds']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            df[f'sqrt_{col}'] = np.sqrt(df[col])

    # Отношения между признаками
    if all(c in df.columns for c in ['MolLogP', 'MolWt']):
        df['LogP_per_weight'] = df['MolLogP'] / (df['MolWt'] + 1)
    if all(c in df.columns for c in ['NumHDonors', 'NumHAcceptors']):
        df['H_don_acc_ratio'] = df['NumHDonors'] / (df['NumHAcceptors'] + 1)
    if all(c in df.columns for c in ['RingCount', 'HeavyAtomCount']):
        df['Ring_density'] = df['RingCount'] / (df['HeavyAtomCount'] + 1)

    # Суммарные признаки фрагментов
    nitro_cols = [c for c in df.columns if c in ['fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho']]
    if nitro_cols:
        df['fr_nitro_total'] = df[nitro_cols].sum(axis=1)

    amine_cols = [c for c in df.columns if c in ['fr_NH0', 'fr_NH1', 'fr_NH2']]
    if amine_cols:
        df['fr_amine_total'] = df[amine_cols].sum(axis=1)

    fr_cols = [c for c in df.columns if c.startswith('fr_')]
    if fr_cols:
        df['fr_total'] = df[fr_cols].sum(axis=1)

    # PCA по блокам PEOE_VSA (если хватает признаков)
    vsa_cols = [c for c in df.columns if c.startswith('PEOE_VSA')]
    if len(vsa_cols) >= 3:
        pca = PCA(n_components=2)
        vsa_pca = pca.fit_transform(df[vsa_cols].fillna(0))
        df['PEOE_VSA_PC1'] = vsa_pca[:, 0]
        df['PEOE_VSA_PC2'] = vsa_pca[:, 1]

    return df

# Логарифмирование заданных признаков
def add_log_features(df, cols):
    for col in cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df

# Взаимодействие признаков
def add_interaction_features(df):
    if "MolWt" in df.columns and "TPSA" in df.columns:
        df["MolWt_x_TPSA"] = df["MolWt"] * df["TPSA"]
    if "MolLogP" in df.columns and "MolWt" in df.columns:
        df["LogP_per_weight"] = df["MolLogP"] / (df["MolWt"] + 1e-6)
    return df

# Нормализация по массе
def add_mass_normalized_features(df):
    if "MolWt" in df.columns:
        df["NumRings_per_weight"] = df["RingCount"] / (df["MolWt"] + 1e-6)
        df["NumHAcceptors_per_weight"] = df["NumHAcceptors"] / (df["MolWt"] + 1e-6)
    return df

# Агрегаты по фрагментам
def add_fragment_aggregates(df):
    df['fr_total'] = df[[c for c in df.columns if c.startswith('fr_')]].sum(axis=1)
    df['fr_nitro_total'] = df[[c for c in df.columns if c in ['fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho']]].sum(axis=1)
    df['fr_amine_total'] = df[[c for c in df.columns if c in ['fr_NH0', 'fr_NH1', 'fr_NH2']]].sum(axis=1)
    df['fr_aromatic_total'] = df[[c for c in df.columns if 'arom' in c]].sum(axis=1)
    df['fr_halogen_total'] = df[[c for c in df.columns if 'halogen' in c]].sum(axis=1)
    return df

# Сводная статистика по группам признаков
def add_block_stats(df, prefix):
    block_cols = [col for col in df.columns if col.startswith(prefix)]
    if block_cols:
        df[f"{prefix}_sum"] = df[block_cols].sum(axis=1)
        df[f"{prefix}_std"] = df[block_cols].std(axis=1)
    return df

# Бинаризация фрагментов (наличие/отсутствие)
def binarize_fragments(df):
    for col in [c for c in df.columns if c.startswith("fr_")]:
        df[f"{col}_bin"] = (df[col] > 0).astype(int)
    return df

# Ниже идут функции препроцессинга для разных задач — регрессия и классификация IC50, CC50, SI
def preprocess_ic50_regression(df):
    """
    Препроцессинг данных для задачи регрессии по IC50:
    1. Логарифмирование значения IC50 (чтобы нормализовать распределение).
    2. Добавление логарифмических и корневых преобразований для выбранных признаков.
    3. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    4. Нормализация признаков с учётом массы.
    5. Агрегирование фрагментов молекул по признакам.
    6. Добавление статистики для блоков признаков, связанных с PEOE_VSA.

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df['IC50'] = np.log1p(df['IC50'])  # Логарифмируем IC50 (с добавлением 1, чтобы избежать логарифмирования 0)

    # Добавляем логарифмические и корневые признаки для заданных столбцов
    df = add_log_features(df, ['MolWt', 'TPSA', 'LabuteASA', 'MolLogP'])

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем признаки, нормализованные по массе
    df = add_mass_normalized_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Добавляем статистику для блока признаков "PEOE_VSA"
    df = add_block_stats(df, "PEOE_VSA")

    return df  # Возвращаем обработанный DataFrame


def preprocess_cc50_regression(df):
    """
    Препроцессинг данных для задачи регрессии по CC50:
    1. Логарифмирование значения CC50 (чтобы нормализовать распределение).
    2. Добавление логарифмических и корневых преобразований для выбранных признаков.
    3. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    4. Нормализация признаков с учётом массы.
    5. Агрегирование фрагментов молекул по признакам.
    6. Добавление статистики для блоков признаков, связанных с SlogP_VSA.

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df['CC50'] = np.log1p(df['CC50'])  # Логарифмируем CC50 (с добавлением 1, чтобы избежать логарифмирования 0)

    # Добавляем логарифмические и корневые признаки для заданных столбцов
    df = add_log_features(df, ['MolWt', 'TPSA', 'LabuteASA', 'MolLogP'])

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем признаки, нормализованные по массе
    df = add_mass_normalized_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Добавляем статистику для блока признаков "SlogP_VSA"
    df = add_block_stats(df, "SlogP_VSA")

    return df  # Возвращаем обработанный DataFrame


def preprocess_si_regression(df):
    """
    Препроцессинг данных для задачи регрессии по SI:
    1. Вычисление нового признака SI как отношение CC50 к IC50.
    2. Логарифмирование значения SI (чтобы нормализовать распределение).
    3. Добавление логарифмических и корневых преобразований для выбранных признаков.
    4. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    5. Агрегирование фрагментов молекул по признакам.
    6. Добавление статистики для блоков признаков, связанных с EState_VSA.
    7. Нормализация признаков с учётом массы.

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df['SI'] = df['CC50'] / (df['IC50'] + 1e-8)  # Вычисляем SI (с небольшим сдвигом для предотвращения деления на ноль)
    df['SI'] = np.log1p(df['SI'])  # Логарифмируем SI

    # Добавляем логарифмические и корневые признаки для заданных столбцов
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP', 'NumRotatableBonds'])

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Добавляем статистику для блока признаков "EState_VSA"
    df = add_block_stats(df, "EState_VSA")

    # Добавляем признаки, нормализованные по массе
    df = add_mass_normalized_features(df)

    return df  # Возвращаем обработанный DataFrame


def preprocess_ic50_classification(df):
    """
    Препроцессинг данных для задачи классификации по IC50:
    1. Добавление логарифмических преобразований для выбранных признаков.
    2. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    3. Добавление агрегации по фрагментам молекул.
    4. Бинаризация фрагментов молекул (преобразование признаков в бинарные значения).

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])  # Добавляем логарифмические признаки

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Бинаризация фрагментов молекул (например, преобразуем количество в бинарные признаки)
    df = binarize_fragments(df)

    return df  # Возвращаем обработанный DataFrame


def preprocess_cc50_classification(df):
    """
    Препроцессинг данных для задачи классификации по CC50:
    1. Добавление логарифмических преобразований для выбранных признаков.
    2. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    3. Добавление агрегации по фрагментам молекул.
    4. Бинаризация фрагментов молекул (преобразование признаков в бинарные значения).

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])  # Добавляем логарифмические признаки

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Бинаризация фрагментов молекул (например, преобразуем количество в бинарные признаки)
    df = binarize_fragments(df)

    return df  # Возвращаем обработанный DataFrame


def preprocess_si_median_classification(df):
    """
    Препроцессинг данных для задачи классификации по SI, где целевой переменной является медиана SI:
    1. Добавление логарифмических преобразований для выбранных признаков.
    2. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    3. Добавление агрегации по фрагментам молекул.
    4. Добавление статистики для блоков признаков, связанных с PEOE_VSA.

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])  # Добавляем логарифмические признаки

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Добавляем статистику для блока признаков "PEOE_VSA"
    df = add_block_stats(df, "PEOE_VSA")

    return df  # Возвращаем обработанный DataFrame


def preprocess_si_gt8_classification(df):
    """
    Препроцессинг данных для задачи классификации по SI, где целевая переменная больше 8:
    1. Добавление логарифмических преобразований для выбранных признаков.
    2. Добавление интерактивных признаков (например, произведения молекулярных характеристик).
    3. Добавление агрегации по фрагментам молекул.
    4. Добавление статистики для блоков признаков, связанных с SlogP_VSA.

    :param df: DataFrame с данными
    :return: DataFrame с добавленными и преобразованными признаками
    """
    df = df.copy()  # Создаём копию данных, чтобы не изменять оригинал
    df = add_log_features(df, ['MolWt', 'TPSA', 'MolLogP'])  # Добавляем логарифмические признаки

    # Добавляем интерактивные признаки (например, произведения молекулярных свойств)
    df = add_interaction_features(df)

    # Добавляем агрегации по фрагментам
    df = add_fragment_aggregates(df)

    # Добавляем статистику для блока признаков "SlogP_VSA"
    df = add_block_stats(df, "SlogP_VSA")

    return df  # Возвращаем обработанный DataFrame
