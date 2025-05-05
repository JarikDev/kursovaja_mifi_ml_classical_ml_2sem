import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest,f_regression ,mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.ml.utils.processors import RegressionProcessor, regression_models
from src.ml.utils.processors import regression_tasks, classification_tasks, classification_models, \
    ClassificationProcessor

# Подавляем основные предупреждения
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  # если нужно
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # sklearn warnings

data_path = '../ml/data/kursovik_data.csv'

def run_regression():
    for task in regression_tasks:
        # Запускаем обработку всех моделей для решения задачи регрессии
        for model in tqdm(regression_models):
            # Строим пайплайн
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков
                ('low_variance', VarianceThreshold(threshold=0.01)),
                ('scaler', RobustScaler()),  # Масштабирование
                # ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('feature_selection', SelectKBest(score_func=f_regression , k=20)),
                # ('feature_selection', SelectKBest(score_func=mutual_info_regression , k=20)),
                # ('corr_filter', CorrelationFilter(threshold=0.95)),  # Удаление коррелированных фич
                ('pca', PCA()),  # PCA уменьшаем размерность
                ('regressor', model.model)
            ])
            regressor_name = type(pipeline.named_steps['regressor']).__name__
            print(f'\nЗапускаем регрессию для: {task.y_col}, модель: {regressor_name}\n')
            # запускаем минифреймворк
            RegressionProcessor(data_path,
                                task.y_col,
                                # simple_preprocess_df,
                                task.preprocessor,
                                # feature_engineering_df_preprocessor,
                                model.param_grid,
                                pipeline).run()

def run_classification():
    for task in classification_tasks:
        # Запускаем обработку всех моделей для решения задачи классификации
        for model in tqdm(classification_models):
            # Строим пайплайн
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков
                ('low_variance', VarianceThreshold(threshold=0.01)),  # Удаление малополезных фич
                ('scaler', RobustScaler()),  # Масштабирование
                # ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
                # ('corr_filter', CorrelationFilter(threshold=0.95)),  # Удаление коррелированных фич
                ('pca', PCA()),  # PCA до 95% дисперсии
                ('clf', model.model)  # Модель
            ])
            regressor_name = type(pipeline.named_steps['clf']).__name__
            print(f'\nЗапускаем классификацию для: {task.y_col}, модель: {regressor_name}\n')
            # запускаем минифреймворк
            ClassificationProcessor(data_path,
                                    task.y_col,
                                    task.preprocessor,
                                    model.param_grid,
                                    pipeline,
                                    task.split_criterion_func).run()
def run_app():
    print('Что делаем ? 1 - регрессию, 2 - кластеризацию, 3 - всё.')
    mode = input()
    match mode:
        case '1':
            run_regression()
        case '2':
            run_classification()
        case '3':
            run_regression()
            run_classification()
        case _:
            raise KeyError(f"Не знаю что это ... {mode}")

run_app()
