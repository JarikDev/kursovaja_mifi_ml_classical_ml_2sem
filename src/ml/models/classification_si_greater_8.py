import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.ml.utils.feature_engineering import preprocess_cc50_classification, preprocess_si_gt8_classification
from src.ml.utils.processors import ClassificationProcessor, get_median_for_col, classification_models, get_8

# Подавляем основные предупреждения
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  # если нужно
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # sklearn warnings

y_col = 'SI'

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
    print(f'\nЗапускаем регрессию для: {y_col}, модель: {regressor_name}\n')
    # запускаем минифреймворк
    ClassificationProcessor("../data/kursovik_data.csv",
                            y_col,
                            preprocess_si_gt8_classification,
                            model.param_grid,
                            pipeline,
                            # get_8).run()
                            lambda x,y:8).run()
