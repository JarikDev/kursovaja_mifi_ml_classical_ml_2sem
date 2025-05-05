import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest,mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.ml.utils.feature_engineering import feature_engineering_df_preprocessor, preprocess_ic50_regression, \
    simple_preprocess_df
from src.ml.utils.processors import RegressionProcessor, regression_models

# Подавляем основные предупреждения
warnings.filterwarnings('ignore')
# warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  # если нужно
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # sklearn warnings

y_col = 'IC50'

# Запускаем обработку всех моделей для решения задачи регрессии
for model in tqdm(regression_models):
    # Строим пайплайн
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков
        ('low_variance', VarianceThreshold(threshold=0.01)),
        ('scaler', RobustScaler()),  # Масштабирование
        # ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('feature_selection', SelectKBest(score_func=f_regression, k=20)),
        # ('feature_selection', SelectKBest(score_func=mutual_info_regression , k=20)),
        # ('corr_filter', CorrelationFilter(threshold=0.95)),  # Удаление коррелированных фич
        ('pca', PCA()),  # PCA уменьшаем размерность
        ('regressor', model.model)
    ])
    regressor_name = type(pipeline.named_steps['regressor']).__name__
    print(f'Запускаем регрессию для: {y_col}, модель: {regressor_name}')
    # запускаем минифреймворк
    RegressionProcessor("../data/kursovik_data.csv",
                        y_col,
                        preprocess_ic50_regression,
                        model.param_grid,
                        pipeline).run()
