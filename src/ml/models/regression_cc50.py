import warnings

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_regression, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.ml.utils.feature_engineering import preprocess_cc50_regression
from src.ml.utils.processors import RegressionProcessor, regression_models

# Подавляем основные предупреждения
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # sklearn warnings

y_col = 'CC50'

# Запускаем обработку всех моделей для решения задачи регрессии CC50.
for model in tqdm(regression_models):
    # Строим пайплайн
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков
        ('low_variance', VarianceThreshold(threshold=0.01)),
        ('scaler', RobustScaler()),  # Масштабирование
        ('feature_selection', SelectKBest(score_func=f_regression, k=20)),
        ('pca', PCA()),  # PCA уменьшаем размерность
        ('regressor', model.model)
    ])
    regressor_name = type(pipeline.named_steps['regressor']).__name__
    print(f'\nЗапускаем регрессию для: {y_col}, модель: {regressor_name}\n')
    # запускаем минифреймворк
    RegressionProcessor(data_path="../data/kursovik_data.csv",
                        y_col=y_col,
                        data_preprocessing_fun=preprocess_cc50_regression,
                        param_grid=model.param_grid,
                        pipeline=pipeline).run()
