import warnings

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.ml.utils.feature_engineering import preprocess_ic50_classification
from src.ml.utils.processors import ClassificationProcessor, classification_models

# Подавляем основные предупреждения
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)  # numpy warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # sklearn warnings

y_col = 'IC50'

# Запускаем обработку всех моделей для решения задачи классификации
for model in tqdm(classification_models):
    # Строим пайплайн
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков
        ('low_variance', VarianceThreshold(threshold=0.01)),  # Удаление малополезных фич
        ('scaler', RobustScaler()),  # Масштабирование
        ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
        ('pca', PCA()),  # PCA до 95% дисперсии
        ('clf', model.model)  # Модель
    ])
    regressor_name = type(pipeline.named_steps['clf']).__name__
    print(f'\nЗапускаем классификацию для: {y_col}, модель: {regressor_name}\n')
    # запускаем минифреймворк
    ClassificationProcessor(data_path="../data/kursovik_data.csv",
                            y_col=y_col,
                            data_preprocessing_fun=preprocess_ic50_classification,
                            param_grid=model.param_grid,
                            pipeline=pipeline,
                            split_criterion_func=lambda data, column: data[column].median()).run()
