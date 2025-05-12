import warnings

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from src.ml.utils.processors import RegressionProcessor, regression_models
from src.ml.utils.processors import regression_tasks, classification_tasks, classification_models, \
    ClassificationProcessor

# Подавляем все предупреждения, чтобы избежать лишних сообщений в консоли
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning)  # подавляем предупреждения от numpy
warnings.filterwarnings("ignore", category=ConvergenceWarning)  # подавляем предупреждения от sklearn

# Указываем путь к данным
data_path = '../ml/data/kursovik_data.csv'


def run_regression():
    """
    Функция для запуска всех моделей для задачи регрессии.
    Для каждой задачи регрессии и каждой модели создаётся пайплайн и запускается процесс.
    """
    for task in regression_tasks:
        # Проходим по всем моделям для каждой задачи регрессии
        for model in tqdm(regression_models):
            # Создание пайплайна для обработки данных
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков медианой
                ('low_variance', VarianceThreshold(threshold=0.01)),  # Убираем признаки с низкой дисперсией
                ('scaler', RobustScaler()),  # Масштабируем данные с использованием RobustScaler
                ('feature_selection', SelectKBest(score_func=f_regression, k=20)),  # Выбираем лучшие 20 признаков
                ('pca', PCA()),  # Уменьшаем размерность с помощью PCA
                ('regressor', model.model)  # Применяем выбранную модель регрессора
            ])
            regressor_name = type(pipeline.named_steps['regressor']).__name__  # Получаем имя модели
            print(f'\nЗапускаем регрессию для: {task.y_col}, модель: {regressor_name}\n')
            # Запуск процесса регрессии для текущей модели и задачи
            RegressionProcessor(data_path="../data/kursovik_data.csv",
                                y_col=task.y_col,
                                data_preprocessing_fun=task.preprocessor,
                                param_grid=model.param_grid,
                                pipeline=pipeline).run()


def run_classification():
    """
    Функция для запуска всех моделей для задачи классификации.
    Для каждой задачи классификации и каждой модели создаётся пайплайн и запускается процесс.
    """
    for task in classification_tasks:
        # Проходим по всем моделям для каждой задачи классификации
        for model in tqdm(classification_models):
            # Создание пайплайна для обработки данных
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  # Заполнение пропусков медианой
                ('low_variance', VarianceThreshold(threshold=0.01)),  # Убираем признаки с низкой дисперсией
                ('scaler', RobustScaler()),  # Масштабируем данные с использованием RobustScaler
                ('feature_selection', SelectKBest(score_func=f_classif, k=20)),  # Выбираем лучшие 20 признаков
                ('pca', PCA()),  # Уменьшаем размерность с помощью PCA
                ('clf', model.model)  # Применяем выбранную модель классификатора
            ])
            classifier_name = type(pipeline.named_steps['clf']).__name__  # Получаем имя модели
            print(f'\nЗапускаем классификацию для: {task.y_col}, модель: {classifier_name}\n')
            # Запуск процесса классификации для текущей модели и задачи
            ClassificationProcessor(data_path="../data/kursovik_data.csv",
                                    y_col=task.y_col,
                                    data_preprocessing_fun=task.preprocessor,
                                    param_grid=model.param_grid,
                                    pipeline=pipeline,
                                    split_criterion_func=lambda data, column: data[column].median()).run()


def run_app():
    """
    Основная функция приложения, которая предлагает пользователю выбрать режим работы: регрессия, классификация или всё.
    """
    print('Что делаем ? 1 - регрессию, 2 - кластеризацию, 3 - всё.')  # Выводим меню
    mode = input()  # Получаем выбор пользователя
    match mode:
        case '1':
            run_regression()  # Запуск режима регрессии
        case '2':
            run_classification()  # Запуск режима классификации
        case '3':
            run_regression()  # Запуск обоих режимов
            run_classification()
        case _:
            raise KeyError(f"Не знаю что это ... {mode}")  # Ошибка при неправильном выборе


# Запуск приложения
run_app()
