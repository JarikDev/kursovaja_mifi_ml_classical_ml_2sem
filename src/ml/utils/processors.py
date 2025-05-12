import warnings
from abc import ABC, abstractmethod
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, \
    ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR

# Импортируем функции предобработки из модуля
from src.ml.utils.feature_engineering import preprocess_cc50_classification, \
    preprocess_ic50_classification, preprocess_si_gt8_classification, preprocess_ic50_regression, \
    preprocess_cc50_regression, preprocess_si_regression, preprocess_si_median_classification, preprocess_df

# Отключаем предупреждения и используем Agg-бэкэнд для matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# Константы и гиперпараметры
data_path_const = "./data/kursovik_data.csv"
n_components = [2, 3, 4, 5, 7, 10, 0.95]  # Кол-во компонент PCA
feature_selection__k = [10, 20, 50, 'all']  # Кол-во признаков после отбора

# Структура для хранения модели и её параметров
ModelHolder = namedtuple('ModelHolder', ['model', 'param_grid'])

# Список моделей для регрессии с гиперпараметрами
regression_models = [
    ModelHolder(Ridge(), {
        'regressor__alpha': [0.01, 0.1, 1, 10, 100]
    }),
    ModelHolder(Lasso(max_iter=10000), {
        'regressor__alpha': [0.001, 0.01, 0.1, 1]
    }),
    ModelHolder(ElasticNet(max_iter=10000), {
        'regressor__alpha': [0.01, 0.1, 1],
        'regressor__l1_ratio': [0.2, 0.5, 0.8]
    }),
    ModelHolder(SVR(), {
        'regressor__C': [0.1, 1, 10],
        'regressor__epsilon': [0.01, 0.1, 0.5],
        'regressor__kernel': ['linear', 'rbf']
    }),
    ModelHolder(RandomForestRegressor(random_state=42), {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [5, 10, None]
    }),
    ModelHolder(GradientBoostingRegressor(random_state=42), {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 5]
    }),
    ModelHolder(KNeighborsRegressor(), {
        'regressor__n_neighbors': [3, 5, 7],
        'regressor__weights': ['uniform', 'distance']
    })
]

# Список моделей для классификации с гиперпараметрами
classification_models = [
    ModelHolder(LogisticRegression(solver='liblinear'), {
        'pca__n_components': n_components,
        'feature_selection__k': feature_selection__k,
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    }),
    ModelHolder(KNeighborsClassifier(), {
        'pca__n_components': n_components,
        'feature_selection__k': feature_selection__k,
        'clf__n_neighbors': [3, 5, 7, 9],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2]
    }),
    ModelHolder(RandomForestClassifier(random_state=42), {
        'pca__n_components': n_components,
        'feature_selection__k': feature_selection__k,
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [3, 5, None],
        'clf__min_samples_split': [2, 5, 10]
    }),
    ModelHolder(ExtraTreesClassifier(random_state=42), {
        'pca__n_components': n_components,
        'feature_selection__k': feature_selection__k,
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10]
    }),
    ModelHolder(GradientBoostingClassifier(random_state=42), {
        'pca__n_components': n_components,
        'feature_selection__k': feature_selection__k,
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_depth': [3, 5]
    }),
    # SVC закомментирован, так как слишком медленный для больших выборок
]

# Интерфейс для всех обработчиков (регрессия, классификация)
class ProcessorInterface(ABC):
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self) -> DataFrame:
        # Загрузка данных и предобработка
        data = pd.read_csv(self.data_path)
        return preprocess_df(data)

    @abstractmethod
    def run(self):
        pass


class RegressionProcessor(ProcessorInterface):
    def __init__(self, data_path, y_col, data_preprocessing_fun, param_grid, pipeline):
        super().__init__(data_path)
        self.data_path = data_path                  # Путь к CSV-файлу с данными
        self.y_col = y_col                          # Название целевой переменной (IC50, CC50 или SI)
        self.data_preprocessing_fun = data_preprocessing_fun  # Функция предобработки
        self.param_grid = param_grid                # Словарь с параметрами для GridSearchCV
        self.pipeline = pipeline                    # Pipeline для обработки и обучения модели

    def train_test_split_for_reg(self, data, test_size):
        # Удаляем лишние колонки и разделяем на X и y
        cols_to_delete = ['Unnamed: 0', 'SI', 'CC50', 'IC50']
        X = data.drop(columns=cols_to_delete, axis=1, errors='ignore')
        y = data[self.y_col]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def regression_report(self, regressor_name, y_true, y_pred):
        # Выводим основные метрики регрессии
        print(f"MAE модели {regressor_name}, для: {self.y_col}: {mean_absolute_error(y_true, y_pred):.4f}")
        print(f"MSE модели {regressor_name}, для: {self.y_col}: {mean_squared_error(y_true, y_pred):.4f}")
        print(f"RMSE модели {regressor_name}, для: {self.y_col}: {mean_squared_error(y_true, y_pred):.4f}")
        print(f"R2 модели {regressor_name}, для: {self.y_col}: {r2_score(y_true, y_pred):.4f}")

    def run(self):
        # Загружаем датасет с диска и применяем предварительную обработку
        # Включает, например, очистку данных, отбор признаков, нормализацию и т.д.
        data = self.load_data()
        data = self.data_preprocessing_fun(data)

        # Делим данные на обучающую и тестовую выборки
        # Выделяем целевую переменную и признаки, исключая ненужные колонки
        X_train, X_test, y_train, y_test = self.train_test_split_for_reg(data, 0.3)

        # Инициализируем GridSearchCV для подбора наилучших гиперпараметров модели
        # Используем 5-кратную кросс-валидацию и метрику R²
        grid = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Определяем имя используемой модели из пайплайна
        regressor_name = type(self.pipeline.named_steps['regressor']).__name__
        print(f"Лучшие параметры для: {regressor_name}")
        print(grid.best_params_)

        # Получаем предсказания на тестовой выборке
        y_pred = grid.predict(X_test)

        # Выводим метрики качества регрессии: MAE, MSE, RMSE, R²
        print(f"Отчёт по регрессии модели {regressor_name}, для: {self.y_col}")
        self.regression_report(regressor_name, y_test, y_pred)

        # Визуализируем поведение модели на тестовой выборке:
        # распределение ошибок, остатки, сравнение предсказаний и фактических значений
        self.plot_regression_diagnostics(regressor_name, y_test, y_pred)

    def plot_regression_diagnostics(self, regressor_name, y_test, y_pred):
        # Полный набор графиков диагностики
        residuals = y_test - y_pred
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle(f'Диагностика модели: {regressor_name} для {self.y_col}', fontsize=14)

        # Гистограмма ошибок
        sns.histplot(residuals, bins=30, kde=True, ax=axes[0])
        axes[0].set_title('Распределение ошибок')
        axes[0].set_xlabel('Остатки')
        axes[0].grid(True)

        # Остатки vs Предсказания
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=axes[1])
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Предсказанные значения')
        axes[1].set_ylabel('Остатки')
        axes[1].set_title('Остатки vs Предсказания')
        axes[1].grid(True)

        # Предсказания vs Фактические
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=axes[2])
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[2].set_xlabel('Фактические значения')
        axes[2].set_ylabel('Предсказанные значения')
        axes[2].set_title('Факт vs Предсказание')
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f'diagnostics_{regressor_name}_{self.y_col}.png'
        plt.savefig(filename)
        plt.close()

    def plot_errors_histogram(self, regressor_name, y_test, y_pred):
        # Отдельный график: распределение ошибок
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title(f'Распределение ошибок (остатков) модели регрессии: {regressor_name} для {self.y_col}')
        plt.xlabel('Остатки')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pic_{regressor_name}_{self.y_col}_residual_distribution.png')

    def plot_residuals(self, regressor_name, y_test, y_pred):
        # Остатки по предсказанным значениям
        residuals = y_test - y_pred
        plt.figure(figsize=(7, 4))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel(f'Предсказанные {self.y_col}')
        plt.ylabel('Остатки')
        plt.title(f'График остатков модели регрессии: {regressor_name} для {self.y_col}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pic_{regressor_name}_{self.y_col}_residuals.png')

    def plot_predictions_vs_actual_values(self, regressor_name, y_test, y_pred):
        # Факт против предсказания
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel(f'Фактические  {self.y_col}')
        plt.ylabel(f'Предсказанные  {self.y_col}')
        plt.title(f'Факт vs Предсказание модели регрессии: {regressor_name} для {self.y_col}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"pic_{regressor_name}_{self.y_col}_regression.png")



class ClassificationProcessor(ProcessorInterface):
    def __init__(self, data_path, y_col, data_preprocessing_fun, param_grid, pipeline, split_criterion_func):
        # Инициализация параметров для обработки задач классификации
        super().__init__(data_path)
        self.data_path = data_path
        self.y_col = y_col  # Целевая переменная
        self.data_preprocessing_fun = data_preprocessing_fun  # Функция предобработки данных
        self.param_grid = param_grid  # Сетка гиперпараметров для GridSearch
        self.pipeline = pipeline  # Pipeline модели
        self.split_criterion_func = split_criterion_func  # Критерий разбиения классов

    def train_test_split_for_clusterization(self, data, test_size) -> (str, str, str, str):
        # Удаление ненужных колонок и деление на X/y
        to_delete = ['Unnamed: 0', 'SI', 'CC50', 'IC50', 'target']
        X = data.drop(columns=to_delete, axis=1, errors='ignore')
        y = data['target']
        # Разделение данных на train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X.columns, X_train, X_test, y_train, y_test

    def plot_predictions_vs_actual_values(self, regressor_name, y_test, y_pred):
        # Визуализация предсказаний против фактических значений
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel(f'Фактические {self.y_col}')
        plt.ylabel(f'Предсказанные {self.y_col}')
        plt.title(f'Факт vs Предсказание классификации: {regressor_name} {self.y_col}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"pic_classification_{regressor_name}_{self.y_col}")

    def visualize_confusion_matrix(self, classifier_name, model, X_test, y_test):
        # Построение матрицы ошибок
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
        plt.title(f'Confusion Matrix для {classifier_name} {self.y_col}')
        plt.savefig(f'pic_Confusion_Matrix_for_{classifier_name}_{self.y_col}')

    def visualize_pr_curve(self, classifier_name, model, X_test, y_test):
        # Построение PR-кривой
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_prec:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve для {classifier_name} {self.y_col}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pic_Precision_Recall_Curve_{classifier_name}_{self.y_col}")

    def visualize_roc_curve(self, classifier_name, model, X_test, y_test):
        # Построение ROC-кривой
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve для {classifier_name} {self.y_col}')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'pic_ROC_Curve_for_{classifier_name}_{self.y_col}')

    def get_most_important_features(self, model, feature_names, top_n):
        # Получение и визуализация важнейших признаков модели
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        feature_importances.head(top_n).plot(kind='barh', figsize=(8, 10))
        plt.title('Top 30 Feature Importances (RandomForest)')
        plt.gca().invert_yaxis()
        return feature_importances.head(top_n).index.tolist()

    def visualize_tsne_3d(self, classifier_name, X_test, y_pred):
        # 3D t-SNE визуализация
        tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=1000)
        X_vis = tsne.fit_transform(X_test)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        palette = {0: "#1f77b4", 1: "#d62728"}

        for class_label in np.unique(y_pred):
            idx = (y_pred == class_label)
            ax.scatter(X_vis[idx, 0], X_vis[idx, 1], X_vis[idx, 2],
                       label=f"Класс {class_label}", color=palette[class_label],
                       alpha=0.7, edgecolors='k')

        ax.set_title(f"3D t-SNE визуализация ({classifier_name})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"pic_tsne3d_classification_{classifier_name}_{self.y_col}.png")

    def visualize_clusters_with_tsne(self, classifier_name, X_test, y_pred):
        # 2D t-SNE визуализация
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        X_vis = tsne.fit_transform(X_test)

        plt.figure(figsize=(8, 6))
        palette = {0: "#1f77b4", 1: "#d62728"}
        sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y_pred,
                        palette=palette, edgecolor='black', alpha=0.8, s=70)
        plt.title(f"Визуализация классов ({classifier_name})")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Классы")
        plt.tight_layout()
        plt.savefig(f"pic_classification_{classifier_name}_{self.y_col}.png")

    def show_class_balance(self, data):
        # Вывод баланса классов
        class_counts = data['target'].value_counts()
        print(f'classes count: {class_counts}')

    def visualize_classification_diagnostics(self, classifier_name, model, X_test, y_test, y_pred):
        # Визуализация общей диагностики классификатора
        y_proba = model.predict_proba(X_test)[:, 1]
        tsne_2d = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(X_test)
        tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42).fit_transform(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f"Диагностика классификатора: {classifier_name} для {self.y_col}", fontsize=16)
        palette = {0: "#1f77b4", 1: "#d62728"}

        ax1 = fig.add_subplot(2, 2, 1)
        sns.scatterplot(x=tsne_2d[:, 0], y=tsne_2d[:, 1], hue=y_pred, palette=palette, ax=ax1,
                        edgecolor='k', alpha=0.7, s=60)
        ax1.set_title("t-SNE (2D)")

        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        for class_label in np.unique(y_pred):
            idx = y_pred == class_label
            ax2.scatter(tsne_3d[idx, 0], tsne_3d[idx, 1], tsne_3d[idx, 2],
                        label=f"Класс {class_label}", color=palette[class_label],
                        alpha=0.6, edgecolors='k')
        ax2.set_title("t-SNE (3D)")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(recall, precision, label=f'AP = {avg_prec:.2f}')
        ax3.set_title("Precision-Recall Curve")
        ax3.grid(True)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax4.plot([0, 1], [0, 1], 'k--')
        ax4.set_title("ROC Curve")
        ax4.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"pic_classification_diagnostics_{classifier_name}_{self.y_col}.png"
        plt.savefig(filename)
        plt.close()
        print(f"[diagnostics] Сохранено: {filename}")

    def run(self):
        # Основной метод запуска классификации

        # Загрузка исходных данных из CSV-файла и базовая предобработка
        data: DataFrame = self.load_data()

        # Применение пользовательской функции предобработки (например, удаление ненужных признаков, нормализация и т.д.)
        data = self.data_preprocessing_fun(data)

        # Удаление строк с пропущенными значениями
        data = data.dropna()

        # Создание целевой переменной 'target' на основе пороговой функции
        # Например, если split_criterion_func = get_median_for_col, то все значения выше медианы -> класс 1, иначе -> класс 0
        data['target'] = np.where(data[self.y_col] > self.split_criterion_func(data, self.y_col), 1, 0)

        # Вывод в консоль баланса классов в целевой переменной
        self.show_class_balance(data)

        # Разделение данных на обучающую и тестовую выборки
        columns, X_train, X_test, y_train, y_test = self.train_test_split_for_clusterization(data, 0.3)

        #  Подбор гиперпараметров с помощью GridSearchCV
        # Проводится перекрёстная проверка (cv=5) с метрикой качества 'accuracy'
        grid = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Получение имени обученного классификатора (например, RandomForestClassifier)
        classifier_name = type(self.pipeline.named_steps['clf']).__name__

        # Вывод наилучших найденных параметров модели
        print(f"Лучшие параметры для: {classifier_name}")
        print(grid.best_params_)

        # Предсказание классов на тестовой выборке
        y_pred = grid.predict(X_test)

        # Вывод отчёта по классификации (precision, recall, f1-score, support)
        print(f"Отчёт по классификации модели {classifier_name}:")
        print(classification_report(y_test, y_pred))

        # Извлечение обученной модели с лучшими параметрами
        model = grid.best_estimator_

        # Построение диагностик классификации — ROC, PR-кривые, t-SNE и т.д.
        self.visualize_classification_diagnostics(classifier_name, model, X_test, y_test, y_pred)


# Функция для получения медианного значения признака (используется как порог для бинарной классификации)
def get_median_for_col(data, column):
    return data[column].median()

# Функция-заглушка, возвращающая фиксированное значение 8 (используется для задач, где порог установлен вручную, например, SI > 8)
def get_8(data, column):
    return 8

# Определяем структуру TaskHolder с полями:
# y_col — имя целевой переменной,
# preprocessor — функция предобработки данных,
# models — список моделей и параметров (ModelHolder),
# split_criterion_func — функция определения порога для бинарной классификации
TaskHolder = namedtuple('TaskHolder', ['y_col', 'preprocessor', 'models', 'split_criterion_func'])

# Список задач классификации:
# Для каждой целевой переменной задаётся своя функция предобработки, модели и функция разделения на классы
classification_tasks = [
    TaskHolder('CC50', preprocess_cc50_classification, classification_models, get_median_for_col),  # разделение по медиане
    TaskHolder('IC50', preprocess_ic50_classification, classification_models, get_median_for_col),  # разделение по медиане
    TaskHolder('SI', preprocess_si_median_classification, classification_models, get_median_for_col),  # разделение по медиане
    TaskHolder('SI', preprocess_si_gt8_classification, classification_models, get_8),  # разделение по порогу SI > 8
]

# Список задач регрессии:
# Порог разделения не используется (None), так как задача — предсказание непрерывной переменной
regression_tasks = [
    TaskHolder('CC50', preprocess_cc50_regression, regression_models, None),
    TaskHolder('IC50', preprocess_ic50_regression, regression_models, None),
    TaskHolder('SI', preprocess_si_regression, regression_models, None),
]

