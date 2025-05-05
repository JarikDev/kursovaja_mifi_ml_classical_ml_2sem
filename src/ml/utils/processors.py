import warnings
from abc import ABC, abstractmethod
from collections import namedtuple

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

from src.ml.utils.feature_engineering import preprocess_cc50_classification, \
    preprocess_ic50_classification, preprocess_si_gt8_classification, preprocess_ic50_regression, \
    preprocess_cc50_regression, preprocess_si_regression, preprocess_si_median_classification

warnings.filterwarnings('ignore')
data_path_const = "./data/kursovik_data.csv"
n_components = [2, 3, 4, 5, 7, 10, 0.95]
feature_selection__k = [10, 20, 50, 'all']

ModelHolder = namedtuple('ModelHolder', ['model', 'param_grid'])

# создаём колелкцию моделей для регрессии и их гиперпараметров для перебора
regression_models = [
    # Вариант 1. Ridge
    ModelHolder(Ridge(), {
        'regressor__alpha': [0.01, 0.1, 1, 10, 100]
    }),
    # Вариант 2. Lasso
    ModelHolder(Lasso(max_iter=10000), {
        'regressor__alpha': [0.001, 0.01, 0.1, 1]
    }),
    # Вариант 3. ElasticNet
    ModelHolder(ElasticNet(max_iter=10000), {
        'regressor__alpha': [0.01, 0.1, 1],
        'regressor__l1_ratio': [0.2, 0.5, 0.8]
    }),
    # Вариант 4. Support Vector Regression (SVR)
    ModelHolder(SVR(), {
        'regressor__C': [0.1, 1, 10],
        'regressor__epsilon': [0.01, 0.1, 0.5],
        'regressor__kernel': ['linear', 'rbf']
    }),
    # Вариант 5. Random Forest Regressor
    ModelHolder(RandomForestRegressor(random_state=42), {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [5, 10, None]
    }),
    # Вариант 6. Gradient Boosting Regressor
    ModelHolder(GradientBoostingRegressor(random_state=42), {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 5]
    }),
    # Вариант 7. K-Nearest Neighbors Regression (KNN)
    ModelHolder(KNeighborsRegressor(), {
        'regressor__n_neighbors': [3, 5, 7],
        'regressor__weights': ['uniform', 'distance']
    })
]

# создаём колелкцию моделей для классификации и их гиперпараметров для перебора
classification_models = [
    # Вариант 1. Логистическая регрессия (LogisticRegression)
    ModelHolder(LogisticRegression(solver='liblinear'), {
        'pca__n_components': n_components,  # Подберем количество компонент
        'feature_selection__k': feature_selection__k,
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    }),
    # Вариант 2. K-ближайших соседей (KNeighborsClassifier)
    ModelHolder(KNeighborsClassifier(), {
        'pca__n_components': n_components,  # Подберем количество компонент
        'feature_selection__k': feature_selection__k,
        'clf__n_neighbors': [3, 5, 7, 9],
        'clf__weights': ['uniform', 'distance'],
        'clf__p': [1, 2]  # 1=Manhattan, 2=Euclidean
    }),
    # Вариант 4. Случайный лес (RandomForestClassifier)
    ModelHolder(RandomForestClassifier(random_state=42), {
        'pca__n_components': n_components,  # Подберем количество компонент
        'feature_selection__k': feature_selection__k,
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [3, 5, None],
        'clf__min_samples_split': [2, 5, 10]
    }),
    # Вариант 4. Экстра-деревья (ExtraTreesClassifier)
    ModelHolder(ExtraTreesClassifier(random_state=42), {
        'pca__n_components': n_components,  # Подберем количество компонент
        'feature_selection__k': feature_selection__k,
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 5, 10]
    }),
    # Вариант 5. Градиентный бустинг (GradientBoostingClassifier)
    ModelHolder(GradientBoostingClassifier(random_state=42), {
        'pca__n_components': n_components,  # Подберем количество компонент
        'feature_selection__k': feature_selection__k,
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.01, 0.1],
        'clf__max_depth': [3, 5]
    }),
    # Вариант 6. Метод опорных векторов (SVC) - очень медленно
    # ModelHolder(SVC(), {
    #     'pca__n_components': n_components,  # Подберем количество компонент
    #     'clf__C': [0.1, 1, 10],
    #     'clf__kernel': ['linear', 'rbf'],  # rbf is too slow
    #     # 'clf__kernel': ['linear'],
    #     'clf__gamma': ['scale', 'auto']
    # })
]


class ProcessorInterface(ABC):

    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self) -> DataFrame:
        data = pd.read_csv(self.data_path)
        data.rename(columns={'CC50, mM': 'CC50'}, inplace=True)
        data.rename(columns={'IC50, mM': 'IC50'}, inplace=True)
        data = data[data['IC50'] < 1001]
        data['SI'] = data['CC50'] / data['IC50']
        data = data[data['SI'] < 1001]
        return data

    @abstractmethod
    def run(self):
        pass


class RegressionProcessor(ProcessorInterface):
    def __init__(self, data_path, y_col, data_preprocessing_fun, param_grid, pipeline):
        super().__init__(data_path)
        self.data_path = data_path
        self.y_col = y_col
        self.data_preprocessing_fun = data_preprocessing_fun
        self.param_grid = param_grid
        self.pipeline = pipeline

    def train_test_split_for_reg(self, data, test_size) -> (str, str, str, str):
        # колонки которые должны быть удалены
        cols_to_delete = ['Unnamed: 0', 'SI', 'CC50', 'IC50']
        X = data.drop(columns=cols_to_delete, axis=1, errors='ignore')
        y = data[self.y_col]
        # Разделим данные на обучающую и тестовую выборки
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def regression_report(self, regressor_name, y_true, y_pred):
        print(f"MAE модели {regressor_name}, для: {self.y_col}: {mean_absolute_error(y_true, y_pred):.4f}")
        print(f"MSE модели {regressor_name}, для: {self.y_col}: {mean_squared_error(y_true, y_pred):.4f}")
        print(
            f"RMSE модели {regressor_name}, для: {self.y_col}: {mean_squared_error(y_true, y_pred):.4f}")
        print(f"R2 модели {regressor_name}, для: {self.y_col}: {r2_score(y_true, y_pred):.4f}")

    def run(self):
        # Загрузка данных
        data: DataFrame = self.load_data()
        # Предобработка данных
        data = self.data_preprocessing_fun(data)
        # Разделим данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = self.train_test_split_for_reg(data, 0.3)
        # подбираем гиперпараметры с GridSearchCV
        grid = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Лучшая модель
        regressor_name = type(self.pipeline.named_steps['regressor']).__name__
        print(f"Лучшие параметры для: {regressor_name}")
        print(grid.best_params_)

        # Предсказание
        y_pred = grid.predict(X_test)

        print(f"Отчёт по регрессии модели {regressor_name}, для: {self.y_col}")
        # выводим статистику полученной модели и регрессии
        self.regression_report(regressor_name, y_test, y_pred)
        # Визуализируем предсказание против реальных значений
        self.plot_predictions_vs_actual_values(regressor_name, y_test, y_pred)
        self.plot_residuals(regressor_name, y_test, y_pred)
        self.plot_errors_histogram(regressor_name, y_test, y_pred)

    def plot_errors_histogram(self, regressor_name, y_test, y_pred):
        residuals = y_test - y_pred
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title(f'Распределение ошибок (остатков) модели регрессии: {regressor_name} для {self.y_col}')
        plt.xlabel('Остатки')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pic_{regressor_name}_{self.y_col}_residual_distribution.png')
        # plt.show()

    def plot_residuals(self, regressor_name, y_test, y_pred):
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
        # plt.show()

    def plot_predictions_vs_actual_values(self, regressor_name, y_test, y_pred):
        # Plot predictions vs actual values
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel(f'Фактические  {self.y_col}')
        plt.ylabel(f'Предсказанные  {self.y_col}')
        plt.title(f'Факт vs Предсказание модели регрессии: {regressor_name} для {self.y_col}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"pic_{regressor_name}_{self.y_col}_regression.png")
        # plt.show()


class ClassificationProcessor(ProcessorInterface):
    def __init__(self, data_path, y_col, data_preprocessing_fun, param_grid, pipeline, split_criterion_func):
        super().__init__(data_path)
        self.data_path = data_path
        self.y_col = y_col
        self.data_preprocessing_fun = data_preprocessing_fun
        self.param_grid = param_grid
        self.pipeline = pipeline
        self.split_criterion_func = split_criterion_func

    def train_test_split_for_clusterization(self, data, test_size) -> (
            str, str, str, str):
        to_delete = ['Unnamed: 0', 'SI', 'CC50', 'IC50', 'target']
        X = data.drop(columns=to_delete, axis=1, errors='ignore')
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X.columns, X_train, X_test, y_train, y_test

    def plot_predictions_vs_actual_values(self, regressor_name, y_test, y_pred):
        # Plot predictions vs actual values
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel(f'Фактические {self.y_col}')
        plt.ylabel(f'Предсказанные {self.y_col}')
        plt.title(f'Факт vs Предсказание классификации: {regressor_name} {self.y_col}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"pic_classification_{regressor_name}_{self.y_col}")
        # plt.show()

    def visualize_confusion_matrix(self, classifier_name, model, X_test, y_test):
        # Матрица ошибок
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
        plt.title(f'Confusion Matrix для {classifier_name} {self.y_col}')
        plt.savefig(f'pic_Confusion_Matrix_for_{classifier_name}_{self.y_col}')
        # plt.show()

    def visualize_pr_curve(self, classifier_name, model, X_test, y_test):
        # Предсказанные вероятности (для положительного класса)
        y_proba = model.predict_proba(X_test)[:, 1]  # или pipeline.predict_proba(X_test)[:, 1]

        # Вычисляем precision и recall
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)

        # Строим график
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_prec:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve для {classifier_name} {self.y_col}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pic_Precision_Recall_Curve_{classifier_name}_{self.y_col}")
        # plt.show()

    def visualize_roc_curve(self, classifier_name, model, X_test, y_test):
        # ROC-кривая
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) ')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve для {classifier_name} {self.y_col}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(f'pic_ROC_Curve_for_{classifier_name}_{self.y_col}')
        # plt.show()

    def get_most_important_features(self, model, feature_names, top_n):
        # Важность признаков
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

        # Визуализация топ-30 признаков
        feature_importances.head(top_n).plot(kind='barh', figsize=(8, 10))
        plt.title('Top 30 Feature Importances (RandomForest)')
        plt.gca().invert_yaxis()
        # plt.show()

        # Отбор лучших признаков
        return feature_importances.head(top_n).index.tolist()

    def visualize_clusters_with_tnse(self, classifier_name, X_test_selected, y_pred_selected, max_outliers_per_row=3):
        """
        Визуализация кластеров через t-SNE:
        - мягкая фильтрация выбросов (robust z-score)
        - настраиваемый порог max_outliers_per_row
        - безопасный подбор perplexity
        - подписи и сохранение графика
        """
        # === 1. Robust Z-score фильтрация выбросов
        Q1 = np.percentile(X_test_selected, 25, axis=0)
        Q3 = np.percentile(X_test_selected, 75, axis=0)
        IQR = Q3 - Q1 + 1e-8  # защитим от деления на 0

        z_robust = np.abs((X_test_selected - Q1) / IQR)
        row_outlier_counts = (z_robust >= 3).sum(axis=1)
        mask = row_outlier_counts <= max_outliers_per_row

        X_test_clean = X_test_selected[mask]
        y_pred_clean = y_pred_selected[mask]
        n_samples = X_test_clean.shape[0]

        print(f"[t-SNE] Оставлено объектов после фильтрации: {n_samples}")

        # === 2. Проверка: достаточно ли объектов
        if n_samples < 5:
            print(f"[!] t-SNE пропущен: после фильтрации осталось только {n_samples} объектов.")
            return

        # === 3. Безопасный подбор perplexity
        perplexity = min(30, max(5, n_samples // 3))

        # === 4. t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        X_test_tsne = tsne.fit_transform(X_test_clean)

        # === 5. Визуализация
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_pred_clean,
                              cmap='coolwarm', alpha=0.7, edgecolors='k')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f'Кластеры после классификации ({classifier_name}, y = {self.y_col})')
        plt.grid(True)
        plt.legend(*scatter.legend_elements(), title="Классы")

        # подпись параметров
        plt.text(0.99, 0.01, f'n={n_samples}, perplexity={perplexity}',
                 ha='right', va='bottom', transform=plt.gca().transAxes,
                 fontsize=8, bbox=dict(facecolor='white', edgecolor='gray'))

        plt.tight_layout()
        plt.savefig(f"pic_classification_{classifier_name}_{self.y_col}.png")
        # plt.show()  # включи при отладке
        print(f"[t-SNE] Сохранено: pic_classification_{classifier_name}_{self.y_col}.png")

        # plt.show()

    # Проверка баланса классов
    def show_class_balance(self, data):
        class_counts = data['target'].value_counts()
        print(f'classes count: {class_counts}')

    def run(self):
        # Загрузка данных
        data: DataFrame = self.load_data()
        # Предобработка данных
        data = self.data_preprocessing_fun(data)
        data['target'] = np.where(data[self.y_col] > self.split_criterion_func(data, self.y_col), 1, 0)
        self.show_class_balance(data)
        # Разделим данные на обучающую и тестовую выборки
        columns, X_train, X_test, y_train, y_test = self.train_test_split_for_clusterization(data, 0.3)

        # GridSearchCV
        grid = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        # Оценка модели
        classifier_name = type(self.pipeline.named_steps['clf']).__name__
        print(f"Лучшие параметры для: {classifier_name}")
        print(grid.best_params_)
        y_pred = grid.predict(X_test)
        print(f"Отчёт по классификации модели {classifier_name}:")
        print(classification_report(y_test, y_pred))
        model = grid.best_estimator_

        # Матрица ошибок
        self.visualize_confusion_matrix(classifier_name, model, X_test, y_test)
        # ROC-кривая
        self.visualize_roc_curve(classifier_name, model, X_test, y_test)
        # Визуализация кластеров через t-SNE
        self.visualize_clusters_with_tnse(classifier_name, X_test, y_pred)
        # Визуализация Precision-Recall кривой
        self.visualize_pr_curve(classifier_name, model, X_test, y_test)

        # # Важность признаков. Отбор лучших признаков
        # selected_features = self.get_most_important_features(model, columns, 30)
        # # Пересборка выборок
        # X_train_selected = pd.DataFrame(X_train, columns=columns)[selected_features].values
        # X_test_selected = pd.DataFrame(X_test, columns=columns)[selected_features].values
        # # Переобучение модели на лучших признаках
        #
        # # model, y_pred_selected = self.model_processing_fun(X_train_selected, X_test_selected, y_train, y_test)
        # # GridSearchCV
        # grid = GridSearchCV(self.pipeline, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        # grid.fit(X_train_selected, y_train)
        #
        # # Оценка модели
        # classifier_name = type(self.pipeline.named_steps['clf']).__name__
        # print(f"Лучшие параметры для: {classifier_name}")
        # print(grid.best_params_)
        # y_pred = grid.predict(X_test)
        # print(f"Отчёт по классификации модели {classifier_name}:", classification_report(y_test, y_pred))
        # model = grid.best_estimator_
        #
        # # Матрица ошибок
        # self.visualize_confusion_matrix(model, X_test_selected, y_test)
        # # ROC-кривая
        # self.visualize_roc_curve(model, X_test_selected, y_test)
        # # Визуализация кластеров через t-SNE
        # self.visualize_clusters_with_tnse(X_test_selected, y_pred)


def get_median_for_col(data, column):
    return data[column].median()


def get_8(data, column):
    return 8


TaskHolder = namedtuple('TaskHolder', ['y_col', 'preprocessor', 'models', 'split_criterion_func'])

classification_tasks = [
    TaskHolder('CC50', preprocess_cc50_classification, classification_models, get_median_for_col),
    TaskHolder('IC50', preprocess_ic50_classification, classification_models, get_median_for_col),
    TaskHolder('SI', preprocess_si_median_classification, classification_models, get_median_for_col),
    TaskHolder('SI', preprocess_si_gt8_classification, classification_models, get_8),
]

regression_tasks = [
    TaskHolder('CC50', preprocess_cc50_regression, regression_models, None),
    TaskHolder('IC50', preprocess_ic50_regression, regression_models, None),
    TaskHolder('SI', preprocess_si_regression, regression_models, None),
]
