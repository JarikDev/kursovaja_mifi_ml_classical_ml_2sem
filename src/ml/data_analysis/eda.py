import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from tabulate import tabulate

# Настройки визуализации
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)


def visualise_distribution(df, column):
    plt.figure()
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Распределение {column}")
    plt.xlabel(column)
    plt.ylabel("Количество")
    plt.savefig(f'./out/distribution_{column}')


def visualise_distributions(df, columns):
    ncols = int(abs(math.sqrt(len(columns))))
    n = len(columns)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(10 * ncols, 5 * nrows)
    )

    # Строим графики и скрываем пустые области
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            column = columns[idx]
            sns.histplot(df[column], ax=ax, bins=30, kde=True)
            ax.set_title(column)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"./out/distributions_{'_'.join(columns)}")


def visualise_box_plot(df, column):
    plt.figure()
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot для {column}")
    plt.savefig(f"./out/boxplot_{column}")


def visualise_box_plots(df, columns):
    figsize_base = 8
    orientation = 'h'

    if not columns:
        print("Нет колонок для отображения!")
        return

    n = len(columns)
    ncols = 1
    if n > 3:
        ncols = min(math.ceil(math.sqrt(n)), 4)  # Ограничиваем максимум 4 колонки в строке
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_base * ncols, figsize_base * nrows * 0.5)
    )

    # Настройки стиля для boxplot
    boxprops = dict(facecolor='#1f77b4', linewidth=1.5)  # Цветовая схема matplotlib
    flierprops = dict(marker='o', markersize=3, markerfacecolor='red')

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            column = columns[idx]

            # Строим boxplot с учетом ориентации
            if orientation == 'v':
                sns.boxplot(y=df[column], ax=ax, width=0.6, boxprops=boxprops, flierprops=flierprops)
                ax.set_ylabel("Значения", fontsize=9)
            else:
                sns.boxplot(x=df[column], ax=ax, width=0.6, boxprops=boxprops, flierprops=flierprops)
                ax.set_xlabel("Значения", fontsize=9)

            # Настройка оформления
            ax.set_title(f"{column}\n", fontsize=10)
            ax.grid(True, axis='y' if orientation == 'v' else 'x', alpha=0.3)
            ax.tick_params(labelsize=8)

            # Убираем лишние оси
            if orientation == 'v':
                ax.set_xlabel('')
                ax.set_xticks([])
            else:
                ax.set_ylabel('')
                ax.set_yticks([])
        else:
            ax.set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig(f'./out/boxplots_{'_'.join(columns)}')


def visualise_top_correlated_features(corr_matrix, target, target_cols, n):
    top_feats = corr_matrix[target].drop(target_cols).abs().sort_values(ascending=False).head(n)
    plt.figure(figsize=(20, 12))
    sns.barplot(x=top_feats.values, y=top_feats.index)
    plt.title(f"Топ 10 признаков по корреляции с {target}")
    plt.xlabel("Абсолютная корреляция")
    plt.ylabel("Признак")
    plt.savefig(f'./out/top_10_correlated_with{target}')


def visualise_class_balance(df, column):
    sns.countplot(x=df[column])
    plt.title(f"Распределение классов для {column}")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.savefig(f'./out/class_balance_{target}')


def visualise_class_balances(df, columns):
    n = len(columns)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    width_per_plot = 4
    height_per_plot = 3
    figsize = (n_cols * width_per_plot, n_rows * height_per_plot)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.countplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Распределение классов для {col}")
        axes[i].set_xlabel("Класс")
        axes[i].set_ylabel("Количество")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'./out/class_balances_{"_".join(columns).replace(">","_greater_")}')


def visualise_corr_mx(df, columns=None):
    plt.figure()
    if columns is None or len(columns) == 0:
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    else:
        sns.heatmap(df[target_cols].corr(), annot=True, cmap='coolwarm')
    plt.title("Матрица корреляций")
    plt.savefig(f'./out/corr_mx{"_".join(columns)}')


# Загрузка данных
print("\nЗагрузка данных")
df: DataFrame = pd.read_csv("../data/kursovik_data.csv")

# Переименуем некоторые колонки
print('\nПереименуем некоторые колонки')
df = df.rename(columns={
    "IC50, mM": "IC50",
    "CC50, mM": "CC50"
})

### Общая информация о данных
print("\nОбщая информация о данных:")
print(df.info())
print(f'\nФорма датасета: {df.shape}')

print("\nСписок признаков:", df.columns.tolist())
print('''\nЭтот набор колонок представляет собой богатое множество химических дескрипторов, 
позволяющих анализировать физико-химические, топологические, электронные и структурные свойства молекул.
''')
print('\nДля большей наглядности сведём полученную информацию в таблицу.')
feature_applicability_df = pd.read_csv('../data/feature_applicability.csv')
print(feature_applicability_df)
feature_applicability_df.to_csv("./out/applicability.csv")
print(tabulate(feature_applicability_df, headers='keys', tablefmt='fancy_grid'))

print('''\nВывод: все признаки полезны для решения всех задач. 
Будем использовать все, плюс сгенерированные в результате фича инжениринга. 
Потом с помощью РСА понизим размерность в процессе построения модели.''')

print("\nОписательная статистика признаков:")
print(df.describe().T)
df.describe().T.to_csv("./out/stats.csv")
print('''\nВыводы:
1. Биологические переменные (IC50, CC50, SI)
1.1. IC50: сильная положительная асимметрия, медиана около 47, максимум > 4000
1.2. CC50: также асимметрично, медиана около 411, максимум > 4500
1.3. SI: экстремально широкий диапазон от 0.01 до 15620, медиана около 3.8
1.4. Вывод: требуется логарифмическое преобразование IC50, CC50 и особенно SI

2. Молекулярные характеристики
2.1. MolWt, ExactMolWt, HeavyAtomMolWt — диапазон от 100 до 900, большинство значений 250–400
2.2. LabuteASA и TPSA — разброс до 350–400, средние значения умеренные
2.3. NumValenceElectrons, HeavyAtomCount, NumRotatableBonds — высокая вариативность
2.4. Вывод: требуется нормализация и масштабирование значений

3. Зарядовые характеристики
3.1. MaxPartialCharge и MinPartialCharge — сбалансированы, присутствует широкий диапазон
3.2. MaxAbsPartialCharge ≈ 0.4–0.5, MinAbsPartialCharge — близки к нулю
3.3. Вывод: возможно использование производных признаков — разностей, отношений, логарифмов

4. Топологические индексы
4.1. Chi, Kappa, BertzCT — распределены умеренно, значительная вариативность
4.2. Ipc — аномально высокое среднее и разброс, максимум 3.95e+13
4.3. Вывод: требуется логарифмическое преобразование Ipc и BertzCT

5. Поверхностные атомные параметры (VSA)
5.1. Многие признаки имеют нулевую медиану и спарсные распределения
5.2. Примеры: PEOE_VSA14, EState_VSA11, SlogP_VSA9 — медиана и 75% равны 0
5.3. Вывод: признаки можно агрегировать (сумма), понизить размерность (PCA) или удалить

6. Фрагментные признаки (fr_*)
6.1. Большинство бинарные, медиана и 75-й перцентиль равны 0
6.2. Исключения: fr_benzene, fr_C_O, fr_ketone — встречаются чаще
6.3. Вывод: использовать агрегацию по группам, фильтрацию по дисперсии

7. Общие рекомендации
7.1. Логарифмировать: IC50, CC50, SI, Ipc, BertzCT, MolWt
7.2. Стандартизировать признаки с широким диапазоном (z-score, PowerTransformer)
7.3. Удалить константные признаки (например, NumRadicalElectrons, fr_* = 0 всегда)
7.4. Снизить размерность признаков: VSA, fr_, Chi, BCUT* — через PCA, SelectKBest
7.5. Добавить производные признаки: разности, отношения, логарифмы (например, TPSA / MolWt)''')


print("\nАнализ пропусков.")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\nПризнаки с пропущенными значениями:")
print(missing)
print("\nВывод: пропусков мало, можно просто удалить.")


print("\nПроведём пред обработку данных")

print("\nУдалим явно ненужную колонку")
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("\nПропуски есть, но их очень мало, просто удалим их.")
target_cols = ['IC50', 'CC50', 'SI']
df = df.dropna()

print("\nПроведём анализ распределения целевых признаков. Визуализируем распределения с помощью гистограмм.")
visualise_distributions(df, target_cols)
print('''\nРаспределения признаков сильно скошены и нуждаются в логарифмировании. 
Есть подозрение на наличие выбросов.''')


print("\nПроведём анализ выбросов. Визуализируем с помощью ящиков с усами.")
visualise_box_plots(df, target_cols)
print('''\nПрисутствуют выбросы. 
Из теоретической справки нам известно, что IC50 более 1000 это почти всегда выброс и очень низкая активность.
Такие соединения обычно отбрасывают как бесполезные. 
CC50 может быть каким угодно, чем больше тем менее токсично. 
SI лучше пересчитать, потому как оно вычисляется как CC50 / IC50, и обычно не бывают выше 1000. 
Следовательно, есть выбросы и их нужно убрать. Для начала обрежем по верхнему перцентилю (например, 99-й).''')

for col in target_cols:
    upper = df[col].quantile(0.99)
    df = df[df[col] <= upper]

print('\nУдалим строки, в которых значения целевых признаков можно считать выбросами.')
df = df[df['IC50'] < 1000]
df['SI'] = df['CC50'] / df['IC50']
df = df[df['SI'] < 1000]

print('''\nТак как распределения целевых признаков сильно скошены, а это ухудшит обучение моделей, 
они требуют логарифмирования для приведения к виду более похожему на нормальное распределение.''')
print("\nПроведём логарифмирование, добавим новые признаки на основе целевых.")
for col in target_cols:
    log_col = f"log_{col.split(',')[0]}"
    df[log_col] = np.log1p(df[col])

print("\nРасширим перечень целевых признаков добавив логарифмированные.")
target_cols = ['IC50', 'CC50', 'SI', 'log_IC50', 'log_CC50', 'log_SI']

print('''\nВизуализируем распределения логарифмированных и не логарифмированных целевых признаков 
с помощью гистограмм после удаления выбросов.''')

visualise_distributions(df, target_cols)


print("\nВизуализируем корреляцию между целевыми признаками, в том числе созданными на их основе.")
plt.figure()
sns.heatmap(df[target_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Корреляции между IC50, CC50 и SI и логарифмированными версиями")
plt.savefig(f'./out/corr_mx_{"_".join(target_cols)}')

print('''\nВизуализировать корреляцию между всеми признаками, в том числе созданными на их основе не будем. 
Признаков много, график получится нечитабельный.''')

print('''\nКорреляция между логарифмированными признаками выше чем между оригинальными. 
Далее для обучения моделей регрессии следует использовать логарифмированные признаки. 
А само наличие высокой корреляции позитивно скажется на качестве построенных моделей.''')

print('''\nЕсть признаки высоко коррелированные между собой, то-есть мультиколлинеарность и 
признаки мало коррелированные с таргетами. Решать эту проблему будем используя PCA.''')

print("\nПостроим корреляционную матрицу всех признаков для анализа признаков наиболее коррелирующих с целевыми.")
corr_matrix = df.corr(numeric_only=True)
high_corr = corr_matrix[target_cols].drop(index=target_cols)

print('''\nПокажем наиболее коррелирующие с таргетами признаки. 
Эти признаки имеют наибольшую линейную связь с таргетом. Они приоритетны для моделей.''')
for target in target_cols:
    print(f"\nПризнаки наиболее коррелирующие с {target}.")
    visualise_top_correlated_features(corr_matrix, target, target_cols, 10)

print('''\nПри сценарии ручного отбора признаков для классификации можно 
использовать результаты анализа корреляции фич с целевыми признаками.''')

print("Проверка на дисбаланс классов для задач классификации, важно для выбора метрик (AUC, F1 и пр.)")
df["IC50>median"] = (df["IC50"] > df["IC50"].median()).astype(int)
df["CC50>median"] = (df["CC50"] > df["CC50"].median()).astype(int)
df["SI>median"] = (df["SI"] > df["SI"].median()).astype(int)
df["SI>8"] = (df["SI"] > 8).astype(int)

class_targets = ["IC50>median", "CC50>median", "SI>median", "SI>8"]
print('\nВизуализируем баланс классов для поставленных задач классификации.')
visualise_class_balances(df, class_targets)

print('\nКлассы достаточно сбалансированы для построения классификации.')
print('\nВывод: Классы достаточно сбалансированы для построения классификации.')

print('''\nВыводы:
1. Есть ненужные признаки которые можно удалить. Константа либо очень низкая дисперсия.
2. Распределения целевых переменных сильно скошены и содержат выбросы, поэтому мы их логарифмировали и очистили.
3. Выделили наиболее полезные признаки для построения моделей. В дальнейшем будем использовать PCA.
4. Классы достаточно сбалансированы для построения классификации.
5. Для повышения качества моделей можно провести фича инжениринг.
6. Оставшиеся после чистки признаки могут быть полезны как для задач регрессии так и классификации''')

print("\nEDA завершён. Можно переходить к построению моделей.")
