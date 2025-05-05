import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Настройки визуализации
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Загрузка данных
df: DataFrame = pd.read_csv("../data/kursovik_data.csv")

# Удаление явно неинформативной колонки
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

### 1. ОБЩАЯ ИНФОРМАЦИЯ О ДАННЫХ
print("👁️ Общая структура данных:")
print(df.info())

print("\n📊 Описательная статистика (основные показатели):")
print(df.describe().T)

### 2. АНАЛИЗ ПРОПУСКОВ
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\n🕳️ Признаки с пропущенными значениями:")
print(missing)

# Визуализация пропусков
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Матрица пропусков")
plt.show()

### 3. АНАЛИЗ РАСПРЕДЕЛЕНИЙ ТАРГЕТОВ
target_cols = ['IC50, mM', 'CC50, mM', 'SI']
for col in target_cols:
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"📈 Распределение {col}")
    plt.xlabel(col)
    plt.ylabel("Количество")
    plt.show()

# Пояснение:
# - Проверяем, есть ли смещённость (skew) в распределениях
# - Это влияет на выбор моделей и необходимость лог-преобразований

### 4. КОРРЕЛЯЦИИ МЕЖДУ ТАРГЕТАМИ
plt.figure()
sns.heatmap(df[target_cols].corr(), annot=True, cmap='coolwarm')
plt.title("🔗 Корреляции между IC50, CC50 и SI")
plt.show()

# Пояснение:
# - Высокая корреляция → возможна мультиколлинеарность
# - Полезно для построения регрессионных моделей

### 5. КОРРЕЛЯЦИОННАЯ МАТРИЦА ВСЕХ ПРИЗНАКОВ
corr_matrix = df.corr(numeric_only=True)
high_corr = corr_matrix[target_cols].drop(index=target_cols)
top_corr = high_corr.abs().sort_values(by='SI', ascending=False).head(10)

print("\n🔎 Топ признаков по корреляции с SI:")
print(top_corr)

# Визуализация:
for target in target_cols:
    top_feats = corr_matrix[target].drop(target).abs().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_feats.values, y=top_feats.index)
    plt.title(f"📌 Топ 10 признаков по корреляции с {target}")
    plt.xlabel("Абсолютная корреляция")
    plt.ylabel("Признак")
    plt.show()

# Пояснение:
# - Эти признаки имеют наибольшую линейную связь с таргетом
# - Они приоритетны для моделей

### 6. ВЫБРОСЫ (OUTLIERS)
for col in target_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"📦 Boxplot для {col}")
    plt.show()

# Пояснение:
# - Позволяет обнаружить аномально высокие/низкие значения
# - Такие значения можно логарифмировать или удалить

### 7. ВЗАИМОСВЯЗИ МЕЖДУ ТАРГЕТАМИ
sns.pairplot(df[target_cols])
plt.suptitle("👥 Парные зависимости между IC50, CC50 и SI", y=1.02)
plt.show()

# Пояснение:
# - Полезно для оценки нелинейных взаимосвязей

### 8. РАСПРЕДЕЛЕНИЯ КЛАССОВ ДЛЯ ЗАДАЧ КЛАССИФИКАЦИИ
df["IC50>median"] = (df["IC50, mM"] > df["IC50, mM"].median()).astype(int)
df["CC50>median"] = (df["CC50, mM"] > df["CC50, mM"].median()).astype(int)
df["SI>median"] = (df["SI"] > df["SI"].median()).astype(int)
df["SI>8"] = (df["SI"] > 8).astype(int)

class_targets = ["IC50>median", "CC50>median", "SI>median", "SI>8"]

for col in class_targets:
    sns.countplot(x=df[col])
    plt.title(f"⚖️ Распределение классов для {col}")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.show()

# Пояснение:
# - Проверка на дисбаланс классов, важна для выбора метрик (AUC, F1 и пр.)

### 9. ЛОГАРИФМИРОВАННЫЕ РАСПРЕДЕЛЕНИЯ (если перекошенные)
for col in target_cols:
    plt.figure()
    sns.histplot(np.log1p(df[col]), bins=30, kde=True)
    plt.title(f"📉 Лог-преобразование распределения {col}")
    plt.xlabel(f"log1p({col})")
    plt.show()

# Пояснение:
# - Если распределения тяжёлые или сильно смещены — лог-преобразование улучшает модели

print("\n✅ EDA завершён. Можно переходить к построению моделей.")
