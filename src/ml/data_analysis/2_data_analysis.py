import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

data: DataFrame = pd.read_csv("../data/kursovik_data.csv")

print("Columns list:", data.columns.tolist())
print(data.head())
print(data.info())
print("Колонки с пропусками")
# Get columns with at least one NaN
nan_columns = data.columns[data.isna().any()]
print("Columns with NaNs:", nan_columns.tolist())
print("Total Nans in Columns with NaNs:", data[nan_columns].isna().sum())
# есть пропуски но очень мало, заполним средним
data = data.fillna(data.median())

print("Columns list:", data.columns.tolist())
# ретест, колонок с пропусками больше нет
nan_columns = data.columns[data.isna().any()]
print("Columns with NaNs:", nan_columns.tolist())
print("Total Nans in Columns with NaNs:", data[nan_columns].isna().sum())


# заметно много колонок где значение равно нулю, видимо это тоже можно воспринимать как Nan, проведём анализ в каких колонках сколько и какой процент нулей.
# Select numeric columns only (exclude strings, etc.)
def get_cols_with_zeros(data):
    numeric_cols = data.select_dtypes(include='number').columns

    # Count zeros in numeric columns
    zero_counts = (data[numeric_cols] == 0).sum()
    # Compute percentage of zeros in each column
    zero_percent = (zero_counts / len(data)) * 100

    # Create a summary DataFrame
    zero_summary = pd.DataFrame({
        'Zero Count': zero_counts,
        'Zero Percent (%)': zero_percent.round(2)  # Round to 2 decimal places
    })

    # Sort by highest zero count (optional)
    return zero_summary.sort_values('Zero Count', ascending=False)


print(get_cols_with_zeros(data))


#  В таргетных колонках IC50, CC50 и SI нулей нет, это хорошо. От колонок где нулей больше 30% избавимся, в оставшихся нули заменим средним.
def remove_zeros_and_fill_remained_with_mean(df, threshold=30):
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns

    # Calculate percentage of zeros in numeric columns
    zero_pct = (df[numeric_cols] == 0).mean() * 100

    # Identify columns to drop (zero percentage > threshold)
    cols_to_drop = zero_pct[zero_pct > threshold].index
    df_filtered = df.drop(columns=cols_to_drop)

    # Replace zeros with column mean in remaining numeric columns
    remaining_numeric = df_filtered.select_dtypes(include='number').columns

    for col in remaining_numeric:
        col_mean = df_filtered[col].mean()  # Compute column mean
        df_filtered[col] = df_filtered[col].replace(0, col_mean)  # Replace 0s

    return df_filtered


data = remove_zeros_and_fill_remained_with_mean(data, threshold=30)

print(get_cols_with_zeros(data))


targets = ['SI', 'CC50, mM', 'IC50, mM']

# строим и визуализируем матрицу корреляций
corr = data.corr()

# Set up the figure with adjustable size
plt.figure(figsize=(20, 18))  # <-- Adjust width/height as needed

# Customize heatmap
sns.set(font_scale=0.8)  # <-- Control label size
heatmap = sns.heatmap(
    corr,
    annot=False,  # Hide annotations for large matrices
    cmap='coolwarm',  # Color map
    linewidths=0.5,  # Gap between cells
    vmin=-1, vmax=1,
    square=True
)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.title("Correlation Matrix (Seaborn)")
plt.tight_layout()  # Prevent label cutoff
plt.show()


# Настраиваем размер графика


# Строим boxplot для всех колонок
import pandas as pd
import matplotlib.pyplot as plt
import os

# Создаем папку для сохранения графиков (если не существует)
output_dir = "boxplots"
os.makedirs(output_dir, exist_ok=True)

# Фильтруем только числовые колонки
numeric_cols = data.select_dtypes(include=['number']).columns

# Настройки визуализации
# plt.style.use('seaborn')
BOXPLOT_PARAMS = {
    'vert': False,
    'patch_artist': True,
    'boxprops': dict(facecolor='cyan', color='navy'),
    'flierprops': dict(marker='o', markersize=5, markerfacecolor='red')
}

# Генерируем отдельный график для каждой колонки
for i, col in enumerate(numeric_cols, 1):
    # Создаем новую фигуру для каждой итерации
    plt.figure(figsize=(10, 3))

    # Строим boxplot
    plt.boxplot(data[col].dropna(), **BOXPLOT_PARAMS)

    # Настройки оформления
    plt.title(f"Boxplot для колонки: {col}\n", fontsize=12, pad=20)
    plt.xlabel("Значения", fontsize=10)
    plt.yticks([])  # Убираем метки на оси Y
    plt.grid(axis='x', alpha=0.5)

    # Сохраняем в файл
    filename = f"{output_dir}/boxplot_{col.replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()  # Важно: закрываем фигуру для экономии памяти

print(f"Создано {len(numeric_cols)} графиков в папке '{output_dir}'")



# Создаем холст с тремя областями для графиков
plt.figure(figsize=(12, 4))

# Выбираем  
columns =['SI','IC50, mM','CC50, mM']

# Строим boxplot для каждой колонки
for i, col in enumerate(columns, 1):
    plt.subplot(1, 3, i)  # 1 строка, 3 столбца, позиция i
    plt.boxplot(data[col], patch_artist=True)
    plt.title(f'Распределение {col}')
    plt.ylabel('Значения')

    plt.grid(alpha=0.3)

# Настраиваем общий заголовок и отступы
plt.suptitle('Сравнение распределений признаков', y=1.05)
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_SI_CC50_IC50.png")
# plt.show()