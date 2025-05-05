import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

data: DataFrame = pd.read_csv("data/kursovik_data.csv")

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


# удалим колонки с корреляцией с таргетными менее 0.02 и между собой более 0.8

def filter_correlated_columns(df, target_cols, corr_threshold=0.02, high_corr=0.8):
    """
    Filters columns based on correlation rules:
    1. Remove columns with absolute correlation < 0.02 with any target column
    2. Remove one from correlated non-target pairs (correlation > 0.8), keeping
       the column with higher absolute correlation to targets
    """
    # Keep original dataframe intact
    df = df.copy()

    # 1. Remove columns with low correlation to targets
    # Calculate maximum absolute correlation with any target
    target_correlations = df.corr()[target_cols].abs()
    max_target_corr = target_correlations.max(axis=1)

    # Identify columns to keep (>= corr_threshold OR target columns)
    columns_to_keep = max_target_corr[(max_target_corr >= corr_threshold) |
                                      (max_target_corr.index.isin(target_cols))].index
    df_filtered = df[columns_to_keep]

    # 2. Remove correlated non-target columns
    non_target_cols = [col for col in df_filtered.columns if col not in target_cols]

    # Calculate correlation matrix for non-target columns
    corr_matrix = df_filtered[non_target_cols].corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)  # Ignore self-correlation

    # Find pairs of highly correlated columns
    to_remove = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > high_corr:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]

                # Compare correlations with targets
                col1_corr = max_target_corr[col1]
                col2_corr = max_target_corr[col2]

                # Remove the column with lower target correlation
                if col1_corr < col2_corr and col1 not in target_cols:
                    to_remove.add(col1)
                elif col2_corr < col1_corr and col2 not in target_cols:
                    to_remove.add(col2)
                else:
                    # If equal, remove the first one alphabetically
                    to_remove.add(sorted([col1, col2])[0])

    return df_filtered.drop(columns=to_remove)


targets = ['SI', 'CC50, mM', 'IC50, mM']

data = filter_correlated_columns(data, targets)
# строим и визуализируем матрицу корреляций

# Compute correlation matrix
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
