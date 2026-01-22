import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("future.no_silent_downcasting", True)

df = pd.read_excel("KendallTau.xlsx", engine='openpyxl')
# pal = sns.diverging_palette(270, 0.5, s=75, l=45, center='light', as_cmap=True)

## test individual pairs
# wordrank1 = df['S_softness']
# wordrank2 = df['S_hardness']
# tau, pvalue = Kendall(wordrank1, wordrank2)
# print("tau:", tau, "\npvalue:", pvalue)


def Kendall(x1, x2):
    for i in range(10):
        x2.replace(x1[i], i + 1, inplace=True)  # replace based on ranks of x1
        x1.replace(x1[i], i + 1, inplace=True)  # 1...10
    res = stats.kendalltau(x1, x2)
    return res.statistic, res.pvalue


def calculate_kendall_tau_matrix(dataframe):
    """Calculates the Kendall's tau correlation matrix for a pandas DataFrame."""
    # Get all column combinations
    cols = dataframe.columns
    num_cols = len(cols)
    tau_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    # Calculate Kendall's tau for all unique pairs
    for i in range(num_cols):
        for j in range(i, num_cols):
            col1 = cols[i]
            col2 = cols[j]
            tau, p_value = Kendall(dataframe[col1].copy(), dataframe[col2].copy())  # use copies so don't change original df
            tau_matrix.loc[col1, col2] = tau
            tau_matrix.loc[col2, col1] = tau  # make symmetric
            p_matrix.loc[col1, col2] = p_value
            p_matrix.loc[col2, col1] = p_value  # make symmetric

    return tau_matrix, p_matrix


tau_matrix, p_matrix = calculate_kendall_tau_matrix(df)

plt.figure(figsize=(8, 6), dpi=300)
ax = sns.heatmap(tau_matrix, annot=False, cmap='coolwarm', fmt=".2f", square=True, vmin=-1, vmax=1, annot_kws={"fontsize": 4})
# heatmap border
ax.axvline(0, color='black', lw=2, zorder=100)
ax.axvline(26, color='black', lw=2, zorder=100)
ax.axhline(0, color='black', lw=2, zorder=100)
ax.axhline(26, color='black', lw=2, zorder=100)
# white space seps
ax.axvline(4, color='white', lw=5)
ax.axhline(4, color='white', lw=5)
ax.axvline(10, color='white', lw=5)
ax.axhline(10, color='white', lw=5)
ax.axvline(14, color='white', lw=5)
ax.axhline(14, color='white', lw=5)
plt.title('Tau Heatmap')
plt.savefig('TauHeatmapAll.png')

plt.figure(figsize=(8, 6), dpi=300)
ax = sns.heatmap(p_matrix, annot=False, cmap='coolwarm', fmt=".2f", square=True, vmin=0, vmax=1, annot_kws={"fontsize": 4})  #'coolwarm'
# heatmap border
ax.axvline(0, color='black', lw=2, zorder=100)
ax.axvline(26, color='black', lw=2, zorder=100)
ax.axhline(0, color='black', lw=2, zorder=100)
ax.axhline(26, color='black', lw=2, zorder=100)
# white space seps
ax.axvline(4, color='white', lw=5)
ax.axhline(4, color='white', lw=5)
ax.axvline(10, color='white', lw=5)
ax.axhline(10, color='white', lw=5)
ax.axvline(14, color='white', lw=5)
ax.axhline(14, color='white', lw=5)
plt.title('P-Value')
plt.savefig('PvalHeatmapAll.png')

plt.show()

