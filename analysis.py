#
# ref:
# https://chartio.com/learn/charts/what-is-a-scatter-plot/
# 

from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_w = pd.read_csv("data/wine+quality/winequality-white.csv", sep=";")
df_r = pd.read_csv("data/wine+quality/winequality-red.csv", sep=";")
print("white dataset has null values: ", df_w.isnull().values.any())
print("red dataset has null values: ", df_r.isnull().values.any())

X_w = df_w.iloc[:, 0:11].values
y_w = df_w.iloc[:, 11].values
X_r = df_r.iloc[:, 0:11].values
y_r = df_r.iloc[:, 11].values

X_w, y_w = RandomOverSampler().fit_resample(X_w,y_w)

unique_w, frequency_w = np.unique(y_w, return_counts = True)
unique_r, frequency_r = np.unique(y_r, return_counts = True)

#quality frequencies
fig, ax = plt.subplots(2,1)
ax[0].bar(unique_w, frequency_w)
ax[0].set_title("White wine")
ax[0].set_xlabel("Quality")
ax[0].set_ylabel("Frequency")
ax[1].bar(unique_r, frequency_r)
ax[1].set_title("Red wine")
ax[1].set_xlabel("Quality")
ax[1].set_ylabel("Frequency")

# histograms of features
x_names = list(df_w.columns)
x_names.remove("quality")
fig, ax = plt.subplots(2, 6)
fig.suptitle('White wine')
for i in range(2):
    ax[i][0].set_ylabel("Frequency")
    for j in range(6):
        if i == 1 and j == 5: break
        ax[i][j].hist(df_w[x_names[int(6*i+j)]], bins=40)
        ax[i][j].set_xlabel(x_names[6*i+j])
fig, ax = plt.subplots(2, 6)
fig.suptitle('Red wine')
for i in range(2):
    ax[i][0].set_ylabel("Frequency")
    for j in range(6):
        if i == 1 and j == 5: break
        ax[i][j].hist(df_r[x_names[int(6*i+j)]], bins=40)
        ax[i][j].set_xlabel(x_names[6*i+j])


# best correlation in white
#  0.838966 - residual sugar - density
# -0.780138 - alcohol - density
# best correlation in red
#  0.671703 - fixed acidity - citric acid
# -0.682978 - fixed acidity - pH
corr_matrix_w = df_w.corr()
corr_matrix_r = df_r.corr()
with open("data/corr_table_white.txt", "w") as f:
  print(corr_matrix_w.to_markdown(), file=f)
with open("data/corr_table_red.txt", "w") as f:
  print(corr_matrix_r.to_markdown(), file=f)
corr_matrix_w = corr_matrix_w.replace(1,0)
corr_matrix_r = corr_matrix_r.replace(1,0)
print("max and min corr-white: ", corr_matrix_w.max(axis=None), corr_matrix_w.min(axis=None))
print("max and min corr-red: ", corr_matrix_r.max(axis=None), corr_matrix_r.min(axis=None))
sns.scatterplot(data=df_w, x=df_w["density"], y=df_w["residual sugar"], hue="quality").figure.savefig("data/figures/density-residual+sugar.png") 
sns.scatterplot(data=df_w, x=df_w["density"], y=df_w["alcohol"], hue="quality").figure.savefig("data/figures/density-alcohol.png") 


sns.boxplot(df[["fixed acidity","volatile acidity","citric acid", \
                "residual sugar","chlorides","density","pH", \
                "sulphates","alcohol"]])
sns.boxplot(df[["free sulfur dioxide","total sulfur dioxide"]])
sns.pairplot(df, hue="quality")
plt.show()

