# ref:
# https://chartio.com/learn/charts/what-is-a-scatter-plot/
# 

from sklearn.decomposition import PCA
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/winequality-red.csv")
print("has null values: ", df.isnull().values.any())

X = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values

x_names = list(df.columns)
x_names.remove("quality")

fig, ax = plt.subplots(2, 6)
for i in range(2):
    for j in range(6):
        if i == 1 and j == 5: break
        ax[i][j].hist(df[x_names[int(6*i+j)]], bins=40)
        ax[i][j].set_xlabel(x_names[6*i+j])
        ax[i][j].set_ylabel("Frequency")
plt.show()
combs = itertools.combinations(x_names, 2)

# corr_matrix = df.corr()

# for c in combs:
#     sns.scatterplot(data=df, x=df[c[0]], y=df[c[1]], hue="quality").figure.savefig("data/figures/"+ c[0] + "-" + c[1] + ".png") 
#     plt.close()

# with open("corr_table.txt", "w") as f:
#   print(corr_matrix.to_markdown(), file=f)


# sns.boxplot(df[["fixed acidity","volatile acidity","citric acid", \
#                 "residual sugar","chlorides","density","pH", \
#                 "sulphates","alcohol"]])
# sns.boxplot(df[["free sulfur dioxide","total sulfur dioxide"]])
# sns.pairplot(df, hue="quality")
# plt.show()

