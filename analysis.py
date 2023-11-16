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
for i,e in list(enumerate(y)):
    if e >= 6:
        y[i] = 1
    else:
        y[i] = 0
pca = PCA(n_components = 2)

X = pca.fit_transform(X)
sns.scatterplot(data=X)
# fig, ax = plt.subplots()
# ax.scatter(X[:,0], X[:,1])
plt.show()
exit()

x_names = list(df.columns)
x_names.remove("quality")
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

