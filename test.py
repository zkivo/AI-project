# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

# Neural network, Nearest neighbors, Baesian belief network
# k-means

# 3 neuroni ha la precisione di 40, 1000 51%
# 1000x1000 59%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import os

print(os.getcwd())
df = pd.read_csv(os.getcwd() + "\winequality-red-with-good.csv")
#df = df.map(hash)
X = df.iloc[0:,:11]
y = df["good"]

#pd.plotting.scatter_matrix(X)

# import plotly.express as px
# #df = px.data.iris()
# fig = px.scatter_matrix(X, color="quality")
# fig.show()

#df.plot.scatter(x="odor",y="bruises")


#print(X.head())
#print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=1)
clf.fit(X_train,y_train)

score = clf.score(X_test, y_test)
print(score)

#plt.show()
