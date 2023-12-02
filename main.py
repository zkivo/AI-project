from BayesClassifier import BayesClassifier
from KNN import KNN
from ANN import ANN
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# MLP from sklearn has the same score of my ANN
# PCA does not seem to improve the score
# using two classes bad and good seams to improve about some 0.07 score points

# Bayes classifier improves if the classes are reduces to bad and good
# about 20 precentage points, from .40 to .60


def get_score():
    rows = np.shape(X_test)[0]
    succ = 0
    for i in range(int(rows)):
        out = ann.forward(X_test[i])
        # print(X[i],out)
        index = np.argmax(out)
        if y_test[i] == index:
            succ += 1
    return succ / rows

def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

pca = PCA(n_components = 2)
#ds  = np.loadtxt("data/redwine-without-head.csv", delimiter=",")

df = pd.read_csv("data/wine+quality/winequality-white.csv", sep=";")
# df = df.drop('density', axis=1)
ds = df.to_numpy()

shape = np.shape(ds)

X_train = ds[:int(shape[0] * 0.8),:-1]
# X_train = pca.fit_transform(X_train)
X_train = X_train / np.max(X_train, axis=0)
y_train = ds[:int(shape[0] * 0.8),-1]

X_test = ds[int(shape[0] * 0.8):,:-1]
# X_test = pca.fit_transform(X_test)
X_test = X_test / np.max(X_test, axis=0)
y_test = ds[int(shape[0] * 0.8):, -1]

# drop citric acid feature because does not approximate a normal curve
# does not improve significantly
# X_train = np.delete(X_train, 2, 1)
# X_test  = np.delete(X_test , 2, 1)

# for i,e in list(enumerate(y_train)):
#     if e >= 6:
#         y_train[i] = 1
#     else:
#         y_train[i] = 0

# for i,e in list(enumerate(y_test)):
#     if e >= 6:
#         y_test[i] = 1
#     else:
#         y_test[i] = 0

# knn = KNN(X_train, y_train, 5)
# print(knn.score(X_test, y_test))

# bcl = BayesClassifier(X_train, y_train)

# print("Bayes classifier score: ", bcl.score(X_test, y_test))

# exit()

# iris = load_iris()
# X = np.array(iris.data)
# y = np.array(iris.target)

# X_test = X[int(np.shape(X)[0] * 0.8):,:]
# y_test = y[int(np.shape(y)[0] * 0.8):]

alpha = 0.001
epochs = 15

ann = ANN([11,10,10], alpha)
mlp = MLPClassifier((10), alpha=alpha, activation='logistic', \
                     max_iter=epochs, shuffle=False, solver='sgd')

mlp.fit(X_train,y_train)
print("mlp: ", mlp.score(X_test,y_test))


first_score = get_score()

ann.fit(X_train, y_train, epochs)

# for _ in range(20):
#     # train the whole dataset
#     # X,y = shuffle(X,y)
#     for i, x in list(enumerate(X)):
#         ann.fit_step(np.array(x),np.array(y[i]))
#     err = []
#     for i, x in list(enumerate(X)):
#         err.append(np.sum((y[i] - ann.forward(x))**2))
#     err = np.average(np.array(err))
#     ann.list_errors.append(err)

print("Score = ", get_score(), " ( starting at", first_score, ")")

fig, ax = plt.subplots()
ax.plot(ann.list_errors)
plt.show()
