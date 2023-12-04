from BayesClassifier import BayesClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from KNN import KNN
from ANN import ANN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_w = pd.read_csv("data/wine+quality/winequality-white.csv", sep=";")
df_r = pd.read_csv("data/wine+quality/winequality-red.csv", sep=";")
ds_w = df_w.to_numpy()
ds_r = df_r.to_numpy()

X_w = ds_w[:,:-1]
X_w = X_w / np.max(X_w, axis=0)
y_w = ds_w[:, -1]

X_r = ds_r[:,:-1]
X_r = X_r / np.max(X_r, axis=0)
y_r = ds_r[:, -1]

X_w, y_w = RandomOverSampler().fit_resample(X_w,y_w)
X_r, y_r = RandomOverSampler().fit_resample(X_r,y_r)

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_w, y_w, test_size=0.20)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.20)

#---------------------------
#          WHITE
#---------------------------

print("--- White wine ---")

knn_w = KNN(X_train_w, y_train_w, 5)
print("KNN = ", knn_w.score(X_test_w, y_test_w))

bcl_w = BayesClassifier(X_train_w, y_train_w)
print("Bayes classifier score: ", bcl_w.score(X_test_w, y_test_w))

alpha_w = 0.001
epochs_w = 20
ann_w = ANN([11,10,10], alpha_w)
ann_w.fit(X_train_w, y_train_w, epochs_w)
print("ANN = ", ann_w.score(X_test_w, y_test_w))

fig, ax = plt.subplots()
ax.plot(ann_w.list_errors)
ax.set_title("White wine")
ax.set_ylabel("cost")
ax.set_xlabel("epoch")

# #---------------------------
# #          RED
# #---------------------------

print("--- Red wine ---")

knn_r = KNN(X_train_r, y_train_r, 5)
print("KNN = ", knn_r.score(X_test_r, y_test_r))

bcl_r = BayesClassifier(X_train_r, y_train_r)
print("Bayes classifier score: ", bcl_r.score(X_test_r, y_test_r))

alpha_r = 0.001
epochs_r = 20
ann_r = ANN([11,10,10], alpha_r)
ann_r.fit(X_train_r, y_train_r, epochs_r)
print("ANN = ", ann_r.score(X_test_r, y_test_r))

fig, ax = plt.subplots()
ax.plot(ann_r.list_errors)
ax.set_title("Red wine")
ax.set_ylabel("cost")
ax.set_xlabel("epoch")

plt.show()
