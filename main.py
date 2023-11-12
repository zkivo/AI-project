from classes import ANN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


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

ds = np.loadtxt("data/redwine-without-head.csv", delimiter=",")

shape = np.shape(ds)

X = ds[:int(shape[0] * 0.8),:-1]
X = X / np.max(X, axis=0)
y = ds[:int(shape[0] * 0.8),-1]

X_test = ds[int(shape[0] * 0.8):,:-1]
y_test = ds[int(shape[0] * 0.8):, -1]

for i,e in list(enumerate(y)):
    if e >= 6:
        y[i] = 1
    else:
        y[i] = 0

for i,e in list(enumerate(y_test)):
    if e >= 6:
        y_test[i] = 1
    else:
        y_test[i] = 0

# iris = load_iris()
# X = np.array(iris.data)
# y = np.array(iris.target)

# X_test = X[int(np.shape(X)[0] * 0.8):,:]
# y_test = y[int(np.shape(y)[0] * 0.8):]

ann = ANN([11,16,16,2], 0.05)

first_score = get_score()

# epochs
for _ in range(50):
    # train the whole dataset
    # X,y = shuffle(X,y)
    for i, x in list(enumerate(X)):
        ann.fit_step(np.array(x),np.array(y[i]))
    err = []
    for i, x in list(enumerate(X)):
        err.append(np.sum((y[i] - ann.forward(x))**2))
    err = np.average(np.array(err))
    ann.list_errors.append(err)

print("Score = ", get_score(), " ( starting at", first_score, ")")

fig, ax = plt.subplots()
ax.plot(ann.list_errors)
plt.show()
