from classes import ANN
import numpy as np
import matplotlib.pyplot as plt

ds = np.loadtxt("data/redwine-without-head.csv", delimiter=",")

X = ds[::,:-1]
X = X / np.max(X, axis=0)
y = ds[::,-1]

for i,e in list(enumerate(y)):
    if e >= 6:
        y[i] = 1
    else:
        y[i] = 0

ann = ANN([11,5,5,1], 0.2)

for e in range(5):
    for i, x in list(enumerate(X)):
        ann.fit_step(np.array(x),np.array(y[i]))
    ann.list_errors.append(np.linalg.norm(y[0] - ann.forward(np.array(X[0]))))

fig, ax = plt.subplots()
ax.plot(ann.list_errors)
plt.show()
