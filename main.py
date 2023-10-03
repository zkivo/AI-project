from classes import ANN
import numpy as np
import matplotlib.pyplot as plt

# ann = ANN(2,1,[2,2],0.1)
# ann.forward(np.array([4.3,-0.22]))
# ann.fit_step(np.array([1,1]),
#              np.array([1]))


# exit()

ds = np.loadtxt("data/redwine-without-head.csv", delimiter=",")

X = ds[::,:-1]
X = X / np.max(X, axis=0)
y = ds[::,-1]

for i,e in list(enumerate(y)):
    if e >= 6:
        y[i] = 1
    else:
        y[i] = 0

ann = ANN(11,1, [5,5],0.2)
# ann.forward(X[3])

for i, x in list(enumerate(X)):
    ann.forward(x)
    ann.fit_step(np.array(x),np.array(y[i]))

fig, ax = plt.subplots()
ax.plot(ann.list_errors)
plt.show()

# print(X)
# print(y)