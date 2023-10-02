from classes import ANN
import numpy as np

ann = ANN(2,1,[2,2],0.1)
ann.forward(np.array([4.3,-0.22]))

exit()

ds = np.loadtxt("data/redwine-without-head.csv", delimiter=",")

X = ds[::,:-1]
X = X / np.max(X, axis=0)
y = ds[::,-1]



print(X)
print(y)