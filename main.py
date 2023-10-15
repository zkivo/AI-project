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

# ann = ANN(11,1, [5,5],0.1)
ann = ANN([11,5,5,1], 0.2)
# ann.forward(np.array(X[0]))
# ann.fit_step(X[1],y[1])
# ann.forward(np.array(X[2]))
# ann.fit_step(X[3],y[3])
#exit()

for e in range(50):
    for i, x in list(enumerate(X)):
        ann.fit_step(np.array(x),np.array(y[i]))
    # ann.list_errors.append(np.linalg.norm(y[0] - ann.forward(np.array(X[0]))))
    ann.list_errors.append(np.linalg.norm(y[0] - ann.forward(np.array(X[0])))**2)
    #print(ann.w[2][1][0])

fig, ax = plt.subplots()
# print(ann.list_errors)
ax.plot(ann.list_errors)
plt.show()

# print(X)
# print(y)