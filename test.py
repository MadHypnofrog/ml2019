import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
X, y = [], []

f = open("C:\\Users\\User\\Desktop\\ml2019\\src\\main\\resources\\chips.csv", "r")
lines = f.readlines()
for line in lines[1:]:
	tokens = line.split(",")
	X.append([float(tokens[0]), float(tokens[1])])
	if tokens[2] == 'P\n':
		y.append(1)
	else:
		y.append(0)
funf = open("C:\\Users\\User\\Desktop\\ml2019\\chips-linear.txt", "r")
coefs = map(float, f.readline().split())
print(coefs)
# fit the model, don't regularize for illustration purposes

X, y = np.asarray(X), np.asarray(y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
print(xy)
print(Z)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors

plt.show()