import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = [], []

f = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\src\\main\\resources\\chips.csv", "r")
lines = f.readlines()
for line in lines[1:]:
	tokens = line.split(",")
	X.append([float(tokens[0]), float(tokens[1])])
	if tokens[2] == 'P\n':
		y.append(1)
	else:
		y.append(0)
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\chips-linear.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\chips-polynomial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\chips-radial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\geyser-radial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\geyser-polynomial.txt", "r")
funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\geyser-linear.txt", "r")
lambdas = funf.readline().split()
lambdas = [float(fl.replace(',', '.')) for fl in lambdas]
b = lambdas[-1]
lambdas = lambdas[:-1]
print(lambdas)

X, y = np.asarray(X), np.asarray(y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

def kernel(fst, snd):
#    return np.array(fst).dot(snd)
#    return pow(np.array(fst).dot(snd), 10)
#   return pow(np.array(fst).dot(snd), 2)
    sub = np.subtract(np.array(fst), snd)
    return math.exp( -0.6 * sub.dot(sub))
#    return math.exp( -0.05 * sub.dot(sub))

def calc(x, X, y, lambdas, b):
    res = 0
    for i in range(len(y)):
        res = res + (1 if y[i] == 1 else -1) * kernel(x, X[i]) * lambdas[i]
    res = res + b
    return res

xx = np.linspace(xlim[0], xlim[1], 60)
yy = np.linspace(ylim[0], ylim[1], 60)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = np.array([calc(coord, X, y, lambdas, b) for coord in xy]).reshape(XX.shape)
print(xy)
print(Z)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=["solid", "dotted", "dashed"])

plt.show()