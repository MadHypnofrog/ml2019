import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = [], []

f = open("C:\\Users\\User\\Desktop\\ml2019\\java\\resources\\chips.csv", "r")
lines = f.readlines()
for line in lines[1:]:
	tokens = line.split(",")
	X.append([float(tokens[0]), float(tokens[1])])
	if tokens[2] == 'P\n':
		y.append(1)
	else:
		y.append(0)
# funf = open("C:\\Users\\User\\Desktop\\ml2019\\chips.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\chips-polynomial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\chips-radial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\geyser-radial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\geyser-polynomial.txt", "r")
# funf = open("C:\\Users\\Dmitrii\\Documents\\jabbaml\\ml2019\\geyser-linear.txt", "r")
m = int(f.readline())
allLambdas, allBs, coefs = [[]], [], []
for i in range(m):
    lambdas = funf.readline().split()
    lambdas = [float(fl.replace(',', '.')) for fl in lambdas]
    b = lambdas[-2]
    coef = lambdas[-1]
    lambdas = lambdas[:-2]
    allLambdas.append(lambdas)
    allBs.append(b)
    coefs.append(coef)

X, y = np.asarray(X), np.asarray(y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

def kernel(fst, snd):
    return np.array(fst).dot(snd)

def calc(x, X, y, lambdas, b):
    res = 0
    for i in range(len(y)):
        res = res + (1 if y[i] == 1 else -1) * kernel(x, X[i]) * lambdas[i]
    res = res + b
    return res

def calcAll(x, X, y, alLLambdas, allBs, coefs)
    res = 0
    for i in range(len(coefs)):
        res = res + calc(x, X, y, lambdas[i], allBs[i]) * coefs[i]
    return res

for i in range(m):
    xx = np.linspace(xlim[0], xlim[1], 60)
    yy = np.linspace(ylim[0], ylim[1], 60)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.array([calcAll(coord, X, y, allLambdas[:i], allBs[:i], coefs[:i]) for coord in xy]).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=["solid", "dotted", "dashed"])

    plt.show()