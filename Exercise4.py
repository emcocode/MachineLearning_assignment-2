from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import CommonFunctions as cf

# 1. Read CSV
invalues = np.loadtxt('A2_datasets/microchips.csv', delimiter=",")
X = invalues[:, :2]
y = invalues[:, 2]
x1, x2 = invalues[:, 0], invalues[:, 1]

plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5, label='OK')
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='s', alpha=0.5, label='Failed')
plt.legend()


# 2. Gradient descent and decision boundary
one = np.ones(len(X))
Xe_n = np.c_[one, x1, x2, x1**2, x2, x2**2] # Since it is normalized

# Vectorized cost
Beta = [0, 0, 0, 0, 0, 0]
cost = cf.vectorized_cost(Xe_n, Beta, y)
print("Cost: ", cost)

# Vectorized gradient descent
alpha, n = 0.05, 100000
Beta, cost = cf.vectorized_gradient_descent(Beta, Xe_n, y, alpha, n)
print("Beta: ", Beta)
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(cost)
plt.title(f"Alpha = {alpha}, N = {n}")
print(f"Alpha = {alpha} and N = {n} (iterations)")

# Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5, label='OK')
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='s', alpha=0.5, label='Failed')
plt.legend()

# Training errors
errors = cf.getErrors(Xe_n, Beta, y)
plt.title(f"Decision boundary and errors: {errors}")

# Decision boundary
x_min, x_max = (x1.min() - 0.1), (x1.max() + 0.1)
y_min, y_max = (x2.min() - 0.1), (x2.max() + 0.1)
meshSize = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, meshSize), np.arange(y_min, y_max, meshSize))

xymesh = np.c_[xx.ravel(), yy.ravel()]
one = np.ones(len(xymesh))
Xe_n = np.c_[one, xymesh[:, 0], xymesh[:, 1], xymesh[:, 0]**2, xymesh[:, 1], xymesh[:, 1]**2]
Z = cf.sigmoid(np.dot(Xe_n, Beta))>0.5
Z = Z.reshape(xx.shape)
mapcolor = ListedColormap(['red', 'blue'])
plt.contourf(xx, yy, Z, cmap=mapcolor, alpha=0.4)


# 3. Polynomial expressions
# The function "mapFeature" is implemented in the CommonFunctions class.


# 4.
print("Using mapFeatures and a model with a polynomial of degree 5.")
Xe_n = cf.mapFeature(x1, x2, 5)

# Vectorized cost
Beta = np.zeros(Xe_n.shape[1])
cost = cf.vectorized_cost(Xe_n, Beta, y)
print("Cost: ", cost)

# Vectorized gradient descent
alpha, n = 4, 80000
Beta, cost = cf.vectorized_gradient_descent(Beta, Xe_n, y, alpha, n)
print("Beta: ", Beta)
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(cost)
plt.title(f"Alpha = {alpha}, N = {n}")
print(f"Alpha = {alpha} and N = {n} (iterations)")

# Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(X[y==1,0], X[y==1,1], color='blue', marker='o', alpha=0.5, label='OK')
plt.scatter(X[y==0,0], X[y==0,1], color='red', marker='s', alpha=0.5, label='Failed')
plt.legend()

# Training errors
errors = cf.getErrors(Xe_n, Beta, y)
plt.title(f"Decision boundary and errors: {errors}")

# Decision boundary
x_min, x_max = (x1.min() - 0.1), (x1.max() + 0.1)
y_min, y_max = (x2.min() - 0.1), (x2.max() + 0.1)
meshSize = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, meshSize), np.arange(y_min, y_max, meshSize))

xymesh = np.c_[xx.ravel(), yy.ravel()]
# one = np.ones(len(xymesh))
Xe_n = cf.mapFeature(xymesh[:, 0], xymesh[:, 1], 5)
Z = cf.sigmoid(np.dot(Xe_n, Beta))>0.5
Z = Z.reshape(xx.shape)
mapcolor = ListedColormap(['red', 'blue'])
plt.contourf(xx, yy, Z, cmap=mapcolor, alpha=0.4)


plt.show()