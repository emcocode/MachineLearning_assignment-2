import numpy as np

# Normalize function
def getNormalized(X, cmp):
    my_x = np.mean(X)
    sigma_x = np.std(X)
    X_n = (cmp - my_x) / sigma_x
    return X_n

# Cost function
def getCost(X, Beta, y):
    cost = (X.dot(Beta) - y).T.dot(X.dot(Beta) - y)
    return cost

# Sigmoid function
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig

# Vectorized cost
def vectorized_cost(X, Beta, y):
    sig = sigmoid(np.dot(X, Beta))
    cost = -1 / len(y) * (np.dot(y.T, np.log(sig)) + np.dot((1 - y).T, np.log(1 - sig)))
    return cost

# Vectoriezed gradient descent
def vectorized_gradient_descent(Beta, X, y, alpha, iterations):
    cost = np.zeros((iterations, 1))
    for i in range(iterations):
        sig = sigmoid(np.dot(X, Beta))
        Beta = Beta - (alpha / len(y)) * np.dot(X.T, sig - y)
        cost[i] = vectorized_cost(X, Beta, y)
    return Beta, cost

# Get errors - from slides
def getErrors(Xe_n, Beta, y):
    Z = np.dot(Xe_n, Beta).reshape(-1,1)
    probability = sigmoid(Z)
    prediction = np.round(probability)
    actual = y.reshape(-1,1)
    errors = np.sum(actual!=prediction)
    return errors

# mapFeature - from slides
def mapFeature(x1, x2, d):
    one = np.ones([len(x1), 1])
    Xe = np.c_[one, x1, x2]
    for i in range(2, d + 1):
        for j in range(0, i + 1):
            Xnew = x1**(i-j)*x2**j
            Xnew = Xnew.reshape(-1,1)
            Xe = np.append(Xe, Xnew, 1)
    return Xe

# mapFeature - from slides
def mapFeatureOnes(x1, x2, d, ones):
    one = np.ones([len(x1), 1])
    if ones:
        Xe = np.c_[one, x1, x2]
    else:
        Xe = np.c_[x1, x2]
    for i in range(2, d + 1):
        for j in range(0, i + 1):
            Xnew = x1**(i-j)*x2**j
            Xnew = Xnew.reshape(-1,1)
            Xe = np.append(Xe, Xnew, 1)
    return Xe
