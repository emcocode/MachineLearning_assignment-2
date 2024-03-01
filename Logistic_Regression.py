from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import CommonFunctions as cf

# Read CSV
invalues = np.loadtxt('A2_datasets/admission.csv', delimiter=",")
X = invalues[:, :2]
y = invalues[:, 2]

# 1. Normalization
X1_n = cf.getNormalized(X[:, 0], X[:, 0])
X2_n = cf.getNormalized(X[:, 1], X[:, 1])

#Plot
mapcolor = ListedColormap(['red', 'blue'])
plt.scatter(X1_n, X2_n, c=y, cmap=mapcolor, s=15)
# plt.show()

# 2. Sigmoid
matrix = np.array([[0, 1], [2, 3]])
newMatrix = cf.sigmoid(matrix)
print(newMatrix)

# 3. Extend
one = np.ones(len(X))
Xe = np.c_[one, X]
Xe_n = np.c_[one, X1_n, X2_n]
# print(Xe)

# 4. Vectorized cost
Beta = [0, 0, 0]
cost = cf.vectorized_cost(Xe_n, Beta, y)
print("cost: ", cost)

# 5. Vectorized gradient descent
alpha, iterations = 0.1, 5000
Beta, cost = cf.vectorized_gradient_descent(Beta, Xe_n, y, alpha, iterations)
print(Beta)
# print(cost)

# 6. Increase number of iterations

x_values = np.array([np.min(X1_n), np.max(X1_n)])
y_values = -(Beta[0] + Beta[1] * x_values) / Beta[2]
plt.plot(x_values, y_values, 'k-')


# 7. Prediction
student = np.array([45,85])
student_n = cf.getNormalized(student, student)
student_ne = np.c_[1, student[0], student[1]]
prob = cf.sigmoid(np.dot(student_ne, Beta) )
print(f"The probability that a student with scores {student[0]}, {student[1]} is admitted is {prob[0]}")

z = np.dot(Xe_n, Beta).reshape(-1,1) # Compute X*beta
p = cf.sigmoid(z) # Probabilities in range [0,1]
pp = np.round(p) # prediction
yy = y.reshape(-1,1) # actual
print("Training errors: ",(np.sum(yy!=pp)))



plt.show()