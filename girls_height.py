import numpy as np
import matplotlib.pyplot as plt

# Read CSV
# X1,X2,X3 = np.genfromtxt ('lecture4_datasets/stat_females.csv', delimiter=" ", unpack=True)
invalues = np.loadtxt('lecture4_datasets/stat_females.csv', delimiter="\t")
girl, mom, dad = invalues[:, 0], invalues[:, 1], invalues[:, 2]

# 1. Scatterplot
fig, ax = plt.subplots(1, 2)

# Mom/girl
ax[0].scatter(mom, girl, s=15)
ax[0].set_xlabel('Mom height (Inches)')
ax[0].set_ylabel('Girl heigth (Inches)')

# Dad/girl
ax[1].scatter(dad, girl, s=15)
ax[1].set_xlabel('Dad height (Inches)')
ax[1].set_ylabel('Girl heigth (Inches)')
# plt.show()


# 2. Compute extended matrix
one = np.ones([len(mom), 1])
Xe = np.c_[one, mom, dad]
XXe = np.array(Xe)
# print(Xe)


# 3. Normal Equation
Beta = np.linalg.inv(XXe.T.dot(XXe)).dot(XXe.T).dot(girl)
predictions = np.round(XXe.dot(Beta), 3) # Round numbers or not?
print(np.c_[XXe, predictions])


# SHOULD NUMBERS BE ROUNDED LIKE IN LECTURE, OR NOT?

# 4. Feature Normalization
fig, ax = plt.subplots(1, 2)
# My's and sigma's
my_mom = np.round(np.mean(XXe[:, 1]), 2)
my_dad = np.round(np.mean(XXe[:, 2]), 2)
sigma_mom = np.round(np.std(XXe[:, 1]), 2)
sigma_dad = np.round(np.std(XXe[:, 2]), 2)
print("My's and Sigma's:", my_mom, my_dad, sigma_mom, sigma_dad)

# Normalized X -> (Xn)
Xn_mom = (XXe[:, 1] - my_mom) / sigma_mom
Xn_dad = (XXe[:, 2] - my_dad) / sigma_dad

# Mom/girl
ax[0].scatter(Xn_mom, girl, s=15)
ax[0].set_xlabel('Mom height (normalized)')
ax[0].set_ylabel('Girl heigth (Inches)')

# Dad/girl
ax[1].scatter(Xn_dad, girl, s=15)
ax[1].set_xlabel('Dad height (normalized)')
ax[1].set_ylabel('Girl heigth (Inches)')
# plt.show()


# 5. Extended matris with Normal Equation
Xen = np.c_[one, Xn_mom, Xn_dad] # Extend normalized lengths
BetaN = np.linalg.inv(Xen.T.dot(Xen)).dot(Xen.T).dot(girl) # Normalized beta
print("Normalized beta:", BetaN)
predictionsN = np.round(Xen.dot(BetaN), 3)

Xn = np.c_[Xn_mom, Xn_dad]
print("Xen: ", Xen)
print("Xn: ", Xn)
# print(np.c_[predictionsN, Xn_mom, Xn_dad])


# 6. Cost function
def getCost(X, Beta, y):
    cost = (X.dot(Beta) - y).T.dot(X.dot(Beta) - y)/len(y)
    return cost
# cost = (Xen.dot(BetaN) - girl).T.dot(Xen.dot(BetaN) - girl)/len(girl)
print("Cost from Beta from Normal equation:", getCost(Xen, BetaN, girl))


# 7. Gradient descent
# a)
def getGradientDescent(a, n, Beta):
    costValues = []
    for i in range(n):
        Beta = Beta - a*Xen.T.dot(Xen.dot(Beta) - girl)
        costValues.append([getCost(Xen, Beta, girl)])
    return costValues, getCost(Xen, Beta, girl)
gd, gdCost = getGradientDescent(0.0001, 1000, Beta)
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Cost J")
plt.plot(gd)
plt.show()

