import numpy as np
import matplotlib.pyplot as plt
import CommonFunctions as cf # Reusing common functions

# Read CSV
invalues = np.loadtxt('A2_datasets/GPUbenchmark.csv', delimiter=",")
X = invalues[:, :6]
y = invalues[:, 6]
CudaCores = X[:, 0]
BaseClock = X[:, 1]
BoostClock = X[:, 2]
MemorySpeed = X[:, 3]
MemoryConfig = X[:, 4]
MemoryBandwidth = X[:, 5]


# 1. Normalization
CudaCores_n = cf.getNormalized(CudaCores, CudaCores)
BaseClock_n = cf.getNormalized(BaseClock, BaseClock)
BoostClock_n = cf.getNormalized(BoostClock, BoostClock)
MemorySpeed_n = cf.getNormalized(MemorySpeed, MemorySpeed)
MemoryConfig_n = cf.getNormalized(MemoryConfig, MemoryConfig)
MemoryBandwidth_n = cf.getNormalized(MemoryBandwidth, MemoryBandwidth)

# 2. Plot
X_n = np.c_[CudaCores_n, BaseClock_n, BoostClock_n, MemorySpeed_n, MemoryConfig_n, MemoryBandwidth_n]
for i in range(6):
    plt.subplot(2, 3, (i + 1))
    plt.scatter(X_n[:, i], y, s=15)
    plt.xlabel('Normalized feature value')
    plt.ylabel('Benchmark speed')

# 3. Beta through Normal equation
one = np.ones([len(CudaCores), 1])
Xe_n = np.c_[one, X_n]
Beta = np.linalg.inv(Xe_n.T.dot(Xe_n)).dot(Xe_n.T).dot(y)

testGPU, testX_n = [2432, 1607, 1683, 8, 8, 256], [1]
for i in range(len(testGPU)):
    testX_n.append(cf.getNormalized(X[:, i], testGPU[i]))
prediction = np.dot(testX_n, Beta)
print("Predicted value of the testGPU stats is:", prediction) # 110.8

# 4. Cost
print("Cost, using Beta from Normal equation and extended normalized values (Xe_n):", cf.getCost(Xe_n, Beta, y)/len(y)) # 12.39

# 5. Gradient Descent
# a)
a, n = 0.025, 1000
testBeta, costValues = [0, 0, 0, 0, 0, 0, 0], []
for i in range(n):
    testBeta = testBeta - a*Xe_n.T.dot(Xe_n.dot(testBeta) - y)
    costValues.append([cf.getCost(Xe_n, testBeta, y)/len(y)])
print(f"Alpha = {a} and N = {n} gives a cost within 1% of the cost through Normal equation.") # alpha = 0.025 and N = 1000
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Cost J")
plt.plot(costValues)

# b)
prediction_GD = np.dot(testX_n, testBeta)
print("Predicted benchmark result for testGPU using GD:", prediction_GD) # 110.9
plt.show()