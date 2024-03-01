import numpy as np
import matplotlib.pyplot as plt

# Read CSV
invalues = np.loadtxt('A2_datasets/secret_polynomial.csv', delimiter=",")
avgTrain, avgTest = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
reps = 10
size = 270 # How many of the 300 data points should be training data (rest will be test data)
for i in range(reps):
    np.random.shuffle(invalues)
    trainingSet = invalues[:size]
    testSet = invalues[size:]
    X = trainingSet[:, 0]
    Y = trainingSet[:, 1]
    testX = testSet[:, 0]
    testY = testSet[:, 1]

    # Polynomial regression
    one = np.ones([len(trainingSet), 1])
    oneTest = np.ones([len(testSet), 1])
    Xe = np.c_[one]
    for d in range(1, 7):
        Xe = np.c_[Xe, X**d]
        Beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(Y)
        y = np.dot(Xe, Beta)
        plt.subplot(2, 3, (d))
        if (i == 1): # Only plot once
            plt.scatter(X, Y, c='b', s=15, label='Training set')
            plt.scatter(testX, testY, c='r', s=15, label='Test set')
            new_x, new_y = zip(*sorted(zip(X, y)))
            plt.plot(new_x, new_y, c='g', label='Model')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.title(f"Degree {d}")
        
        # Get errors
        mseTrain = np.mean((Y - y)**2)
        avgTrain[d-1] += mseTrain/reps
        testXe = np.c_[oneTest, testX.reshape(-1,1)**np.arange(1,d+1)]
        test_pred = np.dot(testXe, Beta)
        mseTest = np.mean((test_pred - testY)**2)
        avgTest[d-1] += mseTest/reps

print("Training error:")
for d in range(len(avgTrain)):
    print(f"\tdegree {d+1}:", avgTrain[d])
print("Test error:")
for d in range(len(avgTest)):
    print(f"\tdegree {d+1}:", avgTest[d])
plt.show()


# In this program we run the same X times to repeat the process with shuffled data set.
# It gives different result depending on the size distribution of the test/training sets. This can easily be changed by
# changing the size variable. So can the number of repetitions, by changing the reps variable.

# I remember in the lecture it was mentioned that the smallest degree possible (that still suits the function) should
# be the best fit. I have tried and tried but as far as I can produce, it is a very even match between degrees 3-6 (or 4-6). 
# It differs from time to time and it is highly dependant on the test size!

# If I have to name one, I would say that degree 4 has the best fit. Using 10 reps and 90/10 (%) training/test distribution,
# degree 4 is the one which most consistently is the lowest (barely).

# Note: My costs are extremely high, I remain unsure as to wheather I have completed this task in the correct manner.