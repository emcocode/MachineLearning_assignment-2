import numpy as np
import matplotlib.pyplot as plt
import CommonFunctions as cf

# 1. Read CSV
invalues = np.loadtxt('A2_datasets/banknote_authentication.csv', delimiter=",")
sizes = [500, 750, 1000, 1200]
reps = 5
totTrainErrors, totTestErrors, totTrainAcc, totTestAcc = [], [], [], []
for size in sizes:
    print("Training size: ", size)
    for rep in range(reps):
        print(f"\tIteration: {rep + 1}")
        np.random.shuffle(invalues)
        Xtot = invalues[:, :4]
        Ytot = invalues[:, 4]
        # print(len(Xtot))

        # 2. Division of train/test data
        trainX, trainY, testX, testY = Xtot[:size], Ytot[:size], Xtot[size:], Ytot[size:]

        # 3. Normalization
        X1_n = cf.getNormalized(trainX[:, 0], trainX[:, 0])
        X2_n = cf.getNormalized(trainX[:, 1], trainX[:, 1])
        X3_n = cf.getNormalized(trainX[:, 2], trainX[:, 2])
        X4_n = cf.getNormalized(trainX[:, 3], trainX[:, 3])
        test_X1_n = cf.getNormalized(testX[:, 0], testX[:, 0])
        test_X2_n = cf.getNormalized(testX[:, 1], testX[:, 1])
        test_X3_n = cf.getNormalized(testX[:, 2], testX[:, 2])
        test_X4_n = cf.getNormalized(testX[:, 3], testX[:, 3])

        # Extend
        one = np.ones(len(X1_n))
        Xe_n = np.c_[one, X1_n, X2_n, X3_n, X4_n]
        testOne = np.ones(len(test_X1_n))
        test_Xe_n = np.c_[testOne, test_X1_n, test_X2_n, test_X3_n, test_X4_n]

        # Vectorized cost
        Beta = [0, 0, 0, 0, 0]
        cost = cf.vectorized_cost(Xe_n, Beta, trainY)

        # Vectorized gradient descent
        alpha, n = 0.1, 10000
        Beta, cost = cf.vectorized_gradient_descent(Beta, Xe_n, trainY, alpha, n)
        # print(f"Alpha = {alpha} and {iterations} iterations") # Not necessary to print every time, but is presented in the end.
        if (rep == 1 and size == 1000): # Only plot once, first rep with training size 1000
            plt.title(f"Alpha = {alpha} and {n} iterations")
            plt.plot(cost)


        # 4. Accuracy
        trainingErrors = cf.getErrors(Xe_n, Beta, trainY)
        trainingAccuracy = 100-((trainingErrors/len(trainY))*100)
        # print(f"\tThere are {trainingErrors} training errors and the training accuracy is {trainingAccuracy}%") # Print this to see in "real time"
        totTrainErrors.append(trainingErrors)
        totTrainAcc.append(trainingAccuracy)


        # 5. Test accuracy
        testErrors = cf.getErrors(test_Xe_n, Beta, testY)
        testAccuracy = 100-((testErrors/len(testY))*100)
        # print(f"\tThere are {testErrors} test errors and the test accuracy is {testAccuracy}%") # Print this to see in "real time"
        totTestErrors.append(testErrors)
        totTestAcc.append(testAccuracy)

# 6. 
print(f"\nAlpha = {alpha} and N = {n} (iterations)") # From part 3, but only want to print once at the end!
print(f"Training size\t\tIteration\t\tTraining errors\t\tTraining accuracy\t\tTest errors\t\tTest accuracy:")
for i in range(len(totTrainErrors)):
    print(f"{sizes[i//reps]}\t\t{((i%reps) + 1):>15}\t\t{totTrainErrors[i]:>15}\t\t{round(totTrainAcc[i], 2):>15} %\t\t{totTestErrors[i]:>15}\t\t{round(totTestAcc[i], 2):>15} %")

plt.show()

# We are now repeating the process "rep" times (5), for each training size [500, 750, 1000, 1200].
# Both of these variables can be changed to see fewer/more repetitions or other training sizes.
# Since they give different results, the results are presented in a sort of a table.

# There is a variance in the results, the training accuracy is quite stable at roughly 98-99% (depending on reps and sizes).
# The test accuracy varies more, which is to be expected - especially when the test portion of the data is too small (say size 1200 for example).
# In these cases the accuracy varies between roughly 96-100%.

# In cases with lower training size, our model is less good - but the test accuracy is more stable. When we have higher training size, our model
# gets better but our test results vary more.

# I would say that in cases with enough data (both training and tests), the results are qualitatively (roughly) the same.
# The variation is dependent on the training/test size. The difference betweem training and testing is expected.
