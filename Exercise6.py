import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Read CSV
invalues = np.loadtxt('A2_datasets/cars-mpg.csv', delimiter=",", skiprows=1)
np.random.seed(1)
np.random.shuffle(invalues)
trainX, testX, trainY, testY = train_test_split(invalues[:, 1:], invalues[:, 0], test_size=0.2)


# 2 & 3. Forward Selection
# Algortihm
models = []
for features in range(1, 7):
    model, bestFeatures = [], []
    remainingX = trainX
    lowestMSE = 10000000
    for feature in range(features):
        lowestCurrentMSE = lowestMSE

        for remainingFeature in range(len(remainingX[1])):
            if (len(bestFeatures) == 0):
                currentX = remainingX[:, remainingFeature].reshape(-1, 1)
            else:
                currentX = np.c_[np.array(bestFeatures), remainingX[:, remainingFeature]]

            linReg = LinearRegression().fit(currentX, trainY)
            prediction = linReg.predict(currentX)
            currentMSE = mean_squared_error(trainY, prediction)

            if (currentMSE < lowestCurrentMSE):
                lowestCurrentMSE = currentMSE
                for c in range(trainX.shape[1]):
                    same = True
                    for i in range(remainingX.shape[0]):
                        if (remainingX[i, remainingFeature] != trainX[i, c]):
                            same = False
                            break
                    if same:
                        oneBestFeature = trainX[:, c]
                        coloumn = c
                        break

        if (lowestCurrentMSE < lowestMSE):
            if (len(bestFeatures) != 0):
                bestFeatures = np.c_[bestFeatures, oneBestFeature]
            else:
                bestFeatures = oneBestFeature

            model.append(coloumn)
            remX = np.delete(trainX, model, 1)
            lowestMSE = lowestCurrentMSE
    
    models.append(model)

# Model selection
model_results = []
for model in models:
    trainX_model = trainX[:, model]
    testX_model = testX[:, model]
    linReg = LinearRegression().fit(trainX_model, trainY)
    prediction = linReg.predict(testX_model)
    testError = mean_squared_error(testY, prediction)
    model_results.append(testError)


bestModel = models[model_results.index(np.min(model_results))]
bestModel_MSE = np.min(model_results)

print(f"The best model is: {bestModel}.")
print(f"Feature number {bestModel[0]} is the most important.")
print(f"The MSE of the best model is {bestModel_MSE}")