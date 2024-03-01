from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import CommonFunctions as cf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Read CSV
invalues = np.loadtxt('A2_datasets/microchips.csv', delimiter=",")
X = invalues[:, :2]
y = invalues[:, 2]
x1, x2 = invalues[:, 0], invalues[:, 1]

# 1. Regularized logistic regression
def regLogReg(c):
    logreg = LogisticRegression(solver='lbfgs', C=c, tol=1e-6, max_iter=10000)

    meshSize = .01
    x_min, x_max = x1.min() - 0.1, x1.max() + 0.1
    y_min, y_max = x2.min() - 0.1, x2.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, meshSize), np.arange(y_min, y_max, meshSize))

    MSE = []
    for d in range(1, 10):
        plt.subplot(3, 3, d)
        Xe_n = cf.mapFeatureOnes(x1, x2, d, False)
        logreg.fit(Xe_n, y)
        
        xy_mesh = cf.mapFeatureOnes(xx.ravel(), yy.ravel(), d, False)
        prediction = logreg.predict(xy_mesh)
        clz_mesh = prediction.reshape(xx.shape)

        mapcolor = ListedColormap(['red', 'blue'])
        plt.pcolormesh(xx, yy, clz_mesh, cmap=mapcolor, alpha=0.4)
        plt.scatter(x1, x2, c=y, marker='o', s=15, cmap=mapcolor)
        plt.title(f"Degree {d} and C = {c}")
        
        # Below is only for part 3.
        errors = 1 - cross_val_score(logreg, Xe_n, y, cv=5).mean()
        MSE.append(errors)
    return MSE


# 2. Difference with regularization.

# When using regularization, we aim to prevent overfitting. We try to avoid the model getting too complicated.
# When we change C from 10 000 -> 1, we increase the regularization. This means that we reduce the impact of specific parameters,
# making it less overfitted. When having C = 10 000, that is with low regularization (basically unregularized), we can see that 
# the higher degrees are overfitted. This effect is diminished by setting C = 1, thereby increasing the regularization.

# (All three tasks are only being run from # 3. to avoid doublettes.)


# 3. Cross-validation
MSE_reg = regLogReg(10000)
plt.figure()
MSE_unreg = regLogReg(1)
plt.figure()

plt.plot(range(1, 10), MSE_reg, color='red', label='Regularized (C = 1)')
plt.plot(range(1, 10), MSE_unreg, color='blue', label='Unregularized (C = 10 000)')
plt.xlabel('Degree (d)')
plt.ylabel('Mean errors')
plt.title('Regularized vs Unregularized Logistic Regression')
plt.legend()


plt.show()