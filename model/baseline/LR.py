import numpy as np
from statsmodels.api import Logit
from sklearn.linear_model import LogisticRegression


def getXSYS(data, timestep):
    XS, YS = [], []
    for i in range(data.shape[0] - timestep):
        x = data[i : i + timestep, :, :, :]
        y = data[i + timestep, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)

    # reshape XS/YS
    XS = XS.transpose((0, 4, 2, 3, 1))
    XS = XS.reshape((-1, timestep))
    YS = YS.reshape(((-1,)))

    return XS, YS


def run_LR(model_dir, trainSet, testSet, timestep):
    # get shape
    H, W, C = trainSet.shape[1], trainSet.shape[2], trainSet.shape[3]
    train_len, test_len = trainSet.shape[0], testSet.shape[0]

    # get XY features
    trainX, trainY = getXSYS(trainSet, timestep)
    testX, testY = getXSYS(testSet, timestep)

    print('Train set shape: X/Y', trainX.shape, trainY.shape)
    print('Test set shape: X/Y', testX.shape, testY.shape)

    # check data imbalance
    neg, pos = np.bincount(trainX.flatten())
    weight_ratio = neg / pos
    print('Weight ratio:', round(weight_ratio, 5))

    # logit
    logit_model = Logit(trainY, trainX)
    result = logit_model.fit()
    print(result.summary2())

    # LR
    logreg = LogisticRegression(class_weight={1:weight_ratio})     # balance pos/neg in training set
    logreg.fit(trainX, trainY)
    predY = logreg.predict(testX)

    y_true = testY.reshape((-1, H, W, C))
    y_pred = predY.reshape((-1, H, W, C))
    print('#Positive predictions: ', y_pred[y_pred!=0].shape[0], '\n')

    return y_true, y_pred
