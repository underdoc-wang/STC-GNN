# -*- coding: utf-8 -*-
'''
@Time    : 2020/1/28 17:33
@Author  : Zekun Cai
@File    : LASSO.py
@Software: PyCharm
'''
import numpy as np
from sklearn.linear_model import Lasso, LassoCV


def getXSYS(data, timestep):
    XS, YS = [], []
    for i in range(data.shape[0] - timestep):
        x = data[i: i + timestep, :, :, :]
        y = data[i + timestep, :, :, :]
        XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)

    # reshape XS/YS
    XS = XS.transpose((0, 4, 2, 3, 1))
    XS = XS.reshape((-1, timestep))
    YS = YS.reshape((-1,))

    return XS, YS


def run_LASSO(model_dir, trainSet, testSet, timestep):
    H, W, C = trainSet.shape[1], trainSet.shape[2], trainSet.shape[3]     # get shape

    # get XY features
    trainX, trainY = getXSYS(trainSet, timestep)
    testX, testY = getXSYS(testSet, timestep)
    print('Train set shape: X/Y', trainX.shape, trainY.shape)
    print('Test set shape: X/Y', testX.shape, testY.shape)

    # CV
    lassocv = LassoCV(cv=5, random_state=0).fit(trainX, trainY)
    alpha = lassocv.alpha_
    print('best alpha：' + str(alpha))

    # LASSO
    lasso = Lasso(alpha=alpha)
    lasso.fit(trainX, trainY)
    predY = lasso.predict(testX)

    y_true = testY.reshape((-1, H, W, C))
    y_pred = predY.reshape((-1, H, W, C))
    print('#Positive predictions: ', y_pred[y_pred != 0].shape[0], '\n')

    return y_true, y_pred
