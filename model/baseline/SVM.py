# -*- coding: utf-8 -*-
'''
@Time    : 2020/1/28 18:14
@Author  : Zekun Cai
@File    : SVM.py
@Software: PyCharm
'''
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


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


def run_SVM(model_dir, trainSet, testSet, timestep):
    H, W, C = trainSet.shape[1], trainSet.shape[2], trainSet.shape[3]     # get shape

    # get XY features
    trainX, trainY = getXSYS(trainSet, timestep)
    testX, testY = getXSYS(testSet, timestep)
    print('Train set shape: X/Y', trainX.shape, trainY.shape)
    print('Test set shape: X/Y', testX.shape, testY.shape)

    # CV
    svr = GridSearchCV(SVR(kernel='rbf'), cv=3, n_jobs=5,
                       param_grid={
                           # 'kernel': ['rbf', 'linear', 'poly'],
                           "C": [1e0, 1e1, 1e2, 1e3]})
    # Larger C gets more training points classified correctly
    # gamma: lower value means each point has a far reach -> avoid wiggly curve to have a more linear curve

    # GridSearch for SVR: best param:{'C': 10.0, 'gamma':1}, best score:-0.008387309873577312
    svr.fit(trainX, trainY)
    print('GridSearch for SVR: best param:{}, best score:{}'.format(svr.best_params_, svr.best_score_))
    predY = svr.predict(testX)

    y_true = testY.reshape((-1, H, W, C))
    y_pred = predY.reshape((-1, H, W, C))
    print('#Positive predictions: ', y_pred[y_pred != 0].shape[0], '\n')

    return y_true, y_pred
