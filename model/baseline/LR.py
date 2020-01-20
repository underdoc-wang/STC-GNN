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


def get_meta(in_dir, train_len, H, W, C, timestep):
    meta = np.genfromtxt(in_dir + '/day_info_onehot2015.csv', delimiter=',', skip_header=True)
    meta = np.delete(meta, obj=0, axis=1)     # remove timestamp column
    meta = meta.astype(int)     # convert to type int

    trainY_meta, testY_meta = meta[timestep:train_len], meta[train_len+timestep:]
    trainY_meta = np.broadcast_to(trainY_meta, (H*W*C, trainY_meta.shape[0], trainY_meta.shape[1]))
    testY_meta = np.broadcast_to(testY_meta, (H*W*C, testY_meta.shape[0], testY_meta.shape[1]))
    trainY_meta, testY_meta = trainY_meta.reshape((-1, trainY_meta.shape[-1])), testY_meta.reshape((-1, testY_meta.shape[-1]))

    return trainY_meta, testY_meta


# global
use_meta = False


def run_LR(model_dir, trainSet, testSet, timestep):
    # get shape
    H, W, C = trainSet.shape[1], trainSet.shape[2], trainSet.shape[3]
    train_len, test_len = trainSet.shape[0], testSet.shape[0]

    # get XY features
    trainX, trainY = getXSYS(trainSet, timestep)
    testX, testY = getXSYS(testSet, timestep)

    # whether or not to use metadata
    if use_meta:
        trainY_meta, testY_meta = get_meta(model_dir, train_len, H, W, C, timestep)
        trainX = np.concatenate([trainX, trainY_meta], axis=1)
        testX = np.concatenate([testX, testY_meta], axis=1)
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
    print('#Positive predictions: ', y_pred[y_pred!=0].shape[0])

    return y_true, y_pred
